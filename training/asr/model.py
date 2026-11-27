"""Random-initialised Conformer-CTC acoustic model.

Research-informed components:
- FastConformer-inspired 8x depthwise-separable convolutional subsampling.
- Reduced 9-frame Conformer convolution kernel for compute efficiency.
- Macaron-style dual feed-forward blocks.
- Rotary positional embeddings for self-attention.
- PyTorch scaled-dot-product attention (FlashAttention kernels are selected
  when the installed PyTorch/CUDA hardware supports them).
- GLU and depthwise convolution modules for local acoustic modelling.

This is a compact, reproducible research baseline, not a claim of matching
large pretrained speech foundation models.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_ARCHITECTURE_VERSION = 3


def validate_checkpoint_architecture(checkpoint: dict) -> None:
    version = int(checkpoint.get("model_architecture_version", 1))
    if version != MODEL_ARCHITECTURE_VERSION:
        raise RuntimeError(
            f"Checkpoint architecture version {version} is incompatible with "
            f"current model version {MODEL_ARCHITECTURE_VERSION}. This model "
            "uses a different acoustic encoder. Start a new run with the configured backend. "
            "Do not resume an older BatchNorm checkpoint."
        )


def subsample_lengths(lengths: torch.Tensor) -> torch.Tensor:
    """Output lengths after three stride-2, kernel-3, padding-1 stages (8x)."""
    for _ in range(3):
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
    return lengths


class DepthwiseSeparableSubsamplingBlock(nn.Module):
    """Stride-2 depthwise convolution with per-position channel normalization."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        # Normalize channels independently at each time/frequency position.
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pointwise(self.depthwise(x))
        lengths = subsample_lengths(lengths)
        valid = torch.arange(x.shape[2], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        x = self.norm(x.permute(0, 2, 3, 1))
        x = self.activation(x)
        x = x.masked_fill(~valid[:, :, None, None], 0.0)
        return x.permute(0, 3, 1, 2).contiguous(), lengths


class ConvSubsampling(nn.Module):
    """Padding-aware 8x depthwise-separable convolutional frontend."""

    def __init__(self, input_dim: int, channels: int, d_model: int):
        super().__init__()
        self.conv = nn.ModuleList(
            [
                DepthwiseSeparableSubsamplingBlock(1, channels),
                DepthwiseSeparableSubsamplingBlock(channels, channels),
                DepthwiseSeparableSubsamplingBlock(channels, channels),
            ]
        )
        frequency_after_subsampling = input_dim
        for _ in range(3):
            frequency_after_subsampling = (frequency_after_subsampling + 1) // 2
        self.projection = nn.Linear(
            channels * frequency_after_subsampling, d_model
        )

    def forward(
        self, features: torch.Tensor, feature_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = features.unsqueeze(1)
        output_lengths = feature_lengths
        for block in self.conv:
            hidden, output_lengths = block(hidden, output_lengths)

        batch, channels, time, frequency = hidden.shape
        hidden = hidden.transpose(1, 2).contiguous().view(
            batch, time, channels * frequency
        )
        hidden = self.projection(hidden)
        valid = torch.arange(time, device=hidden.device).unsqueeze(0) < output_lengths.unsqueeze(1)
        hidden = hidden.masked_fill(~valid.unsqueeze(-1), 0.0)
        return hidden, output_lengths


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, multiplier: int, dropout: float):
        super().__init__()
        hidden = d_model * multiplier
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = F.silu(self.linear1(x))
        x = self.dropout(x)
        return self.dropout(self.linear2(x))


class RotarySelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2:
            raise ValueError("Rotary attention requires an even head dimension")
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = float(dropout)

    def _apply_rope(self, tensor: torch.Tensor) -> torch.Tensor:
        sequence_length = tensor.shape[-2]
        half_dim = self.head_dim // 2
        inverse_frequency = torch.exp(
            torch.arange(half_dim, device=tensor.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half_dim, 1))
        )
        positions = torch.arange(
            sequence_length, device=tensor.device, dtype=torch.float32
        )
        angles = torch.outer(positions, inverse_frequency)
        cosine = angles.cos().to(tensor.dtype)[None, None, :, :]
        sine = angles.sin().to(tensor.dtype)[None, None, :, :]
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        )
        return rotated.flatten(-2)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, time, dimension = x.shape
        qkv = self.qkv(x).view(
            batch, time, 3, self.n_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query = self._apply_rope(query)
        key = self._apply_rope(key)

        # In SDPA boolean masks, True means the key position is visible.
        attention_mask = valid_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch, time, dimension
        )
        return self.output(attended)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Conformer convolution kernel_size must be odd")
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model
        )
        # Per-time-step LayerNorm avoids statistics over padded frames.
        self.channel_norm = nn.LayerNorm(d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        time_mask = valid_mask[:, None, :]
        feature_mask = valid_mask.unsqueeze(-1)
        x = self.norm(x).masked_fill(~feature_mask, 0.0).transpose(1, 2)
        x = F.glu(self.pointwise_in(x), dim=1).masked_fill(~time_mask, 0.0)
        x = self.depthwise(x).transpose(1, 2)
        x = self.channel_norm(x).masked_fill(~feature_mask, 0.0)
        x = F.silu(x)
        x = self.pointwise_out(x.transpose(1, 2))
        return self.dropout(x.transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, ff_multiplier: int,
        conv_kernel_size: int, dropout: float
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ff_multiplier, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = RotarySelfAttention(d_model, n_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, ff_multiplier, dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        query_mask = valid_mask.unsqueeze(-1)
        x = (x + 0.5 * self.ffn1(x)).masked_fill(~query_mask, 0.0)
        x = (x + self.dropout(self.attention(self.attn_norm(x), valid_mask))).masked_fill(
            ~query_mask, 0.0
        )
        x = (x + self.conv(x, valid_mask)).masked_fill(~query_mask, 0.0)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x).masked_fill(~query_mask, 0.0)


class ConformerCTC(nn.Module):
    def __init__(
        self,
        n_mels: int,
        vocab_size: int,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 10,
        ff_multiplier: int = 4,
        conv_kernel_size: int = 31,
        subsampling_channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.subsampling = ConvSubsampling(n_mels, subsampling_channels, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model, n_heads, ff_multiplier, conv_kernel_size, dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, vocab_size)
        self.blank_id = 0

    def forward(
        self, features: torch.Tensor, feature_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, output_lengths = self.subsampling(features, feature_lengths)
        output_lengths = output_lengths.clamp(max=hidden.shape[1])
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        valid_mask = positions.unsqueeze(0) < output_lengths.unsqueeze(1)
        hidden = self.input_dropout(hidden).masked_fill(
            ~valid_mask.unsqueeze(-1), 0.0
        )
        for block in self.blocks:
            hidden = block(hidden, valid_mask)
        return self.classifier(self.final_norm(hidden)), output_lengths


def model_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
