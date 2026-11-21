"""Random-initialised Conformer-CTC acoustic model.

Research-informed components:
- Four-times convolutional feature subsampling.
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


def subsample_lengths(lengths: torch.Tensor) -> torch.Tensor:
    """Output lengths after two stride-2, kernel-3, padding-1 convolutions."""
    first = torch.div(lengths + 1, 2, rounding_mode="floor")
    return torch.div(first + 1, 2, rounding_mode="floor")


class ConvSubsampling(nn.Module):
    def __init__(self, input_dim: int, channels: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        freq_after_first = (input_dim + 1) // 2
        freq_after_second = (freq_after_first + 1) // 2
        self.projection = nn.Linear(channels * freq_after_second, d_model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(features.unsqueeze(1))
        batch, channels, time, frequency = hidden.shape
        hidden = hidden.transpose(1, 2).contiguous().view(
            batch, time, channels * frequency
        )
        return self.projection(hidden)


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
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).transpose(1, 2)
        x = F.glu(self.pointwise_in(x), dim=1)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise_out(x)
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
        x = (x + self.conv(x)).masked_fill(~query_mask, 0.0)
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
        hidden = self.subsampling(features)
        output_lengths = subsample_lengths(feature_lengths).clamp(max=hidden.shape[1])
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
