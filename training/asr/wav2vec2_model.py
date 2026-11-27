"""Wav2Vec 2.0 transfer-learning acoustic model for SentencePiece CTC."""
from __future__ import annotations
import torch
import torch.nn as nn
import torchaudio


class Wav2Vec2CTC(nn.Module):
    """Self-supervised LibriSpeech wav2vec2-base encoder plus a task CTC head.

    The pretrained feature extractor and lower Transformer layers are frozen
    initially. The final encoder layers and new vocabulary head are fine-tuned.
    """
    def __init__(
        self,
        vocab_size: int,
        pretrained: bool = True,
        freeze_encoder_layers: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if pretrained:
            self.encoder = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
        else:
            self.encoder = torchaudio.models.wav2vec2_base()
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(768, int(vocab_size))
        self.freeze_encoder_layers = int(freeze_encoder_layers)
        self._configure_trainable_layers()

    def _configure_trainable_layers(self):
        for parameter in self.encoder.feature_extractor.parameters():
            parameter.requires_grad = False
        layers = self.encoder.encoder.transformer.layers
        frozen = min(max(self.freeze_encoder_layers, 0), len(layers))
        for index, layer in enumerate(layers):
            for parameter in layer.parameters():
                parameter.requires_grad = index >= frozen
        # Keep the encoder's final normalization trainable with the top layers.
        for parameter in self.encoder.encoder.transformer.layer_norm.parameters():
            parameter.requires_grad = True
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

    def forward(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor):
        layer_features, output_lengths = self.encoder.extract_features(
            waveforms, lengths=waveform_lengths
        )
        hidden = layer_features[-1]
        logits = self.classifier(self.dropout(hidden))
        return logits, output_lengths

    def parameter_groups(self, encoder_lr: float, head_lr: float, weight_decay: float):
        encoder_parameters = [
            p for p in self.encoder.parameters() if p.requires_grad
        ]
        head_parameters = list(self.classifier.parameters())
        return [
            {"params": encoder_parameters, "lr": float(encoder_lr),
             "weight_decay": float(weight_decay)},
            {"params": head_parameters, "lr": float(head_lr),
             "weight_decay": float(weight_decay)},
        ]
