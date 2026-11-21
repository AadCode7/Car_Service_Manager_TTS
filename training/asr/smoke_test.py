#!/usr/bin/env python3
"""Fast model/data-path smoke test. Does not download data or run training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ConformerCTC, model_parameter_count, subsample_lengths


def main() -> None:
    torch.manual_seed(7)
    model = ConformerCTC(
        n_mels=80,
        vocab_size=64,
        d_model=96,
        n_heads=4,
        n_layers=2,
        ff_multiplier=2,
        conv_kernel_size=15,
        subsampling_channels=32,
        dropout=0.1,
    )
    model.train()
    features = torch.randn(3, 120, 80)
    feature_lengths = torch.tensor([120, 103, 77], dtype=torch.long)
    logits, output_lengths = model(features, feature_lengths)

    if logits.shape[0] != 3 or logits.shape[-1] != 64:
        raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)}")
    expected_lengths = subsample_lengths(feature_lengths)
    if not torch.equal(output_lengths, expected_lengths):
        raise AssertionError("Encoder output lengths do not match convolutional subsampling.")

    targets = torch.tensor([3, 4, 5, 6, 7, 8], dtype=torch.long)
    target_lengths = torch.tensor([2, 2, 2], dtype=torch.long)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    loss = loss_fn(
        F.log_softmax(logits.float(), dim=-1).transpose(0, 1),
        targets,
        output_lengths,
        target_lengths,
    )
    loss.backward()
    if not torch.isfinite(loss):
        raise AssertionError("CTC loss is not finite.")
    print(
        "Smoke test passed:",
        f"logits={tuple(logits.shape)}",
        f"loss={loss.item():.4f}",
        f"parameters={model_parameter_count(model):,}",
    )


if __name__ == "__main__":
    main()
