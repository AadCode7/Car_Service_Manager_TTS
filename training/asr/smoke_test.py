#!/usr/bin/env python3
"""Exercise preprocessing, batching, model forward and CTC backward without data downloads."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (
    DurationBucketBatchSampler,
    apply_spec_augment,
    waveform_to_log_mel,
)
from .model import ConformerCTC, model_parameter_count, subsample_lengths


def main() -> None:
    torch.manual_seed(7)
    np.random.seed(7)

    audio_config = {
        "sample_rate": 16000,
        "n_fft": 400,
        "win_length": 400,
        "hop_length": 160,
        "n_mels": 80,
        "f_min": 20.0,
        "f_max": 7600.0,
    }
    synthetic_wave = np.sin(
        2.0 * np.pi * 220.0 * np.arange(16000, dtype=np.float32) / 16000
    ).astype(np.float32)
    log_mel = waveform_to_log_mel(synthetic_wave, 16000, audio_config)
    if log_mel.shape != (101, 80):
        raise AssertionError(f"Unexpected log-Mel shape: {log_mel.shape}")
    if not np.isfinite(log_mel).all():
        raise AssertionError("Feature extraction produced non-finite values.")

    sampler = DurationBucketBatchSampler(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        max_batch_seconds=5.0,
        max_batch_size=3,
        shuffle=False,
    )
    first_epoch = [index for batch in sampler for index in batch]
    if sorted(first_epoch) != list(range(5)):
        raise AssertionError("Duration sampler dropped or duplicated examples.")
    sampler.set_epoch(1, start_batch=1)
    remaining = [index for batch in sampler for index in batch]
    if set(remaining) & set(sampler._batches[:1][0]):
        raise AssertionError("Mid-epoch sampler resume repeated the first batch.")

    model = ConformerCTC(
        n_mels=80,
        vocab_size=64,
        d_model=96,
        n_heads=4,
        n_layers=2,
        ff_multiplier=2,
        conv_kernel_size=9,
        subsampling_channels=32,
        dropout=0.1,
    )
    model.train()
    features = torch.randn(3, 120, 80)
    feature_lengths = torch.tensor([120, 103, 77], dtype=torch.long)
    augmented = apply_spec_augment(
        features, feature_lengths, freq_masks=1, freq_width=8,
        time_masks=1, time_width=12
    )
    logits, output_lengths = model(augmented, feature_lengths)

    if logits.shape[0] != 3 or logits.shape[-1] != 64:
        raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if not torch.equal(output_lengths, subsample_lengths(feature_lengths)):
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
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("No gradients were produced.")

    # The same utterance must produce identical valid-frame logits whether it
    # is evaluated alone or padded and batched with a longer recording.
    model.eval()
    short = features[2:3, :77].clone()
    long = features[0:1, :120].clone()
    padded_batch = torch.zeros(2, 120, 80)
    padded_batch[0, :77] = short[0]
    padded_batch[1] = long[0]
    with torch.inference_mode():
        single_logits, single_lengths = model(
            short, torch.tensor([77], dtype=torch.long)
        )
        batch_logits, batch_lengths = model(
            padded_batch, torch.tensor([77, 120], dtype=torch.long)
        )
    valid_output_length = int(single_lengths[0].item())
    if int(batch_lengths[0].item()) != valid_output_length:
        raise AssertionError("Subsampled length changed with batch padding.")
    if not torch.allclose(
        single_logits[0, :valid_output_length],
        batch_logits[0, :valid_output_length],
        atol=1e-5,
        rtol=1e-5,
    ):
        raise AssertionError("Valid-frame logits depend on batch padding.")
    model.train()

    print(
        "Smoke test passed:",
        f"log_mel={log_mel.shape}",
        f"logits={tuple(logits.shape)}",
        f"loss={loss.item():.4f}",
        f"parameters={model_parameter_count(model):,}",
    )


if __name__ == "__main__":
    main()
