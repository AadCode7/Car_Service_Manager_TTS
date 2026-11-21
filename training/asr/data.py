"""Audio preparation, log-Mel features and duration-bucketed batches."""

from __future__ import annotations

import json
import math
import random
from functools import lru_cache
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .tokenizer import ASRTokenizer


@lru_cache(maxsize=16)
def build_mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float
) -> np.ndarray:
    """Build an HTK-style triangular mel filterbank without librosa."""
    def hz_to_mel(value):
        return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)

    def mel_to_hz(value):
        return 700.0 * (10.0 ** (value / 2595.0) - 1.0)

    f_max = min(float(f_max), sample_rate / 2.0)
    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    fft_frequencies = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    filters = np.zeros((n_mels, len(fft_frequencies)), dtype=np.float32)

    for index in range(n_mels):
        left, centre, right = hz_points[index : index + 3]
        rising = (fft_frequencies - left) / max(centre - left, 1e-8)
        falling = (right - fft_frequencies) / max(right - centre, 1e-8)
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))

    enorm = 2.0 / np.maximum(
        hz_points[2 : n_mels + 2] - hz_points[:n_mels], 1e-8
    )
    filters *= enorm[:, None].astype(np.float32)
    return filters


def waveform_to_log_mel(
    waveform: np.ndarray, sample_rate: int, audio_config: dict[str, Any]
) -> np.ndarray:
    """Convert one mono waveform to per-utterance CMVN-normalised log-Mel frames."""
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    target_rate = int(audio_config["sample_rate"])
    if sample_rate != target_rate:
        ratio = Fraction(target_rate, sample_rate).limit_denominator(1000)
        waveform = resample_poly(
            waveform, up=ratio.numerator, down=ratio.denominator
        ).astype(np.float32)
        sample_rate = target_rate

    n_fft = int(audio_config["n_fft"])
    win_length = int(audio_config["win_length"])
    hop_length = int(audio_config["hop_length"])
    n_mels = int(audio_config["n_mels"])
    if waveform.size < win_length:
        waveform = np.pad(waveform, (0, win_length - waveform.size))

    pad = n_fft // 2
    padded = np.pad(
        waveform, (pad, pad), mode="reflect" if waveform.size > 1 else "constant"
    )
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length]
    window = np.hanning(win_length + 1)[:-1].astype(np.float32)
    if win_length != n_fft:
        centred_window = np.zeros(n_fft, dtype=np.float32)
        start = (n_fft - win_length) // 2
        centred_window[start : start + win_length] = window
        window = centred_window

    spectrum = np.fft.rfft(frames * window[None, :], n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    mel_filter = build_mel_filterbank(
        sample_rate, n_fft, n_mels,
        float(audio_config["f_min"]), float(audio_config["f_max"])
    )
    mel_energy = power @ mel_filter.T
    log_mel = np.log(np.maximum(mel_energy, 1e-10)).astype(np.float32)

    mean = log_mel.mean(axis=0, keepdims=True)
    std = log_mel.std(axis=0, keepdims=True)
    log_mel = (log_mel - mean) / np.maximum(std, 1e-5)
    return np.nan_to_num(log_mel, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def read_audio(path: str | Path, target_sample_rate: int) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    waveform = audio.mean(axis=1)
    if sample_rate != target_sample_rate:
        ratio = Fraction(target_sample_rate, sample_rate).limit_denominator(1000)
        waveform = resample_poly(
            waveform, up=ratio.numerator, down=ratio.denominator
        ).astype(np.float32)
        sample_rate = target_sample_rate
    return waveform.astype(np.float32), sample_rate


class ASRDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: ASRTokenizer,
        audio_config: dict[str, Any],
        max_samples: int | None = None,
        speed_perturb: bool = False,
        speed_rates: list[float] | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self.tokenizer = tokenizer
        self.audio_config = audio_config
        self.target_sample_rate = int(audio_config["sample_rate"])
        self.speed_perturb = speed_perturb
        self.speed_rates = speed_rates or [0.9, 1.0, 1.1]
        self.items: list[dict[str, Any]] = []

        with self.manifest_path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                item = json.loads(line)
                audio_path = Path(item["audio_filepath"])
                if not audio_path.is_file():
                    raise FileNotFoundError(
                        f"Audio path from manifest does not exist: {audio_path}"
                    )
                if not item.get("text", "").strip():
                    continue
                item["duration"] = float(
                    item.get("duration", sf.info(str(audio_path)).duration)
                )
                self.items.append(item)
                if max_samples and len(self.items) >= max_samples:
                    break

        if not self.items:
            raise ValueError(f"No valid audio entries found in {self.manifest_path}")
        self.durations = [max(0.1, float(item["duration"])) for item in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        waveform, sample_rate = read_audio(
            item["audio_filepath"], self.target_sample_rate
        )
        if self.speed_perturb and len(self.speed_rates) > 1:
            speed = random.choice(self.speed_rates)
            if not math.isclose(speed, 1.0):
                ratio = Fraction(1.0 / float(speed)).limit_denominator(100)
                waveform = resample_poly(
                    waveform, up=ratio.numerator, down=ratio.denominator
                ).astype(np.float32)

        features = waveform_to_log_mel(waveform, sample_rate, self.audio_config)
        token_ids = self.tokenizer.encode(item["text"])
        if not token_ids:
            raise ValueError(f"Transcript produced no tokens: {item.get('id', index)}")
        return {
            "id": str(item.get("id", index)),
            "features": torch.from_numpy(features),
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "text": item["text"],
            "duration": float(len(waveform) / self.target_sample_rate),
        }


def collate_asr(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_frames = max(item["features"].shape[0] for item in batch)
    feature_dim = batch[0]["features"].shape[1]
    features = torch.zeros(batch_size, max_frames, feature_dim, dtype=torch.float32)
    feature_lengths = torch.empty(batch_size, dtype=torch.long)
    token_lengths = torch.empty(batch_size, dtype=torch.long)

    for index, item in enumerate(batch):
        frame_count = item["features"].shape[0]
        features[index, :frame_count] = item["features"]
        token_lengths[index] = item["tokens"].numel()
        feature_lengths[index] = frame_count

    return {
        "ids": [item["id"] for item in batch],
        "texts": [item["text"] for item in batch],
        "durations": [item["duration"] for item in batch],
        "features": features,
        "feature_lengths": feature_lengths,
        "targets": torch.cat([item["tokens"] for item in batch]),
        "target_lengths": token_lengths,
    }


class DurationBucketBatchSampler:
    """Shuffle duration buckets and cap each batch by total audio seconds."""

    def __init__(
        self,
        durations: list[float],
        max_batch_seconds: float,
        max_batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        bucket_multiplier: int = 80,
    ):
        self.durations = durations
        self.max_batch_seconds = float(max_batch_seconds)
        self.max_batch_size = int(max_batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_multiplier = bucket_multiplier
        self.epoch = 0
        self.start_batch = 0
        self._batches: list[list[int]] | None = None

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        self.epoch = int(epoch)
        self.start_batch = max(0, int(start_batch))
        self._batches = None

    def _make_batches(self) -> list[list[int]]:
        indices = list(range(len(self.durations)))
        rng = random.Random(self.seed + self.epoch)
        indices.sort(key=lambda index: self.durations[index])
        bucket_size = max(
            self.max_batch_size * self.bucket_multiplier, self.max_batch_size
        )
        buckets = [
            indices[start : start + bucket_size]
            for start in range(0, len(indices), bucket_size)
        ]
        batches: list[list[int]] = []

        for bucket in buckets:
            if self.shuffle:
                rng.shuffle(bucket)
            current: list[int] = []
            current_seconds = 0.0
            for index in bucket:
                duration = self.durations[index]
                exceeds = current and (
                    current_seconds + duration > self.max_batch_seconds
                    or len(current) >= self.max_batch_size
                )
                if exceeds:
                    batches.append(current)
                    current = []
                    current_seconds = 0.0
                current.append(index)
                current_seconds += duration
            if current:
                batches.append(current)

        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        if self._batches is None:
            self._batches = self._make_batches()
        yield from self._batches[self.start_batch :]

    def __len__(self) -> int:
        if self._batches is None:
            self._batches = self._make_batches()
        return max(0, len(self._batches) - self.start_batch)


def apply_spec_augment(
    features: torch.Tensor,
    feature_lengths: torch.Tensor,
    freq_masks: int = 2,
    freq_width: int = 15,
    time_masks: int = 10,
    time_width: int = 40,
) -> torch.Tensor:
    """Apply SpecAugment time/frequency masking to a padded B x T x F tensor."""
    augmented = features.clone()
    batch_size, _, num_mels = augmented.shape
    for batch_index in range(batch_size):
        valid_time = int(feature_lengths[batch_index].item())
        for _ in range(freq_masks):
            width = random.randint(0, min(freq_width, num_mels))
            if width:
                start = random.randint(0, num_mels - width)
                augmented[batch_index, :, start : start + width] = 0.0
        for _ in range(time_masks):
            width = random.randint(0, min(time_width, valid_time))
            if width:
                start = random.randint(0, max(0, valid_time - width))
                augmented[batch_index, start : start + width, :] = 0.0
    return augmented
