"""Wav2Vec 2.0 transfer-learning components for English CTC ASR.

The encoder is initialized from torchaudio's self-supervised WAV2VEC2_BASE
checkpoint. A new character-level CTC head is trained on this project's
transcripts, avoiding reuse of the pretrained model's supervised ASR head.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from fractions import Fraction

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import resample_poly
from torch.utils.data import Dataset


PRETRAINED_BUNDLE_NAME = "WAV2VEC2_BASE"
TRANSFER_MODEL_TYPE = "wav2vec2_ctc"
SAMPLE_RATE = 16000


def normalize_transfer_text(text: str) -> str:
    text = str(text).upper().strip()
    # LibriSpeech transcripts are English words. The pretrained acoustic
    # encoder is not tied to a tokenizer; this CTC head uses letters,
    # apostrophe and the word delimiter.
    import re
    text = re.sub(r"[^A-Z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class CharacterCTCTokenizer:
    """Character vocabulary compatible with the wav2vec2 CTC head."""

    blank_id = 0

    def __init__(self, labels: tuple[str, ...] | list[str] | None = None):
        self.labels = tuple(labels or (
            "-", "|", "E", "T", "A", "O", "N", "I", "H", "S",
            "R", "D", "L", "U", "M", "W", "C", "F", "G", "Y",
            "P", "B", "V", "K", "'", "X", "J", "Q", "Z",
        ))
        if self.labels[0] != "-":
            raise ValueError("Wav2Vec2 CTC blank must be label '-' at index 0.")
        self.token_to_id = {label: index for index, label in enumerate(self.labels)}
        if len(self.token_to_id) != len(self.labels):
            raise ValueError("Duplicate labels in CTC vocabulary.")

    @property
    def vocab_size(self) -> int:
        return len(self.labels)

    def encode(self, text: str) -> list[int]:
        normalized = normalize_transfer_text(text).replace(" ", "|")
        unknown = sorted(set(normalized) - set(self.token_to_id))
        if unknown:
            raise ValueError(f"Transcript contains unsupported characters: {unknown}")
        return [self.token_to_id[character] for character in normalized]

    def decode(self, token_ids: list[int]) -> str:
        characters = [
            self.labels[int(index)]
            for index in token_ids
            if 0 < int(index) < len(self.labels)
        ]
        return normalize_transfer_text("".join(characters).replace("|", " "))

    def save(self, path: str | Path) -> None:
        payload = {"labels": list(self.labels), "blank_id": self.blank_id}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharacterCTCTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("blank_id", -1)) != 0:
            raise ValueError("Unsupported transfer tokenizer blank ID.")
        return cls(payload["labels"])


def load_waveform(path: str | Path, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    waveform = audio.mean(axis=1).astype(np.float32)
    if sample_rate != target_rate:
        ratio = Fraction(target_rate, sample_rate).limit_denominator(1000)
        waveform = resample_poly(
            waveform, up=ratio.numerator, down=ratio.denominator
        ).astype(np.float32)
    # Keep the waveform scale expected by the pretrained fairseq encoder.
    return np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)


class TransferASRDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: CharacterCTCTokenizer,
        max_samples: int | None = None,
        speed_perturb: bool = False,
        speed_rates: list[float] | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self.tokenizer = tokenizer
        self.speed_perturb = bool(speed_perturb)
        self.speed_rates = speed_rates or [0.9, 1.0, 1.1]
        self.items = []
        with self.manifest_path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                item = json.loads(line)
                audio_path = Path(item["audio_filepath"])
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Audio path does not exist: {audio_path}")
                text = normalize_transfer_text(item.get("text", ""))
                if not text:
                    continue
                tokens = tokenizer.encode(text)
                if not tokens:
                    continue
                item["text"] = text.lower()
                item["tokens"] = tokens
                item["duration"] = float(item.get("duration", sf.info(str(audio_path)).duration))
                self.items.append(item)
                if max_samples and len(self.items) >= max_samples:
                    break
        if not self.items:
            raise ValueError(f"No valid entries in {self.manifest_path}")
        self.durations = [max(0.1, float(item["duration"])) for item in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        waveform = load_waveform(item["audio_filepath"])
        if self.speed_perturb and len(self.speed_rates) > 1:
            speed = random.choice(self.speed_rates)
            if abs(speed - 1.0) > 1e-6:
                ratio = Fraction(1.0 / float(speed)).limit_denominator(100)
                waveform = resample_poly(
                    waveform, up=ratio.numerator, down=ratio.denominator
                ).astype(np.float32)
        return {
            "id": str(item.get("id", index)),
            "waveform": torch.from_numpy(waveform),
            "tokens": torch.tensor(item["tokens"], dtype=torch.long),
            "text": item["text"],
            "duration": float(len(waveform) / SAMPLE_RATE),
        }


def collate_transfer(batch: list[dict]) -> dict:
    lengths = torch.tensor([item["waveform"].numel() for item in batch], dtype=torch.long)
    max_samples = int(lengths.max().item())
    waveforms = torch.zeros(len(batch), max_samples, dtype=torch.float32)
    for index, item in enumerate(batch):
        waveforms[index, : lengths[index]] = item["waveform"]
    target_lengths = torch.tensor(
        [item["tokens"].numel() for item in batch], dtype=torch.long
    )
    return {
        "ids": [item["id"] for item in batch],
        "texts": [item["text"] for item in batch],
        "durations": [item["duration"] for item in batch],
        "waveforms": waveforms,
        "waveform_lengths": lengths,
        "targets": torch.cat([item["tokens"] for item in batch]),
        "target_lengths": target_lengths,
    }


class Wav2Vec2CTC(nn.Module):
    """Self-supervised Wav2Vec2 encoder with a newly initialized CTC head."""

    def __init__(self, vocab_size: int, pretrained: bool = True):
        super().__init__()
        try:
            import torchaudio
        except Exception as error:
            raise RuntimeError(
                "torchaudio is required for transfer learning and must match "
                "the installed PyTorch version. Install the matching torchaudio "
                "build from https://pytorch.org/get-started/locally/."
            ) from error

        self.pretrained_bundle = PRETRAINED_BUNDLE_NAME
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        if int(bundle.sample_rate) != SAMPLE_RATE:
            raise RuntimeError(f"Unexpected pretrained sample rate: {bundle.sample_rate}")
        self.backbone = bundle.get_model() if pretrained else bundle.get_model()
        hidden_size = 768
        self.ctc_head = nn.Linear(hidden_size, int(vocab_size))
        self.blank_id = 0

    def set_epoch_trainability(
        self, epoch: int, head_warmup_epochs: int = 1,
        freeze_feature_extractor: bool = True,
    ) -> None:
        # First train only the new CTC head. Then fine-tune the pretrained
        # Transformer while keeping the low-level convolutional frontend fixed.
        backbone_trainable = int(epoch) > int(head_warmup_epochs)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(backbone_trainable)
        for parameter in self.ctc_head.parameters():
            parameter.requires_grad_(True)
        if freeze_feature_extractor:
            for parameter in self.backbone.feature_extractor.parameters():
                parameter.requires_grad_(False)

    def forward(
        self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_features, output_lengths = self.backbone.extract_features(
            waveforms, waveform_lengths
        )
        hidden = layer_features[-1]
        return self.ctc_head(hidden), output_lengths
