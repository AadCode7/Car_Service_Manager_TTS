"""Load trained artifacts and transcribe a single audio waveform."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import waveform_to_log_mel
from .metrics import greedy_decode
from .model import ConformerCTC
from .tokenizer import ASRTokenizer


def load_asr_model(
    artifact_dir: str | Path,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    checkpoint_path = artifact_dir / "best.pt"
    tokenizer_path = artifact_dir / "tokenizer.model"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint missing: {checkpoint_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer model missing: {tokenizer_path}")

    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=selected_device, weights_only=False
    )
    model = ConformerCTC(**checkpoint["model_config"]).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = ASRTokenizer(tokenizer_path)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "audio_config": checkpoint["audio_config"],
        "device": selected_device,
        "epoch": int(checkpoint.get("epoch", -1)),
    }


@torch.inference_mode()
def transcribe_audio_array(
    waveform: np.ndarray,
    sample_rate: int,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        # soundfile returns frames x channels.
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("Expected non-empty mono or multi-channel audio.")
    duration = float(waveform.size / sample_rate)

    features = waveform_to_log_mel(
        waveform, sample_rate, bundle["audio_config"]
    )
    device = bundle["device"]
    feature_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    feature_lengths = torch.tensor([features.shape[0]], dtype=torch.long, device=device)
    logits, output_lengths = bundle["model"](feature_tensor, feature_lengths)
    hypothesis = greedy_decode(
        logits.float(), output_lengths, bundle["tokenizer"]
    )[0]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "text": hypothesis,
        "duration_seconds": duration,
        "inference_ms": elapsed_ms,
        "model_epoch": bundle["epoch"],
    }
