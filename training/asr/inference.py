"""Load and run either the original Conformer or transfer-learned Wav2Vec2 ASR."""

from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import resample_poly

from .data import waveform_to_log_mel
from .metrics import greedy_decode
from .model import ConformerCTC, validate_checkpoint_architecture
from .wav2vec2_model import Wav2Vec2CTC
from scipy.signal import resample_poly


def load_asr_model(
    artifact_dir: str | Path,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    checkpoint_path = artifact_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint missing: {checkpoint_path}")

    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=selected_device, weights_only=False
    )

    if checkpoint.get("model_type") == "wav2vec2_ctc":
        from .transfer import CharacterCTCTokenizer, Wav2Vec2CTC

        tokenizer_path = artifact_dir / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Transfer tokenizer missing: {tokenizer_path}")
        tokenizer = CharacterCTCTokenizer.load(tokenizer_path)
        model = Wav2Vec2CTC(tokenizer.vocab_size, pretrained=False).to(selected_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return {
            "model": model,
            "tokenizer": tokenizer,
            "model_type": "wav2vec2_ctc",
            "audio_config": {"sample_rate": 16000},
            "device": selected_device,
            "epoch": int(checkpoint.get("epoch", -1)),
        }

    from .tokenizer import ASRTokenizer

    validate_checkpoint_architecture(checkpoint)
    tokenizer_path = artifact_dir / "tokenizer.model"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer model missing: {tokenizer_path}")
    model_config = checkpoint["model_config"]
    if model_config.get("backend") == "wav2vec2":
        model = Wav2Vec2CTC(
            vocab_size=model_config["vocab_size"],
            pretrained=False,
            freeze_encoder_layers=model_config.get("freeze_encoder_layers", 8),
            dropout=model_config.get("dropout", 0.1),
        ).to(selected_device)
    else:
        model = ConformerCTC(**model_config).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "tokenizer": ASRTokenizer(tokenizer_path),
        "model_type": "conformer_ctc",
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
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("Expected non-empty mono or multi-channel audio.")
    duration = float(waveform.size / sample_rate)
    device = bundle["device"]

    if bundle.get("model_type") == "wav2vec2_ctc":
        if sample_rate != 16000:
            ratio = Fraction(16000, sample_rate).limit_denominator(1000)
            waveform = resample_poly(
                waveform, up=ratio.numerator, down=ratio.denominator
            ).astype(np.float32)
        audio = torch.from_numpy(waveform.copy()).unsqueeze(0).to(device)
        lengths = torch.tensor([audio.shape[1]], dtype=torch.long, device=device)
        logits, output_lengths = bundle["model"](audio, lengths)
    else:
        features = waveform_to_log_mel(
            waveform, sample_rate, bundle["audio_config"]
        )
        feature_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
        feature_lengths = torch.tensor(
            [features.shape[0]], dtype=torch.long, device=device
        )
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
        "model_type": bundle.get("model_type", "conformer_ctc"),
    }
