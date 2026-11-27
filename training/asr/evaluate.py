#!/usr/bin/env python3
"""Evaluate a trained checkpoint on a held-out manifest and save predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ASRDataset, collate_asr
from .transfer import CharacterCTCTokenizer, TransferASRDataset, Wav2Vec2CTC, collate_transfer
from .metrics import (
    corpus_error_rates,
    ctc_prefix_beam_search,
    greedy_decode,
    sentence_error_rates,
)
from .model import ConformerCTC, validate_checkpoint_architecture
from .tokenizer import ASRTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/test-clean.jsonl"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/asr"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=0)
    args = parser.parse_args()

    run_dir = resolve_path(args.run_dir)
    checkpoint_path = run_dir / "best.pt"
    tokenizer_path = run_dir / "tokenizer.model"
    if not checkpoint_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Need both {checkpoint_path} and {tokenizer_path}. Train the model first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    is_transfer = checkpoint.get("model_type") == "wav2vec2_ctc"

    if is_transfer:
        tokenizer_path = run_dir / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Transfer tokenizer missing: {tokenizer_path}")
        tokenizer = CharacterCTCTokenizer.load(tokenizer_path)
        model = Wav2Vec2CTC(tokenizer.vocab_size, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        audio_config = {"sample_rate": 16000}
        dataset = TransferASRDataset(
            resolve_path(args.manifest), tokenizer,
            max_samples=args.max_samples, speed_perturb=False
        )
        collate_fn = collate_transfer
        eval_batch_size = min(args.batch_size, 2)
    else:
        validate_checkpoint_architecture(checkpoint)
        model_config = checkpoint["model_config"]
        audio_config = checkpoint["audio_config"]
        tokenizer_path = run_dir / "tokenizer.model"
        tokenizer = ASRTokenizer(tokenizer_path)
        model = ConformerCTC(**model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        dataset = ASRDataset(
            resolve_path(args.manifest), tokenizer, audio_config,
            max_samples=args.max_samples, speed_perturb=False
        )
        collate_fn = collate_asr
        eval_batch_size = args.batch_size

    model.eval()
    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    loader = DataLoader(
        dataset, batch_size=eval_batch_size, shuffle=False, num_workers=2,
        pin_memory=device.type == "cuda", collate_fn=collate_fn
    )

    total_loss = 0.0
    total_samples = 0
    references: list[str] = []
    hypotheses: list[str] = []
    rows: list[dict[str, object]] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Testing"):
            if is_transfer:
                features = batch["waveforms"].to(device, non_blocking=True)
                feature_lengths = batch["waveform_lengths"].to(device, non_blocking=True)
            else:
                features = batch["features"].to(device, non_blocking=True)
                feature_lengths = batch["feature_lengths"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            logits, output_lengths = model(features, feature_lengths)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            loss = criterion(
                log_probs.transpose(0, 1),
                targets,
                output_lengths.cpu(),
                batch["target_lengths"].cpu(),
            )
            batch_size = len(batch["texts"])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            greedy_hypotheses = greedy_decode(
                logits.float(), output_lengths, tokenizer
            )
            if args.beam_width > 0:
                decoded = []
                for index, output_length in enumerate(output_lengths.tolist()):
                    decoded.append(
                        ctc_prefix_beam_search(
                            log_probs[index, :output_length],
                            tokenizer,
                            beam_width=args.beam_width,
                        )
                    )
                batch_hypotheses = decoded
            else:
                batch_hypotheses = greedy_hypotheses

            for utterance_id, duration, reference, hypothesis in zip(
                batch["ids"], batch["durations"], batch["texts"], batch_hypotheses
            ):
                sentence_wer, sentence_cer = sentence_error_rates(reference, hypothesis)
                rows.append({
                    "id": utterance_id,
                    "duration_seconds": duration,
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "wer": sentence_wer,
                    "cer": sentence_cer,
                })
                references.append(reference)
                hypotheses.append(hypothesis)

    metrics = corpus_error_rates(references, hypotheses)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))
    metrics["decoder"] = "ctc_prefix_beam" if args.beam_width > 0 else "greedy"
    metrics["beam_width"] = int(args.beam_width)
    metrics["manifest"] = str(args.manifest)
    metrics_path = run_dir / "test_metrics.json"
    predictions_path = run_dir / "test_predictions.csv"

    with metrics_path.open("w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)
    with predictions_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(metrics, indent=2))
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
