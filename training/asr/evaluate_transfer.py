#!/usr/bin/env python3
"""Evaluate a Wav2Vec2 transfer-learning checkpoint on a manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import corpus_error_rates, greedy_decode, sentence_error_rates
from .transfer import (
    CharacterCTCTokenizer, TransferASRDataset, Wav2Vec2CTC, collate_transfer,
)

ROOT = Path(__file__).resolve().parents[2]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/test-clean.jsonl"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/asr-transfer"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    run_dir = resolve(args.run_dir)
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing transfer checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "wav2vec2_ctc":
        raise RuntimeError("The selected checkpoint is not a Wav2Vec2 CTC checkpoint.")
    tokenizer = CharacterCTCTokenizer.load(run_dir / "tokenizer.json")
    model = Wav2Vec2CTC(tokenizer.vocab_size, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = TransferASRDataset(
        resolve(args.manifest), tokenizer, max_samples=args.max_samples,
        speed_perturb=False,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
        pin_memory=device.type == "cuda", collate_fn=collate_transfer,
    )
    refs, hyps, rows = [], [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Transfer evaluation"):
            waveforms = batch["waveforms"].to(device, non_blocking=True)
            lengths = batch["waveform_lengths"].to(device, non_blocking=True)
            logits, output_lengths = model(waveforms, lengths)
            decoded = greedy_decode(logits.float(), output_lengths, tokenizer)
            for utterance_id, duration, reference, hypothesis in zip(
                batch["ids"], batch["durations"], batch["texts"], decoded
            ):
                wer, cer = sentence_error_rates(reference, hypothesis)
                rows.append({
                    "id": utterance_id, "duration_seconds": duration,
                    "reference": reference, "hypothesis": hypothesis,
                    "wer": wer, "cer": cer,
                })
                refs.append(reference)
                hyps.append(hypothesis)

    metrics = corpus_error_rates(refs, hyps)
    metrics["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))
    metrics["manifest"] = str(args.manifest)
    (run_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
