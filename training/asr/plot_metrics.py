#!/usr/bin/env python3
"""Create training/evaluation charts from history.csv and test_predictions.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/asr"))
    args = parser.parse_args()
    run_dir = resolve_path(args.run_dir)
    history_path = run_dir / "history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"Training history not found: {history_path}")

    output_dir = run_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(history_path)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(history["epoch"], history["train_loss"], label="Train CTC loss")
    axis.plot(history["epoch"], history["valid_loss"], label="Validation CTC loss")
    axis.set(title="CTC loss by epoch", xlabel="Epoch", ylabel="Loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curves.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(history["epoch"], history["valid_wer"], label="Validation WER")
    axis.plot(history["epoch"], history["valid_cer"], label="Validation CER")
    axis.set(title="Validation error rates", xlabel="Epoch", ylabel="Error rate")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "error_rates.png", dpi=160)
    plt.close(figure)

    if "learning_rate" in history.columns:
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.plot(history["epoch"], history["learning_rate"])
        axis.set(title="Learning-rate schedule", xlabel="Epoch", ylabel="Learning rate")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "learning_rate.png", dpi=160)
        plt.close(figure)

    time_column = "epoch_seconds" if "epoch_seconds" in history.columns else (
        "seconds" if "seconds" in history.columns else None
    )
    if time_column:
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.plot(history["epoch"], history[time_column] / 60.0)
        axis.set(title="Training time per epoch", xlabel="Epoch", ylabel="Minutes")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "epoch_time.png", dpi=160)
        plt.close(figure)

    predictions_path = run_dir / "test_predictions.csv"
    if predictions_path.is_file():
        predictions = pd.read_csv(predictions_path)
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.hist(predictions["wer"].clip(upper=2.0), bins=30)
        axis.set(title="Utterance-level WER distribution", xlabel="WER (clipped at 2)", ylabel="Utterances")
        axis.grid(True, axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "utterance_wer_distribution.png", dpi=160)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(9, 5))
        axis.scatter(predictions["duration_seconds"], predictions["wer"], alpha=0.35)
        axis.set(title="WER versus utterance duration", xlabel="Duration (seconds)", ylabel="WER")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "wer_vs_duration.png", dpi=160)
        plt.close(figure)

    print(f"Charts saved in {output_dir}")


if __name__ == "__main__":
    main()
