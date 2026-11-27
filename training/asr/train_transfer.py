#!/usr/bin/env python3
"""Fine-tune a pretrained Wav2Vec2 encoder on the project's ASR manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DurationBucketBatchSampler
from .metrics import corpus_error_rates, greedy_decode
from .transfer import (
    CharacterCTCTokenizer,
    TransferASRDataset,
    Wav2Vec2CTC,
    collate_transfer,
)

ROOT = Path(__file__).resolve().parents[2]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("training/asr/config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/asr-transfer"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    train_cfg = config["training"]
    data_cfg = config["data"]
    output_dir = resolve(args.output_dir)
    epochs = int(args.epochs or train_cfg.get("transfer_epochs", 12))
    max_train = args.max_train_samples or data_cfg.get("max_train_samples")
    max_valid = args.max_valid_samples or data_cfg.get("max_valid_samples")
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = CharacterCTCTokenizer()
    train_set = TransferASRDataset(
        resolve(data_cfg["train_manifest"]), tokenizer,
        max_samples=max_train,
        speed_perturb=bool(train_cfg.get("speed_perturb", True)),
        speed_rates=train_cfg.get("speed_perturb_rates", [0.9, 1.0, 1.1]),
    )
    valid_set = TransferASRDataset(
        resolve(data_cfg["valid_manifest"]), tokenizer,
        max_samples=max_valid, speed_perturb=False,
    )
    batch_sampler = DurationBucketBatchSampler(
        train_set.durations,
        max_batch_seconds=float(train_cfg.get("transfer_max_batch_seconds", 24.0)),
        max_batch_size=int(train_cfg.get("transfer_max_batch_size", 4)),
        shuffle=True, seed=seed,
    )
    workers = int(data_cfg.get("num_workers", 2))
    train_loader = DataLoader(
        train_set, batch_sampler=batch_sampler, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        collate_fn=collate_transfer,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=int(train_cfg.get("transfer_valid_batch_size", 2)),
        shuffle=False, num_workers=workers, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0, collate_fn=collate_transfer,
    )

    model = Wav2Vec2CTC(tokenizer.vocab_size, pretrained=True).to(device)
    print("Pretrained encoder: torchaudio WAV2VEC2_BASE (self-supervised)")
    print("CTC vocabulary:", tokenizer.vocab_size)
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

    # Distinct learning rates: preserve the pretrained representation while
    # allowing the newly initialized CTC head to learn quickly.
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": float(train_cfg.get("transfer_encoder_lr", 1e-5))},
            {"params": model.ctc_head.parameters(), "lr": float(train_cfg.get("transfer_head_lr", 3e-4))},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        betas=(0.9, 0.98), eps=1e-8,
    )
    amp_enabled = device.type == "cuda" and bool(train_cfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=False, reduction="none")
    head_warmup_epochs = int(train_cfg.get("transfer_head_warmup_epochs", 1))
    freeze_frontend = bool(train_cfg.get("transfer_freeze_feature_extractor", True))
    accumulation = max(1, int(train_cfg.get("grad_accum_steps", 2)))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    patience = int(train_cfg.get("transfer_patience", 4))
    best_wer = float("inf")
    best_epoch = 0
    history = []
    start_epoch = 1
    global_step = 0
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"

    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"Cannot resume; missing {last_path}")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("model_type") != "wav2vec2_ctc":
            raise RuntimeError("Checkpoint is not a Wav2Vec2 transfer-learning checkpoint.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_wer = float(checkpoint.get("best_wer", best_wer))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        history = checkpoint.get("history", [])
        print(f"Resuming at epoch {start_epoch}")

    tokenizer.save(output_dir / "tokenizer.json")
    config_payload = {
        "model_type": "wav2vec2_ctc",
        "pretrained_bundle": "WAV2VEC2_BASE",
        "sample_rate": 16000,
        "labels": list(tokenizer.labels),
        "vocab_size": tokenizer.vocab_size,
        "blank_id": tokenizer.blank_id,
    }
    (output_dir / "model_config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )

    for epoch in range(start_epoch, epochs + 1):
        model.set_epoch_trainability(epoch, head_warmup_epochs, freeze_frontend)
        model.train()
        if epoch <= head_warmup_epochs:
            # Keep encoder dropout disabled while fitting the fresh CTC head.
            model.backbone.eval()
        batch_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_examples = 0
        skipped_impossible = 0
        updates = 0
        pending_micro_batches = 0
        started = time.perf_counter()
        progress = tqdm(train_loader, desc=f"Transfer epoch {epoch}")

        def apply_optimizer_update(micro_batches: int) -> None:
            nonlocal updates, global_step
            scaler.unscale_(optimizer)
            if micro_batches < accumulation:
                correction = accumulation / max(micro_batches, 1)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            optimizer.zero_grad(set_to_none=True)
            if not scaler.is_enabled() or scale_after >= scale_before:
                updates += 1
                global_step += 1

        for batch_index, batch in enumerate(progress):
            waveforms = batch["waveforms"].to(device, non_blocking=True)
            waveform_lengths = batch["waveform_lengths"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"]

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp_enabled
            ):
                logits, output_lengths = model(waveforms, waveform_lengths)

            log_probs = F.log_softmax(logits.float(), dim=-1)
            per_example_loss = criterion(
                log_probs.transpose(0, 1), targets,
                output_lengths.detach().cpu(), target_lengths,
            )
            finite = torch.isfinite(per_example_loss)
            skipped_impossible += int((~finite).sum().item())
            if not finite.any():
                continue
            loss = per_example_loss[finite].mean()
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss / accumulation).backward()
            pending_micro_batches += 1
            if pending_micro_batches >= accumulation:
                apply_optimizer_update(pending_micro_batches)
                pending_micro_batches = 0

            batch_size = int(finite.sum().item())
            epoch_loss += float(loss.detach().item()) * batch_size
            epoch_examples += batch_size
            progress.set_postfix(
                loss=f"{loss.item():.3f}",
                skipped=skipped_impossible,
                encoder_lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            )

        # Flush a final partial accumulation window, including when trailing
        # batches contained no valid CTC alignments.
        if pending_micro_batches:
            apply_optimizer_update(pending_micro_batches)

        # Greedy validation on the held-out validation manifest.
        model.eval()
        refs, hyps = [], []
        valid_loss_total = 0.0
        valid_count = 0
        with torch.inference_mode():
            for batch in tqdm(valid_loader, desc="Validation", leave=False):
                waveforms = batch["waveforms"].to(device, non_blocking=True)
                waveform_lengths = batch["waveform_lengths"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=amp_enabled
                ):
                    logits, output_lengths = model(waveforms, waveform_lengths)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                per_loss = criterion(
                    log_probs.transpose(0, 1), batch["targets"],
                    output_lengths.cpu(), batch["target_lengths"],
                )
                finite = torch.isfinite(per_loss)
                if finite.any():
                    valid_loss_total += float(per_loss[finite].sum().item())
                    valid_count += int(finite.sum().item())
                hyps.extend(greedy_decode(logits.float(), output_lengths, tokenizer))
                refs.extend(batch["texts"])

        rates = corpus_error_rates(refs, hyps)
        valid_loss = valid_loss_total / max(valid_count, 1)
        improved = rates["wer"] < best_wer
        if improved:
            best_wer = float(rates["wer"])
            best_epoch = epoch

        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(epoch_examples, 1),
            "valid_loss": valid_loss,
            "valid_wer": rates["wer"],
            "valid_cer": rates["cer"],
            "train_examples": epoch_examples,
            "skipped_ctc_impossible": skipped_impossible,
            "optimizer_updates": updates,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(
            f"Epoch {epoch}: train_loss={record['train_loss']:.4f}, "
            f"valid_loss={valid_loss:.4f}, WER={rates['wer']:.4f}, "
            f"CER={rates['cer']:.4f}, skipped_CTC={skipped_impossible}"
        )
        if device.type == "cuda":
            print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
            torch.cuda.reset_peak_memory_stats()

        common = {
            "model_type": "wav2vec2_ctc",
            "pretrained_bundle": "WAV2VEC2_BASE",
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_wer": best_wer,
            "best_epoch": best_epoch,
            "history": history,
            "validation": {"wer": rates["wer"], "cer": rates["cer"], "loss": valid_loss},
        }
        torch.save(common, last_path)
        if improved:
            torch.save(
                {
                    "model_type": "wav2vec2_ctc",
                    "pretrained_bundle": "WAV2VEC2_BASE",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation": common["validation"],
                },
                best_path,
            )
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        if epoch - best_epoch >= patience:
            print(f"Early stopping: no WER improvement for {patience} epochs.")
            break

    if not best_path.is_file():
        raise RuntimeError("Transfer learning did not produce best.pt.")
    print(f"Best checkpoint: {best_path} (epoch {best_epoch}, WER {best_wer:.4f})")


if __name__ == "__main__":
    main()
