#!/usr/bin/env python3
"""Train the random-initialised Conformer-CTC ASR model."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    ASRDataset,
    DurationBucketBatchSampler,
    apply_spec_augment,
    collate_asr,
)
from .metrics import corpus_error_rates, greedy_decode
from .model import ConformerCTC, model_parameter_count
from .tokenizer import ASRTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    required = {"data", "audio", "model", "training", "output"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    return config


class ModelEMA:
    """Exponential moving average of model weights for more stable evaluation."""

    def __init__(self, model: nn.Module, decay: float):
        self.model = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.num_updates = 0
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, source: nn.Module) -> None:
        self.num_updates += 1
        # Warm up EMA decay to avoid retaining randomly initialised weights in
        # the early checkpoints.
        decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        source_state = source.state_dict()
        for key, ema_value in self.model.state_dict().items():
            current = source_state[key].detach()
            if ema_value.is_floating_point():
                ema_value.mul_(decay).add_(current, alpha=1.0 - decay)
            else:
                ema_value.copy_(current)


def get_autocast_settings(device: torch.device, config: dict[str, Any]):
    enabled = bool(config.get("amp", True)) and device.type == "cuda"
    requested = str(config.get("amp_dtype", "bfloat16")).lower()
    if requested == "bfloat16" and enabled and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        scaler_enabled = False
    else:
        dtype = torch.float16
        scaler_enabled = enabled
    return enabled, dtype, scaler_enabled


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def build_optimizer(model: nn.Module, config: dict[str, Any]):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(config["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(config["learning_rate"]),
        betas=(0.9, 0.98),
        eps=1e-8,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
):
    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return max(1e-4, float(step + 1) / warmup_steps)
        progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    tokenizer: ASRTokenizer,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    references: list[str] = []
    hypotheses: list[str] = []

    for batch_index, batch in enumerate(tqdm(loader, desc="Validation", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        features = batch["features"].to(device, non_blocking=True)
        feature_lengths = batch["feature_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"]

        with autocast_context(device, amp_enabled, amp_dtype):
            logits, output_lengths = model(features, feature_lengths)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        loss = criterion(
            log_probs.transpose(0, 1),
            targets,
            output_lengths.detach().cpu(),
            target_lengths.detach().cpu(),
        )
        batch_size = len(batch["texts"])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        hypotheses.extend(greedy_decode(logits.float(), output_lengths, tokenizer))
        references.extend(batch["texts"])

    rates = corpus_error_rates(references, hypotheses)
    return {
        "loss": total_loss / max(total_examples, 1),
        "wer": rates["wer"],
        "cer": rates["cer"],
        "word_errors": rates["word_errors"],
        "word_count": rates["word_count"],
        "char_errors": rates["char_errors"],
        "char_count": rates["char_count"],
        "sentences": rates["sentences"],
        "references": references,
        "hypotheses": hypotheses,
    }


def train_one_epoch(
    model: nn.Module,
    ema: ModelEMA | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.CTCLoss,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    config: dict[str, Any],
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    epoch: int,
    initial_global_step: int = 0,
    batch_offset: int = 0,
    checkpoint_callback=None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_examples = 0
    steps_since_update = 0
    global_step = int(initial_global_step)
    accumulation = max(1, int(config.get("grad_accum_steps", 1)))
    max_batches = len(loader)
    started = time.perf_counter()
    progress = tqdm(loader, desc=f"Epoch {epoch}", leave=True)

    for batch_index, batch in enumerate(progress):
        features = batch["features"].to(device, non_blocking=True)
        feature_lengths = batch["feature_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"]
        if config.get("spec_augment", True):
            features = apply_spec_augment(
                features,
                feature_lengths,
                int(config.get("freq_masks", 2)),
                int(config.get("freq_mask_width", 15)),
                int(config.get("time_masks", 10)),
                int(config.get("time_mask_width", 40)),
            )

        with autocast_context(device, amp_enabled, amp_dtype):
            logits, output_lengths = model(features, feature_lengths)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        loss = criterion(
            log_probs.transpose(0, 1),
            targets,
            output_lengths.detach().cpu(),
            target_lengths.detach().cpu(),
        )

        if not torch.isfinite(loss):
            print(f"WARNING: non-finite CTC loss at batch {batch_index}; batch skipped.")
            optimizer.zero_grad(set_to_none=True)
            steps_since_update = 0
            continue

        scaler.scale(loss / accumulation).backward()
        steps_since_update += 1
        batch_size = len(batch["texts"])
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size

        should_update = (
            steps_since_update >= accumulation or batch_index == max_batches - 1
        )
        if should_update:
            scaler.unscale_(optimizer)
            if steps_since_update < accumulation:
                correction = accumulation / max(steps_since_update, 1)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.get("max_grad_norm", 5.0))
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update(model)
            steps_since_update = 0
            global_step += 1
            if checkpoint_callback is not None:
                checkpoint_callback(batch_offset + batch_index, global_step)

        progress.set_postfix(
            loss=f"{loss.item():.3f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    elapsed = time.perf_counter() - started
    return {
        "loss": total_loss / max(total_examples, 1),
        "seconds": elapsed,
        "examples": total_examples,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "global_step": global_step,
    }


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("training/asr/config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(resolve_path(args.config))
    if args.output_dir:
        config["output"]["directory"] = str(args.output_dir)
    elif args.smoke:
        # Keep the tiny two-layer smoke checkpoint separate from the full model
        # so --resume on the real run cannot load incompatible tensor shapes.
        config["output"]["directory"] = "artifacts/asr-smoke"
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.max_train_samples is not None:
        config["data"]["max_train_samples"] = args.max_train_samples
    if args.max_valid_samples is not None:
        config["data"]["max_valid_samples"] = args.max_valid_samples

    if args.smoke:
        config["data"]["num_workers"] = 0
        config["training"].update({
            "epochs": 1,
            "max_batch_seconds": 24.0,
            "max_batch_size": 8,
            "grad_accum_steps": 1,
            "amp": False,
            "spec_augment": False,
            "use_ema": False,
            "max_valid_batches": 3,
        })
        config["data"]["max_train_samples"] = (
            config["data"].get("max_train_samples") or 128
        )
        config["data"]["max_valid_samples"] = (
            config["data"].get("max_valid_samples") or 32
        )
        config["model"].update({
            "d_model": 96,
            "n_heads": 4,
            "n_layers": 2,
            "ff_multiplier": 2,
            "subsampling_channels": 32,
        })

    set_seed(int(config.get("seed", 42)))
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    output_dir = resolve_path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = resolve_path(config["data"]["tokenizer_model"])
    tokenizer = ASRTokenizer(tokenizer_path)
    print(f"Tokenizer vocabulary: {tokenizer.vocab_size}")

    train_dataset = ASRDataset(
        resolve_path(config["data"]["train_manifest"]),
        tokenizer,
        config["audio"],
        max_samples=config["data"].get("max_train_samples"),
        speed_perturb=bool(config["training"].get("speed_perturb", True)),
        speed_rates=config["training"].get("speed_perturb_rates", [0.9, 1.0, 1.1]),
    )
    valid_dataset = ASRDataset(
        resolve_path(config["data"]["valid_manifest"]),
        tokenizer,
        config["audio"],
        max_samples=config["data"].get("max_valid_samples"),
        speed_perturb=False,
    )

    batch_sampler = DurationBucketBatchSampler(
        train_dataset.durations,
        max_batch_seconds=float(config["training"]["max_batch_seconds"]),
        max_batch_size=int(config["training"]["max_batch_size"]),
        shuffle=True,
        seed=int(config.get("seed", 42)),
    )
    workers = int(config["data"].get("num_workers", 2))
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=collate_asr,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=collate_asr,
    )

    model_config = dict(config["model"])
    model_config.update({
        "n_mels": int(config["audio"]["n_mels"]),
        "vocab_size": tokenizer.vocab_size,
    })
    model = ConformerCTC(**model_config).to(device)
    print(f"Trainable parameters: {model_parameter_count(model):,}")

    ema = None
    if config["training"].get("use_ema", True):
        ema = ModelEMA(model, float(config["training"].get("ema_decay", 0.999)))

    optimizer = build_optimizer(model, config["training"])
    updates_per_epoch = math.ceil(
        len(train_loader) / max(1, int(config["training"].get("grad_accum_steps", 1)))
    )
    total_updates = updates_per_epoch * int(config["training"]["epochs"])
    base_lr = float(config["training"]["learning_rate"])
    scheduler = build_scheduler(
        optimizer,
        int(config["training"].get("warmup_steps", 1500)),
        total_updates,
        float(config["training"].get("min_learning_rate", 0.00002)) / base_lr,
    )

    amp_enabled, amp_dtype, scaler_enabled = get_autocast_settings(
        device, config["training"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True, reduction="mean")

    start_epoch = 1
    resume_batch = 0
    global_step = 0
    best_wer = float("inf")
    epochs_without_improvement = 0
    history_path = output_dir / "history.csv"
    history: list[dict[str, Any]] = []

    if args.resume:
        checkpoint_path = output_dir / "last.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint missing: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if ema is not None and checkpoint.get("ema_state_dict"):
            ema.model.load_state_dict(checkpoint["ema_state_dict"])
            ema.num_updates = int(checkpoint.get("ema_num_updates", 0))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        epoch_complete = bool(checkpoint.get("epoch_complete", True))
        start_epoch = int(checkpoint["epoch"]) + (1 if epoch_complete else 0)
        resume_batch = (
            0 if epoch_complete else int(checkpoint.get("batch_in_epoch", -1)) + 1
        )
        global_step = int(checkpoint.get("global_step", 0))
        best_wer = float(checkpoint.get("best_wer", best_wer))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        if history_path.is_file():
            with history_path.open("r", newline="", encoding="utf-8") as source:
                history = list(csv.DictReader(source))
        print(f"Resuming from epoch {start_epoch}")

    destination_tokenizer = output_dir / "tokenizer.model"
    if tokenizer_path.resolve() != destination_tokenizer.resolve():
        shutil.copy2(tokenizer_path, destination_tokenizer)
    with (output_dir / "model_config.json").open("w", encoding="utf-8") as output:
        json.dump(
            {"model": model_config, "audio": config["audio"]},
            output,
            indent=2,
        )

    max_valid_batches = config["training"].get("max_valid_batches")
    max_valid_batches = int(max_valid_batches) if max_valid_batches else None

    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        batch_offset = resume_batch if epoch == start_epoch else 0
        batch_sampler.set_epoch(epoch, start_batch=batch_offset)

        def save_step_checkpoint(completed_batch_index: int, current_global_step: int) -> None:
            interval = int(config["training"].get("checkpoint_every_steps", 300))
            if interval <= 0 or current_global_step % interval != 0:
                return
            step_payload = {
                "epoch": epoch,
                "epoch_complete": False,
                "batch_in_epoch": completed_batch_index,
                "global_step": current_global_step,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.model.state_dict() if ema is not None else None,
                "ema_num_updates": ema.num_updates if ema is not None else 0,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_wer": best_wer,
                "epochs_without_improvement": epochs_without_improvement,
                "model_config": model_config,
                "audio_config": config["audio"],
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "history": history,
            }
            save_checkpoint(output_dir / "last.pt", step_payload)

        train_stats = train_one_epoch(
            model, ema, train_loader, optimizer, scheduler, criterion, scaler,
            device, config["training"], amp_enabled, amp_dtype, epoch,
            initial_global_step=global_step,
            batch_offset=batch_offset,
            checkpoint_callback=save_step_checkpoint,
        )
        global_step = int(train_stats["global_step"])
        resume_batch = 0
        evaluation_model = ema.model if ema is not None else model
        valid_stats = validate(
            evaluation_model, valid_loader, criterion, tokenizer, device,
            amp_enabled, amp_dtype, max_valid_batches
        )

        improved = valid_stats["wer"] < best_wer
        if improved:
            best_wer = float(valid_stats["wer"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "valid_loss": valid_stats["loss"],
            "valid_wer": valid_stats["wer"],
            "valid_cer": valid_stats["cer"],
            "learning_rate": train_stats["learning_rate"],
            "epoch_seconds": train_stats["seconds"],
            "train_examples": train_stats["examples"],
            "valid_sentences": valid_stats["sentences"],
            "best_wer": best_wer,
        }
        history.append(record)
        write_history(history_path, history)
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")

        checkpoint_common = {
            "epoch": epoch,
            "epoch_complete": True,
            "batch_in_epoch": -1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.model.state_dict() if ema is not None else None,
            "ema_num_updates": ema.num_updates if ema is not None else 0,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_wer": best_wer,
            "epochs_without_improvement": epochs_without_improvement,
            "model_config": model_config,
            "audio_config": config["audio"],
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "history": history,
        }
        save_checkpoint(output_dir / "last.pt", checkpoint_common)

        if improved:
            best_weights = (
                ema.model.state_dict() if ema is not None else model.state_dict()
            )
            best_checkpoint = {
                "epoch": epoch,
                "model_state_dict": best_weights,
                "model_config": model_config,
                "audio_config": config["audio"],
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "validation": {
                    "loss": valid_stats["loss"],
                    "wer": valid_stats["wer"],
                    "cer": valid_stats["cer"],
                },
            }
            save_checkpoint(output_dir / "best.pt", best_checkpoint)

        print(
            f"Epoch {epoch}: train_loss={train_stats['loss']:.4f}, "
            f"valid_loss={valid_stats['loss']:.4f}, "
            f"WER={valid_stats['wer']:.4f}, CER={valid_stats['cer']:.4f}, "
            f"best_WER={best_wer:.4f}"
        )
        if device.type == "cuda":
            peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"Peak allocated GPU memory: {peak_gib:.2f} GiB")
            torch.cuda.reset_peak_memory_stats()

        if epochs_without_improvement >= int(
            config["training"].get("early_stopping_patience", 8)
        ):
            print("Early stopping: validation WER has not improved.")
            break

    if not (output_dir / "best.pt").is_file():
        raise RuntimeError("Training ended without producing best.pt.")
    print(f"Training complete. Best checkpoint: {output_dir / 'best.pt'}")
    print("Next: run python -m training.asr.plot_metrics --run-dir " + str(output_dir))


if __name__ == "__main__":
    main()
