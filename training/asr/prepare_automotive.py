#!/usr/bin/env python3
"""Validate an automotive speech collection CSV and create speaker-disjoint ASR manifests.

Expected CSV columns:
  audio_filepath,text,speaker_id,consent,transcript_verified
Optional columns:
  session_id,scenario,noise_condition,microphone,vehicle_context,notes

Audio paths may be absolute or relative to the CSV file's directory.
The output manifests use the same audio_filepath/text/duration schema consumed by
TransferASRDataset in training/asr/transfer.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf


REQUIRED_COLUMNS = (
    "audio_filepath",
    "text",
    "speaker_id",
    "consent",
    "transcript_verified",
)
OPTIONAL_COLUMNS = (
    "session_id",
    "scenario",
    "noise_condition",
    "microphone",
    "vehicle_context",
    "notes",
)
SPLIT_NAMES = ("train", "valid", "test")


def yes(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1", "confirmed"}


def normalize_text(value: str) -> str:
    """Match the existing 39-class Wav2Vec2 CTC vocabulary."""
    value = str(value or "").upper().strip()
    value = re.sub(r"[^A-Z0-9' ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def resolve_audio_path(value: str, csv_dir: Path) -> Path:
    path = Path(str(value).strip()).expanduser()
    if not path.is_absolute():
        path = csv_dir / path
    return path.resolve()


def speaker_split(speaker_counts: dict[str, int], seed: int) -> dict[str, str]:
    """Assign whole speakers to splits; no speaker can occur in two splits."""
    speakers = list(speaker_counts)
    if len(speakers) < 3:
        raise ValueError(
            f"At least 3 distinct speakers are required for train/valid/test; "
            f"found {len(speakers)}. Collect more speakers before splitting."
        )

    rng = random.Random(seed)
    rng.shuffle(speakers)
    # Put the largest speakers first after seeded tie randomisation.
    speakers.sort(key=lambda speaker: speaker_counts[speaker], reverse=True)

    total = float(sum(speaker_counts.values()))
    targets = {
        "train": total * 0.80,
        "valid": total * 0.10,
        "test": total * 0.10,
    }
    assigned = {name: 0 for name in SPLIT_NAMES}
    result: dict[str, str] = {}

    # Guarantee at least one speaker in each split.
    for speaker, split in zip(speakers[:3], SPLIT_NAMES):
        result[speaker] = split
        assigned[split] += speaker_counts[speaker]

    for speaker in speakers[3:]:
        # Pick the split furthest below its target, normalised by target size.
        split = max(
            SPLIT_NAMES,
            key=lambda name: (targets[name] - assigned[name]) / max(targets[name], 1.0),
        )
        result[speaker] = split
        assigned[split] += speaker_counts[speaker]

    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Annotated recording metadata CSV.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/automotive/manifests"),
        help="Directory for train.jsonl, valid.jsonl, test.jsonl and summary.json.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-duration", type=float, default=0.30)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--min-speakers", type=int, default=3)
    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        parser.error(f"Metadata CSV not found: {csv_path}")
    if args.min_duration <= 0 or args.max_duration <= args.min_duration:
        parser.error("Duration limits are invalid.")

    accepted: list[dict] = []
    rejected: list[dict] = []
    seen_audio: set[str] = set()
    seen_ids: set[str] = set()
    speaker_counts: Counter[str] = Counter()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        missing = [column for column in REQUIRED_COLUMNS if column not in headers]
        if missing:
            parser.error(f"CSV is missing required columns: {', '.join(missing)}")

        for row_number, source in enumerate(reader, start=2):
            try:
                speaker_id = str(source.get("speaker_id", "")).strip()
                if not speaker_id:
                    raise ValueError("speaker_id is empty")
                if not yes(source.get("consent", "")):
                    raise ValueError("consent is not confirmed")
                if not yes(source.get("transcript_verified", "")):
                    raise ValueError("transcript has not been human-verified")

                transcript = normalize_text(source.get("text", ""))
                if not transcript:
                    raise ValueError("transcript is empty after normalisation")

                audio_path = resolve_audio_path(source.get("audio_filepath", ""), csv_path.parent)
                if not audio_path.is_file():
                    raise ValueError(f"audio file does not exist: {audio_path}")
                audio_key = str(audio_path)
                if audio_key in seen_audio:
                    raise ValueError("audio_filepath is duplicated")
                seen_audio.add(audio_key)

                info = sf.info(str(audio_path))
                duration = float(info.duration)
                if info.frames <= 0 or info.samplerate < 8000 or info.samplerate > 48000:
                    raise ValueError(
                        f"unsupported audio properties: {info.samplerate} Hz, {info.frames} frames"
                    )
                if not (args.min_duration <= duration <= args.max_duration):
                    raise ValueError(
                        f"duration {duration:.2f}s is outside "
                        f"{args.min_duration:.2f}-{args.max_duration:.2f}s"
                    )

                recording_id = str(source.get("id", "")).strip()
                if not recording_id:
                    recording_id = f"auto_{row_number:07d}"
                if recording_id in seen_ids:
                    raise ValueError(f"duplicate recording id: {recording_id}")
                seen_ids.add(recording_id)

                item = {
                    "id": recording_id,
                    "audio_filepath": audio_key,
                    "text": transcript,
                    "duration": round(duration, 4),
                    "speaker_id": speaker_id,
                }
                for column in OPTIONAL_COLUMNS:
                    value = str(source.get(column, "") or "").strip()
                    if value:
                        item[column] = value

                accepted.append(item)
                speaker_counts[speaker_id] += 1
            except Exception as error:
                rejected.append({"row": row_number, "reason": str(error)})

    if rejected:
        print(f"Rejected {len(rejected)} rows:")
        for item in rejected[:30]:
            print(f"  CSV row {item['row']}: {item['reason']}")
        if len(rejected) > 30:
            print(f"  ... and {len(rejected) - 30} more rejected rows")

    if not accepted:
        raise RuntimeError("No valid recordings were found. Fix the CSV/audio issues and rerun.")
    if len(speaker_counts) < args.min_speakers:
        raise RuntimeError(
            f"Only {len(speaker_counts)} speakers passed validation; "
            f"the configured minimum is {args.min_speakers}."
        )

    split_by_speaker = speaker_split(dict(speaker_counts), args.seed)
    split_rows = {name: [] for name in SPLIT_NAMES}
    for item in accepted:
        split_rows[split_by_speaker[item["speaker_id"]]].append(item)

    empty = [name for name, rows in split_rows.items() if not rows]
    if empty:
        raise RuntimeError(f"Speaker assignment produced empty splits: {empty}")

    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, rows in split_rows.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", rows)

    summary = {
        "source_csv": str(csv_path),
        "seed": args.seed,
        "split_method": "speaker-disjoint",
        "recordings_accepted": len(accepted),
        "recordings_rejected": len(rejected),
        "speakers": len(speaker_counts),
        "split_counts": {
            name: {
                "recordings": len(rows),
                "speakers": len({row["speaker_id"] for row in rows}),
                "hours": round(sum(row["duration"] for row in rows) / 3600, 3),
            }
            for name, rows in split_rows.items()
        },
        "rejected_rows": rejected,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Defensive leakage check: each speaker must appear in exactly one split.
    speaker_sets = {
        name: {row["speaker_id"] for row in rows}
        for name, rows in split_rows.items()
    }
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1:]:
            overlap = speaker_sets[left] & speaker_sets[right]
            if overlap:
                raise RuntimeError(f"Speaker leakage between {left} and {right}: {sorted(overlap)}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nManifests written to: {output_dir}")
    print("These JSONL files are compatible with TransferASRDataset.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
