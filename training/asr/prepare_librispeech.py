#!/usr/bin/env python3
"""Download/extract LibriSpeech and create JSONL manifests.

Default splits: train-clean-100, dev-clean and test-clean. For an initial smoke
test, train on a limited number of examples before attempting the full 100 hours.
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

import soundfile as sf
from tqdm import tqdm

BASE_URL = "https://www.openslr.org/resources/12"
VALID_SPLITS = {
    "train-clean-100", "train-clean-360", "train-other-500",
    "dev-clean", "dev-other", "test-clean", "test-other",
}


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1024:
        print(f"Using existing archive: {destination}")
        return

    request = urllib.request.Request(url, headers={"User-Agent": "CarServiceASR/0.1"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length", "0"))
            with temporary.open("wb") as output, tqdm(
                total=total or None, unit="B", unit_scale=True, desc=destination.name
            ) as progress:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    progress.update(len(chunk))
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def safe_extract(archive_path: Path, destination: Path) -> None:
    """Extract only regular files/directories and reject path traversal or links."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        safe_members = []
        for member in archive.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts:
                continue
            if member.issym() or member.islnk() or member.isdev():
                continue
            target = (destination / Path(*pure.parts)).resolve()
            if target != root and root not in target.parents:
                continue
            safe_members.append(member)
        archive.extractall(destination, members=safe_members)


def split_is_extracted(split_dir: Path) -> bool:
    return split_dir.exists() and next(split_dir.rglob("*.trans.txt"), None) is not None


def prepare_split(root: Path, split: str, keep_archives: bool = False) -> Path:
    split_dir = root / "LibriSpeech" / split
    if split_is_extracted(split_dir):
        print(f"{split} is already extracted.")
        return split_dir

    archive_path = root / f"{split}.tar.gz"
    download_file(f"{BASE_URL}/{split}.tar.gz", archive_path)
    print(f"Extracting {archive_path.name} ...")
    safe_extract(archive_path, root)
    if not split_is_extracted(split_dir):
        raise RuntimeError(f"No transcript files found after extracting {archive_path}")
    if not keep_archives:
        archive_path.unlink(missing_ok=True)
    return split_dir


def normalize_transcript(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def create_manifest(split_dir: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as manifest:
        for transcript_file in sorted(split_dir.rglob("*.trans.txt")):
            with transcript_file.open("r", encoding="utf-8") as source:
                for line in source:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    utterance_id, raw_text = parts
                    audio_path = transcript_file.parent / f"{utterance_id}.flac"
                    if not audio_path.is_file():
                        continue
                    text = normalize_transcript(raw_text)
                    if not text:
                        continue
                    try:
                        duration = float(sf.info(str(audio_path)).duration)
                    except (RuntimeError, OSError):
                        continue
                    record = {
                        "id": utterance_id,
                        "audio_filepath": str(audio_path.resolve()),
                        "text": text,
                        "duration": duration,
                    }
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1

    if not count:
        raise RuntimeError(f"No utterances were written to {output_path}")
    print(f"Wrote {count:,} utterances to {output_path}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/librispeech"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument(
        "--splits", nargs="+",
        default=["train-clean-100", "dev-clean", "test-clean"],
        choices=sorted(VALID_SPLITS),
    )
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()

    for split in args.splits:
        split_dir = prepare_split(args.root, split, args.keep_archives)
        create_manifest(split_dir, args.manifest_dir / f"{split}.jsonl")


if __name__ == "__main__":
    main()
