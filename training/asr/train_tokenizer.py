#!/usr/bin/env python3
"""Train a SentencePiece tokenizer using training transcripts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm

from .tokenizer import normalize_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/train-clean-100.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/asr/tokenizer.model"))
    parser.add_argument("--vocab-size", type=int, default=500)
    parser.add_argument("--model-type", choices=["unigram", "bpe"], default="unigram")
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(f"Training manifest not found: {args.manifest}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text_path = args.output.with_suffix(".training_text.txt")
    rows = 0

    with args.manifest.open("r", encoding="utf-8") as source, text_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            item = json.loads(line)
            text = normalize_text(item["text"])
            if text:
                target.write(text + "\n")
                rows += 1

    if rows == 0:
        raise RuntimeError("No transcripts were found in the training manifest.")

    prefix = args.output.with_suffix("")
    spm.SentencePieceTrainer.train(
        input=str(text_path),
        model_prefix=str(prefix),
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=1.0,
        pad_id=0,
        pad_piece="<blank>",
        unk_id=1,
        unk_piece="<unk>",
        bos_id=-1,
        eos_id=-1,
        hard_vocab_limit=False,
        shuffle_input_sentence=True,
        input_sentence_size=2_000_000,
        num_threads=4,
    )
    generated_model = prefix.with_suffix(".model")
    if generated_model.resolve() != args.output.resolve():
        generated_model.replace(args.output)
    print(f"Tokenizer trained on {rows:,} transcripts: {args.output}")


if __name__ == "__main__":
    main()
