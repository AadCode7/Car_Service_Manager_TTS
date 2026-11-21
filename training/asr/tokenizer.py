"""SentencePiece tokenizer helpers for CTC."""

from __future__ import annotations

import re
from pathlib import Path

import sentencepiece as spm


def normalize_text(text: str) -> str:
    """Normalise English training and evaluation text consistently."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class ASRTokenizer:
    """SentencePiece tokenizer. ID 0 is reserved as the CTC blank token."""

    blank_id = 0

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer model not found: {self.model_path}. "
                "Run python -m training.asr.train_tokenizer first."
            )
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))
        if self.processor.pad_id() != self.blank_id:
            raise ValueError(
                f"Expected SentencePiece pad_id=0 for CTC blank, got {self.processor.pad_id()}"
            )

    @property
    def vocab_size(self) -> int:
        return int(self.processor.get_piece_size())

    def encode(self, text: str) -> list[int]:
        normalized = normalize_text(text)
        if not normalized:
            return []
        ids = self.processor.encode(normalized, out_type=int)
        return [int(token_id) for token_id in ids if int(token_id) != self.blank_id]

    def decode(self, token_ids: list[int]) -> str:
        filtered = [
            int(token_id) for token_id in token_ids
            if int(token_id) != self.blank_id and 0 <= int(token_id) < self.vocab_size
        ]
        return normalize_text(self.processor.decode(filtered))
