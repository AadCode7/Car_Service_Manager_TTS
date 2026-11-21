"""CTC decoding and corpus-level WER/CER metrics."""

from __future__ import annotations

from typing import Iterable

import torch

from .tokenizer import ASRTokenizer, normalize_text


def greedy_decode(
    logits: torch.Tensor,
    output_lengths: torch.Tensor,
    tokenizer: ASRTokenizer,
    blank_id: int = 0,
) -> list[str]:
    best_ids = logits.argmax(dim=-1)
    hypotheses: list[str] = []
    lengths = output_lengths.detach().cpu().tolist()

    for row, valid_length in zip(best_ids, lengths):
        sequence = row[: int(valid_length)].detach().cpu().tolist()
        collapsed: list[int] = []
        previous = None
        for token_id in sequence:
            token_id = int(token_id)
            if token_id != blank_id and token_id != previous:
                collapsed.append(token_id)
            previous = token_id
        hypotheses.append(tokenizer.decode(collapsed))
    return hypotheses


def ctc_prefix_beam_search(
    log_probabilities: torch.Tensor,
    tokenizer: ASRTokenizer,
    beam_width: int = 8,
    blank_id: int = 0,
    token_top_k: int = 20,
) -> str:
    """Small pure-Python CTC prefix beam decoder; no external language model."""
    import math

    def log_add(*values: float) -> float:
        finite = [value for value in values if value != -math.inf]
        if not finite:
            return -math.inf
        maximum = max(finite)
        return maximum + math.log(sum(math.exp(value - maximum) for value in finite))

    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -math.inf)}
    frame_values = log_probabilities.detach().float().cpu()
    for frame in frame_values:
        values, indices = torch.topk(frame, k=min(token_top_k, frame.numel()))
        candidates = list(zip(indices.tolist(), values.tolist()))
        updated: dict[tuple[int, ...], tuple[float, float]] = {}

        def add(prefix: tuple[int, ...], blank_score: float = -math.inf,
                nonblank_score: float = -math.inf) -> None:
            old_blank, old_nonblank = updated.get(prefix, (-math.inf, -math.inf))
            updated[prefix] = (
                log_add(old_blank, blank_score),
                log_add(old_nonblank, nonblank_score),
            )

        for prefix, (prob_blank, prob_nonblank) in beams.items():
            total = log_add(prob_blank, prob_nonblank)
            add(prefix, blank_score=total + float(frame[blank_id]))
            for token_id, token_score in candidates:
                if token_id == blank_id:
                    continue
                token_score = float(token_score)
                if prefix and token_id == prefix[-1]:
                    # Repeating a token without an intervening blank does not
                    # append a new output token.
                    add(prefix, nonblank_score=prob_nonblank + token_score)
                    # A repeated token after blank does append a new token.
                    add(prefix + (token_id,), nonblank_score=prob_blank + token_score)
                else:
                    add(prefix + (token_id,), nonblank_score=total + token_score)

        beams = dict(
            sorted(
                updated.items(),
                key=lambda item: log_add(*item[1]),
                reverse=True,
            )[:beam_width]
        )

    best_prefix = max(
        beams.items(), key=lambda item: log_add(*item[1])
    )[0]
    return tokenizer.decode(list(best_prefix))


def edit_distance(reference: list[str] | str, hypothesis: list[str] | str) -> int:
    """Memory-efficient Levenshtein distance."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, start=1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, start=1):
            insertion = current[column - 1] + 1
            deletion = previous[column] + 1
            substitution = previous[column - 1] + (ref_item != hyp_item)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def word_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    return edit_distance(ref_words, hyp_words), max(1, len(ref_words))


def char_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    ref_text = normalize_text(reference).replace(" ", "")
    hyp_text = normalize_text(hypothesis).replace(" ", "")
    return edit_distance(ref_text, hyp_text), max(1, len(ref_text))


def corpus_error_rates(
    references: Iterable[str], hypotheses: Iterable[str]
) -> dict[str, float | int]:
    word_errors = word_count = char_errors = char_count = 0
    sentence_count = 0
    for reference, hypothesis in zip(references, hypotheses):
        w_errors, w_count = word_error_counts(reference, hypothesis)
        c_errors, c_count = char_error_counts(reference, hypothesis)
        word_errors += w_errors
        word_count += w_count
        char_errors += c_errors
        char_count += c_count
        sentence_count += 1

    return {
        "wer": word_errors / max(word_count, 1),
        "cer": char_errors / max(char_count, 1),
        "word_errors": word_errors,
        "word_count": word_count,
        "char_errors": char_errors,
        "char_count": char_count,
        "sentences": sentence_count,
    }


def sentence_error_rates(reference: str, hypothesis: str) -> tuple[float, float]:
    w_errors, w_count = word_error_counts(reference, hypothesis)
    c_errors, c_count = char_error_counts(reference, hypothesis)
    return w_errors / w_count, c_errors / c_count
