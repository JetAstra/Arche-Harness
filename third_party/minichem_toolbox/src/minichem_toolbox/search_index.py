"""Small deterministic BM25 implementation for local toolbox catalogs."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping


def lexical_stem(value: str) -> str:
    for suffix in ("izations", "isations", "ization", "isation"):
        if value.endswith(suffix) and len(value) - len(suffix) >= 4:
            return value[: -len(suffix)] + "iz"
    for suffix in (
        "ations",
        "ation",
        "ments",
        "ment",
        "ing",
        "ed",
        "es",
        "s",
    ):
        if value.endswith(suffix) and len(value) - len(suffix) >= 5:
            return value[: -len(suffix)]
    return value


def tokenize(text: str) -> list[str]:
    return [lexical_stem(token) for token in re.findall(r"[a-z0-9]+", text.casefold())]


@dataclass(frozen=True)
class RankedDocument:
    document_id: str
    score: float
    matched_terms: tuple[str, ...]


class BM25Index:
    """Rank a small in-memory document set without external services."""

    def __init__(self, documents: Mapping[str, str], *, k1: float = 1.5, b: float = 0.75):
        self._documents = dict(documents)
        self._tokens = {key: tokenize(value) for key, value in documents.items()}
        self._frequencies = {key: Counter(tokens) for key, tokens in self._tokens.items()}
        self._lengths = {key: len(tokens) for key, tokens in self._tokens.items()}
        self._average_length = (
            sum(self._lengths.values()) / len(self._lengths) if self._lengths else 0.0
        )
        self._document_frequency: Counter[str] = Counter()
        for tokens in self._tokens.values():
            self._document_frequency.update(set(tokens))
        self._k1 = k1
        self._b = b

    def search(self, query: str) -> list[RankedDocument]:
        terms = tuple(dict.fromkeys(tokenize(query)))
        if not terms:
            return [RankedDocument(key, 0.0, ()) for key in sorted(self._documents)]
        count = len(self._documents)
        ranked: list[RankedDocument] = []
        for document_id, frequencies in self._frequencies.items():
            score = 0.0
            matched: list[str] = []
            length = self._lengths[document_id]
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                matched.append(term)
                document_frequency = self._document_frequency[term]
                inverse_document_frequency = math.log(
                    1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + self._k1 * (
                    1.0 - self._b
                    + self._b * length / max(self._average_length, 1.0)
                )
                score += inverse_document_frequency * frequency * (self._k1 + 1.0) / denominator
            if matched:
                ranked.append(RankedDocument(document_id, score, tuple(matched)))
        ranked.sort(key=lambda item: (-item.score, item.document_id))
        return ranked


def normalize_scores(values: Mapping[str, float]) -> dict[str, float]:
    maximum = max(values.values(), default=0.0)
    if maximum <= 0.0:
        return {key: 0.0 for key in values}
    return {key: value / maximum for key, value in values.items()}


def weighted_text(parts: Iterable[tuple[str, int]]) -> str:
    """Repeat short fields to provide transparent field weighting."""

    return " ".join(text for text, weight in parts for _ in range(max(weight, 0)) if text)


__all__ = ["BM25Index", "RankedDocument", "lexical_stem", "normalize_scores", "tokenize", "weighted_text"]
