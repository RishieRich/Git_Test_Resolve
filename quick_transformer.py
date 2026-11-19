"""A compact, dependency-free transformer implementation."""

from __future__ import annotations

import math
from collections.abc import Sequence


Vector = list[float]
Matrix = list[Vector]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same size")
    return sum(a * b for a, b in zip(left, right))


def _softmax(values: Sequence[float]) -> Vector:
    if not values:
        return []
    peak = max(values)
    exponentials = [math.exp(value - peak) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def scaled_dot_product_attention(
    queries: Sequence[Sequence[float]],
    keys: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
) -> Matrix:
    """Apply scaled dot-product attention to a sequence of vectors."""
    if not queries:
        return []
    if not keys or len(keys) != len(values):
        raise ValueError("keys and values must be non-empty and equally sized")

    key_size = len(keys[0])
    value_size = len(values[0])
    if key_size == 0 or value_size == 0:
        raise ValueError("key and value vectors cannot be empty")
    if any(len(query) != key_size for query in queries):
        raise ValueError("query and key dimensions must match")
    if any(len(key) != key_size for key in keys):
        raise ValueError("all keys must share a dimension")
    if any(len(value) != value_size for value in values):
        raise ValueError("all values must share a dimension")

    scale = math.sqrt(key_size)
    attended: Matrix = []
    for query in queries:
        weights = _softmax([_dot(query, key) / scale for key in keys])
        attended.append(
            [
                sum(weight * value[index] for weight, value in zip(weights, values))
                for index in range(value_size)
            ]
        )
    return attended
