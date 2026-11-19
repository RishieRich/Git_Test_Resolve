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


def _layer_norm(vector: Sequence[float], epsilon: float = 1e-5) -> Vector:
    mean = sum(vector) / len(vector)
    variance = sum((value - mean) ** 2 for value in vector) / len(vector)
    scale = math.sqrt(variance + epsilon)
    return [(value - mean) / scale for value in vector]


def _linear(vector: Sequence[float], weights: Matrix) -> Vector:
    return [_dot(vector, row) for row in weights]


class TransformerBlock:
    """A small self-attention block with residuals and a feed-forward layer."""

    def __init__(self, model_size: int, hidden_size: int | None = None) -> None:
        if model_size <= 0:
            raise ValueError("model_size must be positive")
        self.model_size = model_size
        self.hidden_size = hidden_size or model_size * 2
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        model_scale = math.sqrt(model_size)
        hidden_scale = math.sqrt(self.hidden_size)
        self.input_weights = [
            [math.sin((row + 1) * (column + 1)) / model_scale for column in range(model_size)]
            for row in range(self.hidden_size)
        ]
        self.output_weights = [
            [math.cos((row + 1) * (column + 1)) / hidden_scale for column in range(self.hidden_size)]
            for row in range(model_size)
        ]

    def __call__(self, sequence: Sequence[Sequence[float]]) -> Matrix:
        if not sequence:
            return []
        if any(len(token) != self.model_size for token in sequence):
            raise ValueError("every token must match model_size")

        contexts = scaled_dot_product_attention(sequence, sequence, sequence)
        attended = [
            _layer_norm([value + context[index] for index, value in enumerate(token)])
            for token, context in zip(sequence, contexts)
        ]

        transformed: Matrix = []
        for token in attended:
            hidden = [max(0.0, value) for value in _linear(token, self.input_weights)]
            projected = _linear(hidden, self.output_weights)
            transformed.append(
                _layer_norm([value + projected[index] for index, value in enumerate(token)])
            )
        return transformed
