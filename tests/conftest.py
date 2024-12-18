"""Fixtures for CLM tests."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tensorflow as tf

from lstm_clm.vocab import Vocabulary

if TYPE_CHECKING:
    from collections.abc import Callable

SCALE_FACTOR = 1e6
MAX_LEN = 7
SEQ_LEN = 8

BATCH_SIZE = 64


FIRST_LSTM = 32
SECOND_LSTM = 16
LSTM_DIMS = [[FIRST_LSTM], [FIRST_LSTM, SECOND_LSTM]]

FEW_TOKENS = ["A", "G", "E", "U", "C", "c", "O", "o", "Cl"]
INPUT_DIMS = len(FEW_TOKENS)


### FIXTURES FOR CONSTANT VALUES ###
@pytest.fixture
def batch_size() -> int:
    return BATCH_SIZE


@pytest.fixture
def max_len() -> int:
    return MAX_LEN


@pytest.fixture
def input_dims() -> int:
    return INPUT_DIMS


@pytest.fixture
def lstm_dims() -> list[int]:
    return [FIRST_LSTM, SECOND_LSTM]


@pytest.fixture
def seq_len() -> int:
    return SEQ_LEN


@pytest.fixture
def scale_factor() -> float:
    return SCALE_FACTOR


####################################
####################################


def build_model(
    dims: list[int],
    batch_norm: bool = False,
    time_dist: bool = False,
    embedding: bool = False,
) -> tf.keras.Model:
    layers = tf.keras.layers
    input_layer = (
        layers.Embedding(INPUT_DIMS, dims[0]) if embedding else layers.Dense(INPUT_DIMS)
    )
    bn_layer = [layers.BatchNormalization()] if batch_norm else []
    lstm_layers = [layers.LSTM(dim, return_sequences=True) for dim in dims]
    dense_layer = layers.Dense(INPUT_DIMS)
    final_layer = layers.TimeDistributed(dense_layer) if time_dist else dense_layer
    model = tf.keras.Sequential([input_layer, *bn_layer, *lstm_layers, final_layer])
    model.build((None, SEQ_LEN, INPUT_DIMS))
    weights: list[np.ndarray] = model.get_weights()

    # adjust weights to be close in a range the floating point error
    # are small.
    adjusted_weights = [
        np.arange(weight.size).reshape(weight.shape) / 100 / weight.size
        for weight in weights
    ]
    model.set_weights(adjusted_weights)
    return model


@pytest.fixture
def small_model() -> tf.keras.Model:
    return build_model([FIRST_LSTM, SECOND_LSTM], True, True)


@pytest.fixture
def build_small_model() -> Callable[[bool], tf.keras.Model]:
    return lambda emb: build_model([FIRST_LSTM, SECOND_LSTM], True, True, emb)


@pytest.fixture
def small_vocab() -> Vocabulary:
    return Vocabulary(FEW_TOKENS, MAX_LEN, "A", "G", "E", "U")


@pytest.fixture
def train_model() -> tf.keras.Model:
    """Build a compiled model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(49, 64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LSTM(1024, return_sequences=True),
            tf.keras.layers.LSTM(256, return_sequences=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(49, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    )
    return model


@pytest.fixture
def train_vocab() -> Vocabulary:
    """Build a vocabulary."""
    return Vocabulary(
        ["A", "G", "E", "U", "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I", "b", "c", "n", "o", "si", "p", "s", "se", "+", "-", "=", "#", ".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "%", "(", ")", "[", "]", "/", "\\", "@", "@@"], # noqa: E501
        SEQ_LEN,
        "A",
        "G",
        "E",
        only_two_char_tokens=True,
    )  # fmt: skip


def to_logits(tensor: tf.Tensor) -> tf.Tensor:
    return tf.math.log(tf.cast(tensor, tf.float64))


def softmax(logits: tf.Tensor) -> tf.Tensor:
    exp_logits = tf.math.exp(logits)
    return exp_logits / tf.math.reduce_sum(exp_logits)


### VOCABULARY FIXTURES ###
@pytest.fixture
def decode_map() -> dict[int, str]:
    """Some sample characters + special tokens (UBEA)."""
    return dict(enumerate("helowrdiUBEA"))


@pytest.fixture
def encode_map(decode_map: dict[int, str]) -> dict[str, int]:
    """Reverse the decode map."""
    return {v: k for k, v in decode_map.items()}


@pytest.fixture
def adjusted_decode_map(
    decode_map: dict[int, str], encode_map: dict[str, int]
) -> dict[int, str]:
    """Adjusted decode map with special tokens (UBEA) set to ''."""
    adjusted_decode_map = decode_map.copy()
    for k in "UBEA":
        adjusted_decode_map[encode_map[k]] = ""
    return adjusted_decode_map


@pytest.fixture
def split_on_every_char() -> re.Pattern:
    """Split on every character: 'a10b' -> ['a', '1', '0', 'b']."""
    return re.compile(r"\s")


@pytest.fixture
def split_on_numbers() -> re.Pattern:
    """Split on numbers: 'a10b' -> ['a', '10', 'b']."""
    return re.compile(r"(\d+)")


@pytest.fixture(params=["hi", "hello", "world"])
def sample_string(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=["U", "B", "E", "A"])
def special_token(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]
