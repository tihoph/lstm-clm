"""Test the multinomial module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tensorflow as tf

from lstm_clm.clm import MultinomialCLM

if TYPE_CHECKING:
    from collections.abc import Sequence


BATCH_SIZE = 2
SEQ_LEN = 5
VOCAB_SIZE = 10


@pytest.fixture
def mock_model(lstm_dims: list[int]) -> MultinomialCLM:
    layers = tf.keras.layers

    class MockModel:
        def __init__(self) -> None:
            self.counter = tf.Variable(1)

        dims = lstm_dims
        start_token = tf.constant([0])
        has_embedding = True
        seq_len = SEQ_LEN
        model = tf.keras.Sequential(
            [
                layers.Embedding(VOCAB_SIZE, lstm_dims[0]),
                layers.LSTM(lstm_dims[0], return_sequences=True),
                layers.LSTM(lstm_dims[1], return_sequences=True),
                layers.TimeDistributed(layers.Dense(VOCAB_SIZE)),
            ]
        )

        def call_cell(
            self,
            x_t: tf.Tensor,
            hidden_states: Sequence[Sequence[tf.Tensor]],
            /,
            training: bool = False,
        ) -> tuple[tf.Tensor, list[list[tf.Tensor]], None]:
            del x_t, training
            batch = tf.fill((BATCH_SIZE,), self.counter)
            self.counter.assign_add(1)
            y_t = tf.one_hot(batch, 10, dtype=tf.float32)
            return y_t, hidden_states, None  # type: ignore[return-value]

        def sample_next_tokens(
            self, i: tf.Tensor, x_t: tf.Tensor, /, temp: float
        ) -> tf.Tensor:
            del i, temp
            return tf.argmax(x_t, axis=1, output_type=tf.int32)

    return MockModel()  # type: ignore[return-value]


def test_generate_batch(mock_model: MultinomialCLM) -> None:
    samples, preds = MultinomialCLM._generate_batch(  # noqa: SLF001
        mock_model, 1.0, BATCH_SIZE, mock_model.call_cell, mock_model.sample_next_tokens
    )
    assert samples.shape == (BATCH_SIZE, SEQ_LEN + 1)
    expected_samples = np.tile(np.arange(SEQ_LEN + 1), BATCH_SIZE).reshape(
        BATCH_SIZE, SEQ_LEN + 1
    )
    np.testing.assert_equal(samples, expected_samples)
    assert preds.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)  # not seqlen + 1
    expected_preds = tf.one_hot(
        samples[:, 1:], VOCAB_SIZE, dtype=tf.float32
    )  # from index 1 on
    np.testing.assert_allclose(preds, expected_preds)
