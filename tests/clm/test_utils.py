"""Test the basic CLM utilities functions."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tensorflow as tf

from lstm_clm.clm.utils import (
    categorical_sample,
    init_hidden_states,
    init_tensor_array,
    lstm_model_call,
    multinomial_sample,
)
from tests.conftest import build_model, softmax, to_logits

if TYPE_CHECKING:
    from collections.abc import Callable


def repeated_sample(
    distribution: tf.Tensor,
    temp: float,
    func: Callable[..., tf.Tensor],
    scale_factor: int,
    is_logits: bool = False,
) -> tf.Tensor:
    repeated = tf.repeat([distribution], int(scale_factor), axis=0)
    samples = func(repeated, temp=temp)
    counts = tf.math.bincount(samples)
    scaled = tf.cast(counts, tf.float32) / scale_factor
    logits = to_logits(distribution) if not is_logits else distribution
    adjusted_base = softmax(logits / temp)
    np.testing.assert_allclose(scaled, adjusted_base, rtol=0.01, atol=0.01)


def test_sample_next_tokens(scale_factor: int) -> None:
    base_distribution = tf.constant([0.1, 0.2, 0.3, 0.4], dtype=tf.float32)

    distribution = tf.constant([[0.1, 0.2, 0.3, 0.4]], dtype=tf.float32)

    logits = to_logits(distribution)
    sample = categorical_sample(logits, temp=1.0)
    sample2 = multinomial_sample(distribution, temp=1.0)

    assert sample.shape == (1,)
    assert sample.dtype == tf.int32
    assert sample2.shape == (1,)
    assert sample2.dtype == tf.int32

    base_logits = to_logits(base_distribution)
    repeated_sample(base_logits, 1.0, categorical_sample, scale_factor, is_logits=True)
    repeated_sample(base_distribution, 1.0, multinomial_sample, scale_factor)


@pytest.mark.parametrize("temp", [0.1, 1.0, 2.0, 5.0])
def test_distribution_sample(temp: float, scale_factor: int) -> None:
    base_distribution = tf.constant([0.1, 0.2, 0.3, 0.4], dtype=tf.float32)
    repeated_sample(base_distribution, temp, multinomial_sample, scale_factor)


@pytest.mark.parametrize("dtype", [tf.int32, tf.float32])
def test_init_tensor_array(dtype: tf.DType, seq_len: int) -> None:
    arr = init_tensor_array(dtype, seq_len)
    assert isinstance(arr, tf.TensorArray)
    assert arr.dtype == dtype
    assert arr.size() == seq_len

    for ix in range(seq_len):
        arr.write(ix, tf.ones([2, 2], dtype=dtype))

    with pytest.raises(tf.errors.OutOfRangeError):
        arr.write(seq_len, tf.ones([2, 2], dtype=dtype))


def test_init_hidden_states(lstm_dims: list[int], batch_size: int) -> None:
    hidden_states = init_hidden_states(lstm_dims, 64)

    assert isinstance(hidden_states, list)
    assert len(hidden_states) == len(lstm_dims)

    for ix, dim in enumerate(lstm_dims):
        assert isinstance(hidden_states[ix], list)
        assert len(hidden_states[ix]) == 2

        for state in hidden_states[ix]:
            assert isinstance(state, tf.Tensor)
            assert state.shape == (batch_size, dim)
            np.testing.assert_allclose(state, np.zeros((batch_size, dim)))


params = itertools.product([True, False], [True, False], [1, 2, 3])


@pytest.mark.parametrize(("batch_norm", "time_dist", "seq_len"), params)
def test_multi_lstm_model_call(
    batch_norm: bool,
    time_dist: bool,
    seq_len: int,
    batch_size: int,
    input_dims: int,
    lstm_dims: list[int],
) -> None:
    product = batch_size * seq_len * input_dims
    np_inputs = np.arange(product).reshape(batch_size, seq_len, input_dims) / product
    inputs = tf.convert_to_tensor(np_inputs)

    model = build_model(lstm_dims, batch_norm, time_dist)

    hidden_states = init_hidden_states(lstm_dims, batch_size)

    expected = model(inputs)

    all_outputs: list[tf.Tensor] = []
    for ix in range(seq_len):
        outputs, hidden_states, _ = lstm_model_call(
            model.layers, lstm_dims, inputs[:, ix, :], hidden_states
        )
        all_outputs.append(outputs)

    all_outputs = tf.stack(all_outputs, axis=1)

    np.testing.assert_allclose(all_outputs, expected, rtol=1e-4, atol=1e-4)
