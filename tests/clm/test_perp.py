"""Test perplexity."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import pytest

from lstm_clm.clm import VocabMultinomialCLM

if TYPE_CHECKING:
    from collections.abc import Callable

    import tensorflow as tf

    from lstm_clm.vocab import Vocabulary


def assert_shapes(
    samples: tf.Tensor,
    preds: tf.Tensor,
    scores: tf.Tensor | None,
    n_batches: int,
    embedding: bool,
    batch_size: int,
    small_vocab: Vocabulary,
) -> None:
    if embedding:
        assert samples.shape == (n_batches * batch_size, small_vocab.max_len + 2)
    else:
        assert samples.shape == (
            n_batches * batch_size,
            small_vocab.max_len + 2,
            len(small_vocab),
        )
    assert preds.shape == (
        n_batches * batch_size,
        small_vocab.max_len + 1,
        len(small_vocab),
    )
    if scores is not None:
        assert scores.shape == (n_batches * batch_size,)


params = list(itertools.product([1, 2], [True, False]))


@pytest.mark.parametrize(("n_batches", "embedding"), params)
def test_generate(
    n_batches: int,
    embedding: bool,
    build_small_model: Callable[[bool], tf.keras.Model],
    batch_size: int,
    small_vocab: Vocabulary,
) -> None:
    small_model = build_small_model(embedding)
    clm = VocabMultinomialCLM.from_vocab(small_model, small_vocab)
    samples, preds = clm.generate(n_batches, batch_size=batch_size)
    assert_shapes(samples, preds, None, n_batches, embedding, batch_size, small_vocab)


@pytest.mark.parametrize(("n_batches", "embedding"), params)
def test_perplexity(
    n_batches: int,
    embedding: bool,
    build_small_model: Callable[[bool], tf.keras.Model],
    batch_size: int,
    small_vocab: Vocabulary,
) -> None:
    small_model = build_small_model(embedding)
    clm = VocabMultinomialCLM.from_vocab(small_model, small_vocab)
    samples, preds, scores = clm.generate_with_perplexity(
        n_batches, batch_size=batch_size
    )
    assert_shapes(samples, preds, scores, n_batches, embedding, batch_size, small_vocab)
