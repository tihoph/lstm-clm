"""Test the base class for all CLM models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from lstm_clm.clm.base import BaseCLM

if TYPE_CHECKING:
    import tensorflow as tf

    from lstm_clm.vocab import Vocabulary


def test_baseclm_vocab(
    small_model: tf.keras.Model, small_vocab: Vocabulary, max_len: int
) -> None:
    BaseCLM(model=small_model, max_len=max_len)

    class TestBase(BaseCLM):
        ASSERT_VOCAB = True

    with pytest.raises(ValueError, match="Vocabulary is required for this model"):
        TestBase(model=small_model, max_len=max_len)

    clm_a = TestBase(model=small_model, max_len=max_len, vocab=small_vocab)
    with pytest.raises(
        ValueError, match=r"Model/Vocab discrepancy: max_len does not match"
    ):
        TestBase(model=small_model, max_len=max_len + 1, vocab=small_vocab)

    clm_b = TestBase.from_vocab(small_model, small_vocab)

    assert clm_a.model == clm_b.model
    assert clm_a.vocab == clm_b.vocab
    assert clm_a.seq_len == clm_b.seq_len
    assert clm_a.vocab_size == clm_b.vocab_size
    assert clm_a.dims == clm_b.dims
    np.testing.assert_equal(clm_a.start_token.numpy(), clm_b.start_token.numpy())
