"""Test that we can access wrapped functions in joblib workers."""

from __future__ import annotations

import multiprocessing as mp
import sys
from typing import TYPE_CHECKING

from lstm_clm.clm.wrapper import tf_func_wrapper, tf_kw_func_wrapper

if TYPE_CHECKING:
    import tensorflow as tf


def check_for_tensorflow_import(ix: int) -> tuple[str, bool]:
    from lstm_clm.vocab import Vocabulary

    del Vocabulary, ix  # unused import

    return mp.current_process().name, "tensorflow" in sys.modules


@tf_kw_func_wrapper(input_signature=[])
def test_kw_wrapper() -> tf.Tensor:
    """A tf_func_wrapper-wrapped function."""
    import tensorflow as tf

    return tf.reduce_sum(tf.range(10))


@tf_func_wrapper
def test_wrapper() -> tf.Tensor:
    """A tf_func_wrapper-wrapped function."""
    import tensorflow as tf

    return tf.reduce_sum(tf.range(10))
