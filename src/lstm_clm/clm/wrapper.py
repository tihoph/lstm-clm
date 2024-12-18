"""Wrapper for TensorFlow in worker processes.

With using multiprocessing import a module which contains
tensorflow or even worse, tf.functions, can lead to retracing
and slows down the execution. This module provides a wrapper
which only imports tensorflow in the main process and raises
an error if it is accessed in a worker process.
"""

from __future__ import annotations

import logging
import multiprocessing as _mp
import os
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

COVERAGE_DEBUG = os.getenv("COVERAGE_DEBUG", "0") == "1"
TRACING_DEBUG = True

P = ParamSpec("P")

R = TypeVar("R")

logger = logging.getLogger(__package__)

_MAIN_PROCESS = _mp.current_process().name == "MainProcess"
logger.debug("Main process: %s %s", _MAIN_PROCESS, _mp.current_process().name)

if TYPE_CHECKING:  # pragma: no cover
    import tensorflow as tf

    class tf_dtype:  # noqa: N801
        """Wrapper for tensorflow data types."""

        int32: tf.DType = tf.int32
        int64: tf.DType = tf.int64
        float32: tf.DType = tf.float32
        float64: tf.DType = tf.float64
        bool: tf.DType = tf.bool
        string: tf.DType = tf.string

    def tf_tensor_spec(
        shape: tuple[()] | list[int | None], dtype: tf.DType
    ) -> tf.TensorSpec:
        """Create a tensor spec from shape and dtype.

        Args:
            shape: Shape of the tensor.
            dtype: Data type of the tensor.

        Returns:
            Tensor spec.
        """

    def tf_func_wrapper(func: Callable[P, R]) -> Callable[P, R]:
        """Allow retracing only in the main process.

        Joblib imports can lead to retracing tf.functions,
        although the functions are never called.

        Args:
            func: Function to wrap.

        Returns:
            Wrapped function.

        Raises:
            RuntimeError: If the function is called on a worker process.
        """
        return func

    def tf_kw_func_wrapper(
        **unused_kwargs: Any,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Allow retracing only in the main process.

        Joblib imports can lead to retracing tf.functions,
        although the functions are never called.

        Args:
            unused_kwargs: Not used.

        Returns:
            Wrapped function.

        Raises:
            RuntimeError: If the function is called on a worker process.
        """
        return tf_func_wrapper

elif _MAIN_PROCESS and COVERAGE_DEBUG:  # pragma: no cover
    # import the tensorflow for tests
    logger.debug("Wrap tf.function to avoid coverage masking")

    import tensorflow as tf  # type: ignore[unreachable]

    def _identity(func: Any | None = None, /, **kwargs: Any) -> Any:
        del kwargs  # not used
        if func is None:
            return _identity
        return func

    # don't use tf.function as its masks function calls for pytest-cov
    tf_tensor_spec = tf.TensorSpec
    tf_func_wrapper = _identity
    tf_kw_func_wrapper = _identity
    tf_dtype = tf.dtypes

elif _MAIN_PROCESS:  # pragma: no cover
    logger.debug("Using original tensorflow")

    # import tensorflow for usage
    import tensorflow as tf

    # use tf.function as it is
    tf_tensor_spec = tf.TensorSpec
    tf_func_wrapper = tf.function
    tf_kw_func_wrapper = tf.function
    tf_dtype = tf.dtypes

else:  # pragma: no cover
    # mock tensorflow for worker processes
    # class tf(metaclass=RecursiveModule):  # type: ignore[no-redef, unreachable]
    #    """Tensorflow mock class, which raises an error when accessed on workers."""
    logger.debug("Mocking tensorflow to avoid imports on workers")

    # find out why this is even imported

    def tf_tensor_spec(shape: tuple[()] | list[int | None], dtype: tf.DType) -> None:
        """Create a tensor spec from shape and dtype."""
        del shape, dtype  # not used

    def tf_func_wrapper(func: Callable[P, R]) -> Callable[P, R]:
        """Allow retracing only in the main process."""
        if TRACING_DEBUG:
            logger.debug(
                "tf_func_wrapper: %s calling %s",
                _mp.current_process().name,
                func.__name__,
            )

        def raise_on_call(*args: P.args, **kwargs: P.kwargs) -> R:
            del args, kwargs  # not used as we prohibit execution on workers
            raise RuntimeError("tf_func_wrapper is not available on worker processes")

        return raise_on_call

    def tf_kw_func_wrapper(
        **unused_kwargs: Any,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Allow retracing only in the main process."""
        return tf_func_wrapper

    class tf_dtype:  # type: ignore[unreachable,no-redef] # noqa: N801
        """Wrapper for tensorflow data types."""

        int32 = None
        int64 = None
        float32 = None
        float64 = None
        bool = None
