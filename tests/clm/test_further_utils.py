"""Test tf utils."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import joblib
import loky
import pytest
from matplotlib.figure import Figure

from lstm_clm.clm.utils import NullStrategy, freeze_batch_norm, plot_history
from tests import _joblib_module

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable

    import tensorflow as tf
    from loky.reusable_executor import _ReusablePoolExecutor

    from lstm_clm.clm.utils import HistoryProto

R = TypeVar("R")

N_JOBS = 4


def auto_parallel(
    func: Callable[..., R], it: Iterable[Any | tuple[Any, ...]]
) -> list[R]:
    arg_list = ((x,) if not isinstance(x, tuple) else x for x in it)
    parallel = joblib.Parallel(n_jobs=4, return_as="list")
    jobs = (joblib.delayed(func)(*args) for args in arg_list)
    return parallel(jobs)  # type: ignore[no-any-return]


@pytest.fixture
def executor() -> _ReusablePoolExecutor:
    return loky.get_reusable_executor(max_workers=4)


@pytest.mark.timeout(0)
def test_tf_import(executor: _ReusablePoolExecutor) -> None:
    """Test that just import e.g. Vocabulary does not import tf on workers."""
    inputs = list(range(N_JOBS))
    func = _joblib_module.check_for_tensorflow_import

    for parallel_func in (auto_parallel, executor.map):
        workers_tf_imported = parallel_func(func, inputs)
        worker_names, tf_imported = zip(*workers_tf_imported, strict=True)
        assert len(set(worker_names)) == N_JOBS
        assert not any(tf_imported)


@pytest.fixture
def model() -> tf.keras.Model:
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, input_shape=(1,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1),
        ]
    )


def is_bn(layer: tf.keras.layers.Layer) -> bool:
    import tensorflow as tf

    return isinstance(layer, tf.keras.layers.BatchNormalization)


def test_freeze_batch_norm(model: tf.keras.Model) -> None:
    assert all(layer.trainable for layer in model.layers if is_bn(layer))
    model = freeze_batch_norm(model)
    assert not any(layer.trainable for layer in model.layers if is_bn(layer))


@pytest.fixture
def dataset() -> tf.data.Dataset:
    import tensorflow as tf

    return tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])


def test_null_strategy(dataset: tf.data.Dataset) -> None:
    strategy = NullStrategy()
    with strategy.scope() as scope:
        assert scope is None

    distributed = strategy.experimental_distribute_dataset(dataset)
    assert distributed is dataset


@pytest.fixture
def history() -> HistoryProto:
    class History:
        def __init__(self) -> None:
            self.history: dict[str, list[float]] = {
                "lr": [0.001, 0.001, 0.001, 0.001, 0.001],
                "loss": [0.5, 0.4, 0.3, 0.2, 0.1],
                "val_loss": [0.4, 0.3, 0.2, 0.1, 0.05],
            }

    return History()


def test_history_plot(history: HistoryProto) -> None:
    fig = plot_history(history)
    assert isinstance(fig, Figure)
