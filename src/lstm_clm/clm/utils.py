"""Utility functions for the CLM generator."""

from __future__ import annotations

import logging
import platform
from contextlib import nullcontext
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from matplotlib.figure import Figure
from tqdm import tqdm

from lstm_clm.clm.proto import DatasetT, HistoryProto, StrPath
from lstm_clm.clm.wrapper import (
    tf_dtype,
    tf_func_wrapper,
    tf_kw_func_wrapper,
    tf_tensor_spec,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Sequence

    import tensorflow as tf
    from matplotlib.axes import Axes


logger = logging.getLogger(__package__)

# string types for avoiding tensorflow import
BatchableT = TypeVar("BatchableT", "tf.Tensor", "tuple[tf.Tensor, ...]")


def allow_memory_growth() -> None:
    """Allow memory growth for TensorFlow GPU devices."""
    # Allow memory growth
    # Only in main process
    import tensorflow as tf

    if gpus := tf.config.list_physical_devices("GPU"):
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            logger.exception("Failed to set memory growth")


def build_model_cp(
    models_path: StrPath,
    scheme: str = "{epoch}.h5",
    best_only: bool = False,
    weights_only: bool = True,
) -> tf.keras.callbacks.ModelCheckpoint:
    """Get the model checkpoint callback.

    Args:
        models_path: Path to save the models
        scheme: Scheme to save the models. Defaults to "{epoch}.h5".
        best_only: Save only the best model. Defaults to False.
        weights_only: Save only the weights. Defaults to True.

    Returns:
        The model checkpoint callback
    """
    import tensorflow as tf

    return tf.keras.callbacks.ModelCheckpoint(
        str(Path(models_path) / scheme),
        save_best_only=best_only,
        save_weights_only=weights_only,
    )


def build_adam_optimizer(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    """Return an Adam optimizer with the given learning rate.

    As there are some issues with the Adam optimizer on Apple M1,
    we use the legacy version in this case.

    Args:
        learning_rate: Initial learning rate.

    Returns:
        Adam optimizer.
    """
    import tensorflow as tf

    if platform.system() == "Darwin" and platform.processor() == "arm":
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def batch_tensor_slices(
    data: BatchableT, batch_size: int, desc: str | None = None, verbose: int = 0
) -> Iterable[BatchableT]:
    """Batch tensor slices to :class:`tf.data.Dataset`.

    Args:
        data: Data to batch. Either a single tensor or a tuple! of tensors.
        batch_size: Batch size.
        desc: Description for tqdm. Defaults to None.
        verbose: Verbosity level. Defaults to 0.

    Returns:
        A iterable over a :class:`tf.data.Dataset` of batched data.
    """
    import tensorflow as tf

    dataset = (
        tf.data.Dataset.from_tensor_slices(data)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    if verbose > 0:
        batches = ceil(len(data) / batch_size)
        return tqdm(dataset, desc=desc, total=batches)  # type: ignore[no-any-return]

    return dataset  # type: ignore[no-any-return]


def freeze_batch_norm(model: tf.keras.Model) -> tf.keras.Model:
    """Set all BatchNormalization layers in model to non-trainable (in-place).

    Args:
        model: The model

    Returns:
        The model with all BatchNormalization layers set to non-trainable
    """
    import tensorflow as tf

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    return model


@tf_func_wrapper  # no input signature as lstm_dims: ListTraceType is not supported
def init_hidden_states(
    lstm_dims: Sequence[int], batch_size: int
) -> list[list[tf.Tensor]]:
    """Initialize LSTM hidden states with zeros (tf.function wrapped).

    Args:
        lstm_dims: Neurons per LSTM layer.
        batch_size: Batch size.

    Returns:
        Initialized hidden states. [[2 x [Batch, Hidden] {float32}], ...]
    """
    import tensorflow as tf

    return [[tf.zeros([batch_size, d]), tf.zeros([batch_size, d])] for d in lstm_dims]


@tf_func_wrapper
def lstm_model_call(
    layers: Sequence,
    lstm_dims: Sequence[int],
    x_t: tf.Tensor,
    hidden_states: Sequence[Sequence[tf.Tensor]],
    training: bool = False,
) -> tuple[tf.Tensor, list[list[tf.Tensor]], None]:
    """Call a model with LSTM layers for one time step (tf.function wrapped).

    Args:
        layers: Model layers.
        lstm_dims: Neurons per LSTM layer.
        x_t: Input at time step t.
            If embedding: [Batch] {int32}
            Else: [Batch, Vocab] {float32}
        hidden_states: LSTM layers' hidden states. [2 x [Batch, Hidden] {float32}, ...]
        training: Whether in training mode. Defaults to False.

    Returns:
        tuple[tf.Tensor, list[list[tf.Tensor]], None]:
            output: Time step t output. [Batch, Vocab] {float32}
            states: LSTM layers' output states. [2 x [Batch, Hidden] {float32}, ...]
            info: Always None.
    """
    import tensorflow as tf

    # Set initial LSTM index to 0
    lstm_ix = 0

    # Create list for output states
    output_states: list[list[tf.Tensor]] = [[] for _ in lstm_dims]

    # Loop through layers in the model
    for layer in layers:
        # Apply batch normalization or time distributed layer
        # Expand, apply, squeeze as these layers expect 3D input
        if isinstance(
            layer, tf.keras.layers.BatchNormalization | tf.keras.layers.TimeDistributed
        ):
            x_t = tf.expand_dims(x_t, axis=1)
            x_t = layer(x_t, training=training)
            x_t = tf.squeeze(x_t, axis=1)

        # Apply LSTM layer
        elif isinstance(layer, tf.keras.layers.LSTM):
            x_t, curr_states = layer.cell(
                x_t, hidden_states[lstm_ix], training=training
            )

            output_states[lstm_ix] = curr_states
            lstm_ix += 1

        # Apply other layer types
        else:
            x_t = layer(x_t, training=training)

    # Return output and output states, and None for compatibility with subclasses
    return x_t, output_states, None


def init_tensor_array(dtype: tf.DType, seq_len: int) -> tf.TensorArray:
    """Initialize TensorArray for generated samples.

    Args:
        dtype: Data type (tf.int32 or tf.float32).
        seq_len: Sequence length.

    Returns:
        Initialized TensorArray. [Length] {dtype}
    """
    # do not use tf.function here: else it will be a tf.Tensor at runtime!!!
    import tensorflow as tf

    return tf.TensorArray(
        dtype=dtype, size=seq_len, dynamic_size=False, infer_shape=False
    )


@tf_kw_func_wrapper(
    input_signature=[
        tf_tensor_spec(shape=[None, None], dtype=tf_dtype.float32),
        tf_tensor_spec(shape=(), dtype=tf_dtype.float64),
    ]
)
def multinomial_sample(x_t: tf.Tensor, /, temp: float) -> tf.Tensor:
    """Sample from multinomial distribution (tf.function wrapped).

    Args:
        x_t: Probability distribution at time step t. [Batch, Vocab] {float32}
        temp: Temperature factor. [1] {float64}

    Returns:
        Sampled tokens. [Batch] {int32}
    """
    import tensorflow as tf

    # Adjust probabilities with Temperature factor and better precision with float64
    logits = tf.math.log(tf.cast(x_t, tf.float64))
    return categorical_sample(logits, temp)


@tf_kw_func_wrapper(
    input_signature=[
        tf_tensor_spec(shape=[None, None], dtype=tf_dtype.float64),
        tf_tensor_spec(shape=(), dtype=tf_dtype.float64),
    ]
)
def categorical_sample(logits: tf.Tensor, /, temp: float) -> tf.Tensor:
    """Sample from logits (tf.function wrapped).

    Args:
        logits: Logits. [Batch, Vocab] {float64}
        temp: Temperature factor. [1] {float64}

    Returns:
        Sampled tokens. [Batch] {int32}
    """
    import tensorflow as tf

    logits /= temp  # Rescale logits with Temperature factor
    next_tokens = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
    return tf.squeeze(next_tokens, axis=-1)  # Remove last dimension


def plot_history(history_data: HistoryProto | dict[str, Sequence[float]]) -> Figure:
    """Plot training history to file.

    Calculates the best epoch and loss from the validation data.

    Args:
        history_data: Keras history object or dictionary of metrics

    Returns:
        Image object
    """
    # - include a scaled LR plot, if LR is available in the history
    # - accept dictionary of metrics to plot
    history: dict[str, Sequence[float]] = (
        history_data.history  # type: ignore[assignment]
        if isinstance(history_data, HistoryProto)
        else history_data
    )

    fig = Figure(figsize=(10, 6))
    ax: Axes = fig.add_subplot(111)
    losses = history["loss"]
    val_losses = history["val_loss"]
    ax.plot(losses, label="loss")
    ax.plot(val_losses, label="val_loss")
    if "lr" in history:
        lrs = history["lr"]
        ax2: Axes = ax.twinx()
        ax2.plot(lrs, color="C2")
        ax.plot([], color="C2", label="lr")
        ax2.set_ylabel("lr")

    best_epoch = int(np.argmin(val_losses)) + 1
    best_loss = min(val_losses)
    ax.set_title(f"Best epoch: {best_epoch}, val_loss: {best_loss:.3f}")
    ax.legend()
    fig.tight_layout()
    return fig


class NullStrategy:
    """Null strategy for single GPU training.

    As a placeholder for :class:`tf.distribute.Strategy`.
    """

    def experimental_distribute_dataset(self, x: DatasetT) -> DatasetT:
        """Returns the input dataset as is.

        Args:
            x: The input dataset

        Returns:
            The input dataset
        """
        return x

    def scope(self) -> nullcontext[None]:
        """Returns nullcontext.

        Returns:
            A null context manager
        """
        return nullcontext()


__all__ = [
    "NullStrategy",
    "allow_memory_growth",
    "batch_tensor_slices",
    "build_adam_optimizer",
    "build_model_cp",
    "categorical_sample",
    "init_hidden_states",
    "init_tensor_array",
    "lstm_model_call",
    "multinomial_sample",
    "plot_history",
]
