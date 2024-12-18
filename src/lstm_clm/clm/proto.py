"""Protocols for model classes."""

from __future__ import annotations  # pragma: no cover

from os import PathLike
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)  # pragma: no cover

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence
    from contextlib import AbstractContextManager

    import tensorflow as tf

    from lstm_clm.vocab import VocabProto

StrPath: TypeAlias = str | PathLike[str]
StrArray: TypeAlias = NDArray[np.str_]
Int32Array: TypeAlias = NDArray[np.int32]
Float32Array: TypeAlias = NDArray[np.float32]

T_co = TypeVar("T_co", covariant=True)

# string types for avoiding tensorflow import
DatasetT = TypeVar("DatasetT", bound="tf.data.Dataset")


class CallFunc(Protocol[T_co]):
    """Protocol for a callable that processes one RNN timestep.

    Args:
        x_t (tf.Tensor): Input tensor.
            If embedding: [Batch] {float32}
            Else: [Batch, Vocab] {float32}
        hidden_states (Sequence[Sequence[tf.Tensor]]):
            Initial RNN states. [2 x [Batch, Hidden] {float32}, ...]
        training (bool): Whether in training mode. Defaults to False.

    Returns:
        tuple[tf.Tensor, list[list[tf.Tensor]], T_co]:
            (output, hidden_states, info) where
            output: Model output. [Batch, Vocab] {float32}
            hidden_states: Updated RNN states. [2 x [Batch, Hidden] {float32}, ...]
            info: Additional information.
    """

    def __call__(
        self,
        x_t: tf.Tensor,
        hidden_states: Sequence[Sequence[tf.Tensor]],
        /,
        training: bool = False,
    ) -> tuple[tf.Tensor, list[list[tf.Tensor]], T_co]:
        """Process one RNN timestep."""


class SampleFunc(Protocol):
    """Protocol for sampling next tokens for a batch from a model output.

    Args:
        i (tf.Tensor): Current index. [1] {int32}
        x_t (tf.Tensor): Output at time step t. [Batch, Vocab] {float32}
        temp (tf.Tensor): Temperature factor. [1] {float32}

    Returns:
        tf.Tensor: Sampled tokens for timestep t. [Batch] {int32}
    """

    def __call__(self, i: tf.Tensor, x_t: tf.Tensor, /, temp: float) -> tf.Tensor:
        """Sample next tokens for a batch from a model output."""


class GenModel(Protocol[T_co]):
    """Protocol for a sample-generating model.

    Requirements:
    - Output a probability distribution (not logits)
    - Implement `call_cell` method to process one timestep
    - Implement `call_model` method to wrap the underlying TF model call
    """

    def call_cell(
        self,
        x_t: tf.Tensor,
        hidden_states: Sequence[Sequence[tf.Tensor]],
        /,
        training: bool = False,
    ) -> tuple[tf.Tensor, list[list[tf.Tensor]], T_co]:
        """Process one RNN timestep.

        Args:
            x_t: Input tensor.
                If embedding: [Batch] {float32}
                Else: [Batch, Vocab] {float32}
            hidden_states: Initial RNN states. [2 x [Batch, Hidden] {float32}, ...]
            training: Whether in training mode. Defaults to False.

        Returns:
            tuple[tf.Tensor, list[list[tf.Tensor]], T_co]:
                (output, hidden_states, info) where
                output: Model output. [Batch, Vocab] {float32}
                hidden_states: Updated RNN states.
                    [2 x [Batch, Hidden] {float32}, ...]
                info: Additional information.
        """

    def __call__(self, inputs: tf.Tensor, /, training: bool = False) -> tf.Tensor:
        """Wrap the underlying TensorFlow model call.

        Args:
        inputs: Input sequence.
            If embedding: [Batch, Length] {int32}
            Else: [Batch, Length, Vocab] {float32}
        training: Whether in training mode. Defaults to False.

        Returns:
            Output sequence. [Batch, Length, Vocab] {float32}
        """


class MultinomialModel(GenModel, Protocol):
    """Protocol for a model that generates samples using multinomial sampling.

    Requirements:
    - Implement `generate` method to return samples and probabilities
    - Implement `sample_next_tokens` method to sample next tokens
    """

    def generate(
        self,
        n_batches: int = 1,
        temp: float = 1.0,
        batch_size: int = 1024,
        call_func: CallFunc | None = None,
        sample_func: SampleFunc | None = None,
        verbose: int = 0,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Generate samples."""

    def sample_next_tokens(
        self, i: tf.Tensor, x_t: tf.Tensor, /, temp: float
    ) -> tf.Tensor:
        """Sample next tokens from the model."""


class VocabMultinomialModel(MultinomialModel, Protocol):
    """Protocol for a model using multinomial sampling with a vocabulary.

    Requirements:
    - Implement `generate` method to return samples and probabilities
    - Implement `sample_next_tokens` method to sample next tokens
    - Have `vocab` attribute, fulfilling :class:`VocabProto` protocol
    """

    vocab: VocabProto


class RandomizeFunc(Protocol):
    """Protocol for randomizing multiple SMILES strings.

    Args:
        smis (Sequence[str]): The SMILES strings to randomize.
        n_jobs (int): The number of jobs to use for parallelization.
        verbose (int): Verbosity level. Defaults to 0.

    Returns:
        list[str]: The randomized SMILES strings.
    """

    def __call__(
        self, smis: Sequence[str] | StrArray, n_jobs: int, verbose: int = 0
    ) -> list[str]:
        """Randomize multiple SMILES strings."""


class Factory(Protocol[T_co]):
    """Protocol for a factory function.

    A function which creates an instance
    of :class:`T_co` without any arguments
    """

    def __call__(self) -> T_co:
        """Create an instance of the class."""


class StrategyProto(Protocol):
    """Protocol which mirrors the :class:`tf.distribute.Strategy` layout.

    Implements the `experimental_distribute_dataset` and `scope` methods
    """

    def experimental_distribute_dataset(self, x: DatasetT) -> DatasetT:
        """Distribute the input dataset.

        Args:
            x: The input dataset

        Returns:
            The distributed dataset
        """

    def scope(self) -> AbstractContextManager[None]:
        """Return a context manager for the strategy scope.

        Returns:
            A context manager
        """


@runtime_checkable
class HistoryProto(Protocol):
    """Protocol which mirrors the :class:`tf.keras.callbacks.History` layout.

    Implements the `history` attribute.
    """

    history: dict[str, list[float]]


__all__ = [
    "CallFunc",
    "Factory",
    "GenModel",
    "HistoryProto",
    "MultinomialModel",
    "RandomizeFunc",
    "SampleFunc",
    "StrategyProto",
    "VocabMultinomialModel",
]
