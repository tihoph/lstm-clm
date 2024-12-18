"""Custom callbacks for metrics."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Protocol

import numpy as np
import tensorflow as tf

from lstm_clm.clm.perplexity import batch_tensor_slices

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from lstm_clm.clm.multinomial import MultinomialCLM
    from lstm_clm.clm.proto import Float32Array, Int32Array, NDArray

logger = logging.getLogger(__package__)


def jensen_shannon_divergence(*dists: Sequence[float]) -> float:
    """Calculates the Uniformity-completeness Jensen-Shannon divergence.

    According to:
        ```
        Arús-Pous, J., Johansson, S.V., Prykhodko, O. et al.
        Randomized SMILES strings improve the quality of molecular generative models.
        J Cheminform 11, 71 (2019).
        ```

    - https://doi.org/10.1186/s13321-019-0393-0
    - Implementation from https://github.com/undeadpixel/reinvent-randomized

    Args:
        *dists: List of distributions to calc the divergence of
            (must be of the same length)

    Returns:
        Calculated Jensen-Shannon divergence
    """
    import scipy.stats

    num_dists = len(dists)
    if num_dists < 2:  # noqa: PLR2004
        raise ValueError("At least two distributions are required to calc the JSD")
    if not all(len(dist) == len(dists[0]) for dist in dists):
        raise ValueError("All distributions must have the same length")

    avg_dist: Float32Array = np.sum(dists, axis=0, dtype=np.float32) / num_dists
    entropies: list[list[float]] = [
        scipy.stats.entropy(dist, avg_dist).tolist() for dist in dists
    ]
    return float(np.sum(entropies) / num_dists)


class _DatasetProto(Protocol):
    """Protocol for a dataset.

    Implements a `encoded` property which returns an array
    Also possible with changing content (e.g. randomized dataset)
    """

    @property
    def encoded(self) -> Int32Array:
        """The encoded dataset."""


class JSDCallback(tf.keras.callbacks.Callback):
    """Callback to calculate the Jensen-Shannon divergence.

    JSD is calculatet between the NLL of the samples
    and the NLL of the training and validation sets.
    If `train` and `val` are not provided,
    the callback can be used as a standalone metric
    with a fixed dataset using `get_jsd`.

    Attributes:
        gen (GeneratorWrapper): The generator.
        train (_DatasetProto | None): The training dataset if used as a callback.
        val (_DatasetProto | None): The validation dataset if used as a callback.
        sample_size (int): The sample size.
        batch_size (int): The batch size.
        verbose (int): Verbosity level.
    """

    def __init__(
        self,
        gen: MultinomialCLM,
        train: _DatasetProto | None = None,
        val: _DatasetProto | None = None,
        sample_size: int = 10000,
        batch_size: int = 1024,
        verbose: int = 0,
    ) -> None:
        """Initialize the JSDCallback.

        Args:
            gen: The generator.
            train: The training dataset. Defaults to None.
            val: The validation dataset. Defaults to None.
            sample_size: The sample size. Defaults to 10000.
            batch_size: The batch size. Defaults to 1024.
            verbose: Verbosity level. Defaults to 0.
        """
        super().__init__()
        self.gen = gen
        """@private"""
        self.train = train
        """@private"""
        self.val = val
        """@private"""
        self.sample_size = sample_size
        """@private"""
        self.batch_size = batch_size
        """@private"""
        self.verbose = verbose
        """@private"""

        if train is None or val is None:
            logger.warning(
                "Can't be used as a callback, "
                "only as a standalone metric with a fixed dataset"
            )
            self.model = self.gen.model

    def _sample(self, n_samples: int, temp: float = 1.0) -> tuple[tf.Tensor, tf.Tensor]:
        """Sample from the generator."""
        n_batches = math.ceil(n_samples / self.batch_size)
        samples, preds = self.gen.generate(
            n_batches, temp, self.batch_size, verbose=self.verbose
        )
        return samples[:n_samples], preds[:n_samples]

    def _predict_nll(self, encoded: Float32Array) -> Float32Array:
        """Returns the NLL of the encoded dataset."""
        replace = len(encoded) < self.sample_size
        choices = np.random.choice(len(encoded), self.sample_size, replace=replace)
        encoded = encoded[choices]

        x_true, y_true = encoded[:, :-1], encoded[:, 1:]
        dataset = batch_tensor_slices(
            x_true, self.batch_size, "Predict NLL:", verbose=self.verbose
        )
        y_pred_arr = [self.model(x_batch) for x_batch in dataset]
        y_pred = tf.concat(y_pred_arr, axis=0)
        return self._get_nll(y_true, y_pred)

    def _get_nll(
        self, y_true: Float32Array | tf.Tensor, y_pred: Float32Array | tf.Tensor
    ) -> Float32Array:
        """Calculates the negative log likelihood."""
        nll = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        nll = tf.reduce_sum(nll, axis=-1)
        return nll.numpy()  # type:ignore[no-any-return]

    def get_jsd(
        self, samples: tf.Tensor, preds: tf.Tensor, train: NDArray, val: NDArray
    ) -> tuple[float, Float32Array, Float32Array, Float32Array]:
        """Get Jensen-Shannon divergence between the provided sets.

        It uses the NLL of the samples and the NLL of the training and validation sets.

        Args:
            samples: The samples
            preds: The predictions
            train: The training set
            val: The validation set

        Returns:
            A tuple with the calculated Jensen-Shannon divergence,
            and the NLLs of the training and validation sets, and the samples
        """
        sample_nll = self._get_nll(samples[:, 1:], preds)
        nll = self._predict_nll(train)
        val_nll = self._predict_nll(val)
        jsd = jensen_shannon_divergence(
            sample_nll.tolist(), nll.tolist(), val_nll.tolist()
        )
        return jsd, nll, val_nll, sample_nll

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> dict:
        """Calculate the Jensen-Shannon divergence after every epoch.

        Args:
            epoch: The epoch number. Unused.
            logs: The logs. Defaults to None.

        Returns:
            The logs
        """
        del epoch  # unused
        if self.train is None or self.val is None:
            raise ValueError(
                "Can't be used as a callback, ",
                "only as a standalone metric with a fixed dataset",
            )

        if logs is None:
            logs = {}

        samples, preds = self._sample(self.sample_size)

        logs["jsd"], logs["nll"], logs["val_nll"], logs["sample_nll"] = self.get_jsd(
            samples, preds, self.train.encoded, self.val.encoded
        )

        return logs


__all__ = ["JSDCallback", "jensen_shannon_divergence"]
