# %%
"""Utilities using randomized SMILES strings."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Self

from lstm_clm.clm.perplexity import perplexity_from_samples

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator, Sequence

    import tensorflow as tf

    from lstm_clm.clm.multinomial import MultinomialCLM
    from lstm_clm.clm.proto import Int32Array, RandomizeFunc
    from lstm_clm.vocab import VocabProto


class DatasetNotInitializedError(Exception):
    """Dataset is not initialized."""


class RandomizedDataset:
    """Creates a dataset to randomized SMILES strings.

    Usage:
        >>> mock_rnd_func = lambda smis, n_jobs, verbose: smis
        >>> smis = ["CCO", "CCC", "CCCC"]
        >>> vocabulary = Vocabulary(["C", "O"]), 140, "A")
        >>> rd = RandomizedDataset(smis, vocab, mock_rnd_func, n_jobs=1)
            rdd = rd.dataset(batch_size=1)
        >>> model.fit(rdd, steps_per_epoch=len(rdd), epochs=10)

    Attributes:
        smis (list[str]): The SMILES strings to randomize.
        vocab (VocabProto): The vocabulary to use for encoding.
        rnd_func (RandomizeFunc): The function to use for randomizing SMILES strings.
        n_jobs (int): The number of jobs to use for parallelization
        epoch (int): The current epoch.
        just_shuffle (bool): If True, only shuffles the SMILES strings (only canonical).
        n_precalc (int | None): If not None, precalcs the randomized SMILES
            for the first `n_precalc` epochs.
        verbose (int): Whether to show progress bars.
    """

    def __init__(
        self,
        smis: Sequence[str],
        vocab: VocabProto,
        rnd_func: RandomizeFunc,
        n_jobs: int,
        init_epoch: int = 0,
        just_shuffle: bool = False,
        n_precalc: int | None = None,
        verbose: int = 0,
    ) -> None:
        """Initialize the RandomizedDataset.

        Args:
            smis: The SMILES strings to randomize.
            vocab: The vocabulary to use for encoding.
            rnd_func: The function to use for randomizing SMILES strings.
            n_jobs: The number of jobs to use for parallelization.
            init_epoch: The initial epoch. Defaults to 0.
            just_shuffle: If True, only shuffles the SMILES strings (only canonical).
                Defaults to False.
            n_precalc:If not None, precalcs the randomized SMILES
                for the first `n_precalc` epochs. Defaults to None.
            verbose: Whether to show progress bars. Defaults to 0.
        """
        self.smis = list(smis)  # copy to avoid side effects

        self.vocab = vocab
        self.rnd_func = rnd_func

        # Initialize attributes
        self.n_jobs = n_jobs

        # Initialize epoch counter for save files
        self._init_epoch = init_epoch
        self.epoch = init_epoch

        self.just_shuffle = just_shuffle
        self.n_precalc = n_precalc
        self.verbose = verbose

        self._precalcd: list[int] = []

        self._encoded: Int32Array | None = None

        if self.n_precalc is not None:
            self.on_train_begin(self.n_precalc)

    @property
    def used(self) -> int:
        """The number of epochs used."""
        return self.epoch - self._init_epoch

    def __len__(self) -> int:
        """Returns the number of SMILES in the dataset."""
        return len(self.smis)

    def __call__(self) -> Iterator[Int32Array]:
        """Returns an iterator over the encoded SMILES."""
        # needs to be a generator for tf.data.Dataset.from_generator

        # Increment epoch counter
        self.epoch += 1

        if self.epoch not in self._precalcd:
            self.on_epoch_begin()  # Shuffle, randomize and encode SMILES

        enc = self.encoded

        yield from enc

    def build_dataset(
        self, batch_size: int, strategy: tf.distribute.Strategy | None = None
    ) -> BuiltRandomizedDataset:
        """Builds a tf.data.Dataset from the encoded SMILES.

        Args:
            batch_size: The batch size.
            strategy: The strategy to use for distributed training. Defaults to None.

        Returns:
            The built dataset.

        Raises:
            ValueError: If the dataset is already used.
        """
        if self.used:
            raise ValueError("Dataset already used. Create a new instance.")

        return BuiltRandomizedDataset(self, batch_size, strategy, self._init_epoch)

    def randomized_perplexity(
        self, gen: MultinomialCLM, n_repr: int, n_jobs: int, verbose: int = 0
    ) -> tf.Tensor:
        """Calculate the perplexity of the randomized SMILES.

        - Means over `N` randomizations per SMILES string.

        Args:
            gen: The generator.
            n_repr: The number of times to randomize each SMILES string.
            n_jobs: The number of jobs to use for parallelization.
            verbose: Verbosity level. Defaults to 0.

        Returns:
            The perplexity scores.
        """
        # repeat smiles -> smi1, smi1, smi1, smi2, smi2, smi2
        import tensorflow as tf

        smis = np.repeat(self.smis, n_repr)

        randomized = self.rnd_func(smis, n_jobs, self.verbose)

        encoded = self.vocab.encode(randomized)

        perplexity = perplexity_from_samples(gen, 1024, encoded, verbose=verbose)

        return tf.reshape(perplexity, (-1, n_repr))

    def on_epoch_begin(self) -> None:
        """Prepares the dataset for the next epoch.

        Called at the beginning of each epoch.
        Randomizes them if `just_shuffle` is False.
        Encodes them (with `just_shuffle` only in the first epoch).
        """
        # If just_shuffle, return here (don't randomize),

        if self.just_shuffle:
            if self._encoded is None:
                self._encoded = self.vocab.encode(self.smis)
            return

        rnd_smis = self.rnd_func(self.smis, self.n_jobs, verbose=self.verbose)
        self._encoded = self.vocab.encode(rnd_smis)

    def on_train_begin(self, epochs: int) -> None:
        """Prepares the dataset for training.

        Called at the beginning of training, if `n_precalc` is activated.
        Shuffles the SMILES strings.
        Randomizes them (if not `just_shuffle`). Encodes them.

        Args:
            epochs: The final epoch.
        """
        smiles: list[str] = np.tile(self.smis, epochs).tolist()
        # 4 already saved, initial epoch = 5, last_epoch 10 -> np.arange(5, 11)
        epoch_arr = np.arange(self.epoch, epochs) + 1

        if not self.just_shuffle:
            smiles = self.rnd_func(smiles, self.n_jobs, self.verbose)

        self._encoded = self.vocab.encode(smiles)

        self._precalcd = epoch_arr.tolist()

    @property
    def encoded(self) -> Int32Array:
        """The encoded data of the current epoch."""
        if self._encoded is None:
            raise DatasetNotInitializedError("Run build_dataset first.")

        curr_indices = np.random.permutation(len(self.smis))

        if self.epoch not in self._precalcd:
            return self._encoded[curr_indices]  # type: ignore[no-any-return]

        epoch_ix = self._precalcd.index(self.epoch)
        epoch_indices = np.arange(len(self.smis)) + epoch_ix * len(self.smis)
        epoch_encoded = self._encoded[epoch_indices]
        return epoch_encoded[curr_indices]  # type: ignore[no-any-return]

    @classmethod
    def build(
        cls,
        train: Sequence[str],
        val: Sequence[str],
        vocab: VocabProto,
        rnd_func: RandomizeFunc,
        n_jobs: int,
        init_epoch: int = 0,
        just_shuffle: bool = False,
        n_precalc: int | None = None,
        verbose: int = 0,
    ) -> tuple[Self, Self]:
        """Builds the train and val datasets for training.

        Args:
            train: The training data.
            val: The validation data.
            vocab: The vocabulary.
            rnd_func: The function to use for randomizing SMILES strings.
            n_jobs: The number of jobs.
            init_epoch: The initial epoch. Defaults to 0.
            just_shuffle: Whether to just shuffle the data. Defaults to False.
            n_precalc: If not None, precalcs the randomized SMILES
                for the first `n_precalc` epochs. Defaults to None.
            verbose: Verbosity level. Defaults to 0.

        Returns:
            The train and val datasets
        """
        args = (vocab, rnd_func, n_jobs, init_epoch, just_shuffle, n_precalc)
        train_dataset = cls(train, *args, verbose=verbose)
        val_dataset = cls(val, *args, verbose=verbose)
        return train_dataset, val_dataset


class BuiltRandomizedDataset:
    """Batched and distributed dataset for training.

    Warns:
        Built :class:`RandomizedDataset` is infinite.
        It will never stop yielding SMILES strings. Use with `steps_per_epoch`.

    Attributes:
        dataset (tf.data.Dataset | tf.distribute.DistributedDataset):
            Yields `batch_size`-sized batches of randomized SMILES strings.
        batch_size (int): The batch size.
        steps (int): The number of steps per epoch.
        init_epoch (int): The initial epoch.
    """

    def __init__(
        self,
        ds: RandomizedDataset,
        batch_size: int = 128,
        strategy: tf.distribute.Strategy | None = None,
        init_epoch: int = 0,
    ) -> None:
        """Creates a :class:`tf.data.Dataset` from the :class:`RandomizedDataset`.

        Args:
            ds: The RandomizedDataset
            batch_size: The batch size. Defaults to 128.
            strategy: The strategy. Defaults to None.
            init_epoch: The initial epoch. Defaults to 0.
        """
        import tensorflow as tf

        dataset: tf.data.Dataset | tf.distribute.DistributedDataset

        # always +1 for bos + potentially +1 for eos
        max_len_model = ds.vocab.max_len + 1 + (1 if ds.vocab.eos is not None else 0)
        dataset = (
            tf.data.Dataset.from_generator(
                ds,
                output_signature=(
                    tf.TensorSpec(shape=(max_len_model,), dtype=tf.int32)
                ),
            )
            .batch(batch_size)
            .map(lambda x: (x[:, :-1], x[:, 1:]))
            .prefetch(tf.data.experimental.AUTOTUNE)
            .repeat()
        )

        if strategy is not None:
            dataset = strategy.experimental_distribute_dataset(dataset)

        self.dataset = dataset
        self.batch_size = batch_size
        self.init_epoch = init_epoch
        self.steps = math.ceil(len(ds) / batch_size)


__all__ = ["BuiltRandomizedDataset", "RandomizedDataset"]
