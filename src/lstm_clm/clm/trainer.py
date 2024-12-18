"""Module to train a model with a :class:`rnd_utils.RandomizedDataset`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lstm_clm.clm.randomized import RandomizedDataset
from lstm_clm.clm.utils import NullStrategy, build_model_cp

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence
    from pathlib import Path

    import tensorflow as tf

    from lstm_clm.clm.multinomial import MultinomialCLM
    from lstm_clm.clm.proto import Factory, RandomizeFunc, StrategyProto
    from lstm_clm.vocab import VocabProto


def get_strategy(multi_gpu: bool = False) -> StrategyProto:
    """Get the strategy.

    Args:
        multi_gpu: Whether to use multiple GPUs. Defaults to False.

    Returns:
        The strategy
    """
    import tensorflow as tf

    if len(tf.config.list_physical_devices("GPU")) > 1 and multi_gpu:
        return tf.distribute.MirroredStrategy()  # type: ignore[no-any-return]

    return NullStrategy()


class Trainer:
    """Convenience class to train a model with a :class:`RandomizedDataset`."""

    @staticmethod
    def fit(
        model: tf.keras.Model,
        train: RandomizedDataset,
        val: RandomizedDataset,
        epochs: int,
        batch_size: int,
        callbacks: list[tf.keras.callbacks.Callback],
        strategy: tf.distribute.Strategy | None = None,
    ) -> tf.keras.callbacks.History:
        """Fit the model.

        Args:
            model: The model
            train: The training dataset
            val: The validation dataset
            epochs: The number of epochs
            batch_size: The batch size
            callbacks: The list of callbacks
            strategy: The strategy. Defaults to None.

        Returns:
            The history of the training
        """
        built_train = train.build_dataset(batch_size, strategy)
        built_val = val.build_dataset(batch_size, strategy)

        return model.fit(
            built_train.dataset,
            epochs=epochs,
            steps_per_epoch=built_train.steps,
            validation_data=built_val.dataset,
            validation_steps=built_val.steps,
            callbacks=callbacks,
            initial_epoch=built_train.init_epoch,
        )

    @staticmethod
    def callbacks(
        clm: MultinomialCLM,
        train: RandomizedDataset,
        val: RandomizedDataset,
        cp: tf.keras.callbacks.ModelCheckpoint | None = None,
        calc_jsd: bool = False,
        decay: tf.keras.callbacks.ReduceLROnPlateau | None = None,
        models_path: Path | str | None = None,
        early_stopping: tf.keras.callbacks.EarlyStopping | None = None,
        additional_cbs: dict[int, list[tf.keras.callbacks.Callback]] | None = None,
    ) -> list[tf.keras.callbacks.Callback]:
        """Get the callbacks for training.

        - Additional positions:
            - 0 at beginning,
            - 1 after JSD,
            - 2 after decay,
            - 3 after checkpoint,
            - 4 after trainingCP,
            - 5 after early stopping
        - TODO: make the int as enum for add_cbs.

        Args:
            clm: The CLM
            train: The training dataset
            val: The validation dataset
            cp: The checkpoint to load. Defaults to None.
            calc_jsd: Whether to calc the JSD. Defaults to False.
            decay: The decay storer. Defaults to None.
            models_path: The _path to store the models. Defaults to None.
            early_stopping: The early stopping. Defaults to None.
            additional_cbs: Additional callbacks to add at specific positions.
                Defaults to empty dict.

        Returns:
            The list of callbacks
        """
        add_cbs = additional_cbs if additional_cbs is not None else {}

        callbacks: list[tf.keras.callbacks.Callback] = add_cbs.get(0, [])

        if calc_jsd:
            # import here to don't import tensorflow on import
            from lstm_clm.callbacks.jsd import JSDCallback

            callbacks.append(JSDCallback(clm, train, val))

        callbacks.extend(add_cbs.get(1, []))

        if decay is not None:
            callbacks.append(decay)

        callbacks.extend(add_cbs.get(2, []))

        if models_path is not None:
            callbacks.append(build_model_cp(models_path))

        callbacks.extend(add_cbs.get(3, []))

        if cp is not None:
            callbacks.append(cp)

        callbacks.extend(add_cbs.get(4, []))

        if early_stopping is not None:
            callbacks.append(early_stopping)

        callbacks.extend(add_cbs.get(5, []))

        return callbacks

    @staticmethod
    def init(
        build_model: Factory[tf.keras.Model],
        build_opt: Factory[tf.keras.Model],
        multi_gpu: bool = False,
    ) -> tuple[tf.keras.Model, StrategyProto]:
        """Initializes the model and strategy.

        Args:
            build_model: Function to build the model
            build_opt: Function to build the optimizer
            multi_gpu: Whether to use multiple GPUs. Defaults to False.

        Returns:
            A tuple with the model, strategy, initial epoch, and decay storer/None
        """
        strategy = get_strategy(multi_gpu)

        with strategy.scope():
            # need to build model and optimizer in scope (or load it in scope) IMPORTANT
            model = build_model()
            opt = build_opt()
            model.compile(loss="sparse_categorical_crossentropy", optimizer=opt)

        return model, strategy

    @staticmethod
    def build(
        train: Sequence[str],
        val: Sequence[str],
        vocab: VocabProto,
        rnd_func: RandomizeFunc,
        epochs: int,
        n_jobs: int,
        just_shuffle: bool = False,
        init_epoch: int = 0,
        finetune: bool = False,
        verbose: int = 0,
    ) -> tuple[RandomizedDataset, RandomizedDataset]:
        """Builds the train and val _datasets for training.

        Args:
            train: The training data.
            val: The validation data.
            vocab: The vocabulary.
            rnd_func: Function to randomize the SMILES strings.
            epochs: The number of epochs.
            n_jobs: The number of jobs.
            just_shuffle: Whether to just shuffle the data. Defaults to False.
            init_epoch: The initial epoch. Defaults to 0.
            finetune: Whether its finetune data and thus
                data is precalc'd. Defaults to False.
                Defaults to 4.
            verbose: Verbosity level. Defaults to 0.

        Returns:
            A tuple with train and val _datasets
        """
        n_precalc = epochs if finetune else None

        built_train, built_val = RandomizedDataset.build(
            train,
            val,
            vocab,
            rnd_func,
            n_jobs,
            init_epoch,
            just_shuffle,
            n_precalc,
            verbose=verbose,
        )

        return built_train, built_val


__all__ = ["Trainer", "get_strategy"]
