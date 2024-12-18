"""The base class for further CLM models."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from typing_extensions import Self

from lstm_clm.clm.utils import lstm_model_call
from lstm_clm.vocab.tensor_impl import get_data_from_vocab

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    import tensorflow as tf

    from lstm_clm.clm.proto import StrPath
    from lstm_clm.vocab import VocabProto

MAX_LEN_BOUNDARY = 150


class BaseCLM:
    """Absolute base class for all CLM models.

    Implements a default call per time step and a default sample function.
    Compliance with the :class:`GenModel` protocol.

    Class variables:
        - ASSERT_VOCAB (bool): Whether to assert that a vocabulary is provided.

    Attributes:
        model (tf.keras.Model): Model to wrap with LSTM layers.
        vocab (Vocabulary | None): Vocabulary.
        vocab_size (int): Size of vocabulary.
        seq_len (int): Sequence length.
        dims (list[int]): Number of neurons per LSTM layers.
        has_embedding (bool): Whether the model has an Embedding layer.
        start_token (tf.Tensor): Start token for generation.
            If embedding: [1] {int32}
            Else: [1, Vocab] {float32}
    """

    ASSERT_VOCAB: ClassVar[bool] = False
    """@private"""

    def __init__(
        self,
        model: tf.keras.Model,
        max_len: int,
        start_index: int = 1,
        seq_len_append: int = 1,
        vocab: VocabProto | None = None,
    ) -> None:
        """Initialize BaseCLM.

        The start index is the index of the BOS token in the vocabulary.
        The sequence length append is 1 if the EOS token is present, else 0.
        The values are extracted from the vocabulary, if provided.

        Args:
            model: Model to wrap with LSTM layers.
            max_len: Maximum length of generated samples.
            start_index: Index of the start token. Defaults to 1.
            seq_len_append: Additional sequence length for EOS token. Defaults to 1.
            vocab: Vocabulary. Defaults to None.

        Raises:
            ValueError: If vocabulary does not match provided lengths.
            ValueError: If `ASSERT_VOCAB` is True and no vocabulary is provided.
        """
        import tensorflow as tf

        if self.ASSERT_VOCAB and vocab is None:
            raise ValueError("Vocabulary is required for this model")

        if max_len > MAX_LEN_BOUNDARY:  # pragma: no cover # TODO: Remove this
            raise ValueError("max_len should be less than 150")

        self.model = model
        """@private"""
        self.vocab = vocab
        """@private"""
        self.seq_len = max_len + seq_len_append
        """@private"""
        self.vocab_size: int = self.model.output_shape[-1]
        """@private"""

        self._validate_vocab(max_len, start_index, seq_len_append)

        # Get number of neurons in LSTM layers, used for initializing hidden states
        lstm_layers: list[tf.keras.layers.LSTM] = [
            layer
            for layer in self.model.layers
            if isinstance(layer, tf.keras.layers.LSTM)
        ]

        self.dims: list[int] = [layer.units for layer in lstm_layers]
        """@private"""

        # Check if model has Embedding layer
        self.has_embedding = any(
            isinstance(layer, tf.keras.layers.Embedding) for layer in self.model.layers
        )
        """@private"""

        start_token_emb = tf.constant([start_index], dtype=tf.int32)  # [1] {int32}
        start_token_one_hot = tf.one_hot(
            start_token_emb, depth=self.model.output_shape[-1]
        )  # [1, Vocab] {float32}

        # One-hot encode start tokens if model has no Embedding layer
        self.start_token: tf.Tensor = (
            start_token_emb if self.has_embedding else start_token_one_hot
        )
        """@private"""

    def _validate_vocab(
        self, max_len: int, start_index: int, seq_len_append: int
    ) -> None:
        """Validate the vocabulary against the provided lengths.

        Raises:
            ValueError: If vocabulary does not match provided lengths.
        """
        if self.vocab is not None:
            names = ["max_len", "start_index", "seq_len_append"]
            vocab_size, *vocab_data = get_data_from_vocab(self.vocab)

            if vocab_size != self.model.output_shape[-1]:
                raise ValueError(
                    "Vocabulary size does not match the last layers' output dims: "
                    f"{vocab_size} != {self.model.output_shape[-1]}"
                )

            for a, b, name in zip(
                vocab_data, (max_len, start_index, seq_len_append), names, strict=False
            ):
                if a != b:
                    raise ValueError(
                        f"Model/Vocab discrepancy: {name} does not match ({a} != {b})"
                    )

    @classmethod
    def from_vocab(cls, model: tf.keras.Model, vocab: VocabProto) -> Self:
        """Initialize :class:`BaseCLM` from a vocabulary.

        Args:
            model: Model to wrap with LSTM layers.
            vocab: Vocabulary.

        Returns:
            :class:`BaseCLM` instance.
        """
        _, max_len, start_index, seq_len_append = get_data_from_vocab(vocab)
        return cls(model, max_len, start_index, seq_len_append, vocab)

    def call_cell(
        self,
        x_t: tf.Tensor,
        hidden_states: Sequence[Sequence[tf.Tensor]],
        /,
        training: bool = False,
    ) -> tuple[tf.Tensor, list[list[tf.Tensor]], None]:
        """Call the model for one time step.

        Args:
            x_t: Input at time step t.
                If embedding: [Batch] {int32}
                Else: [Batch, Vocab] {float32}
            hidden_states: LSTM layers' hidden states.
                [2 x [Batch, Hidden] {float32}, ...]
            training: Whether in training mode. Defaults to False.

        Returns:
            tuple[tf.Tensor, list[list[tf.Tensor]], None]:
                output: Time step t output. [Batch, Vocab] {float32}
                states: LSTM layers' output states. [2 x [Batch, Hidden] {float32}, ...]
                info: Always None.
        """
        return lstm_model_call(
            self.model.layers, self.dims, x_t, hidden_states, training
        )

    def load_optim(
        self, path: StrPath, loss: tf.keras.losses.Loss | str, shape: tuple[int, ...]
    ) -> None:
        """Load optimizer from a file and compile the model.

        Important:
            If batch normalization should be freezed,
            use it before saving or loading the optimizer.

        Args:
            path: Path to the file.
            loss: Loss function.
            shape: Shape of the in- and output of the model.
        """
        import tensorflow as tf

        bytes_content = Path(path).read_bytes()
        data: tuple[dict[str, Any], list[np.ndarray]] = pickle.loads(bytes_content)
        config, weights = data

        opt = tf.keras.optimizers.Adam.from_config(config)
        self.model.compile(optimizer=opt, loss=loss)

        # fit to initialize variables
        prev_weights = self.model.get_weights()
        test = np.zeros((1, *shape), dtype=np.float32)
        self.model.fit(test, test, epochs=1, batch_size=1, verbose=0)
        # reset weights
        self.model.set_weights(prev_weights)

        # set optimizer weights
        opt.set_weights(weights)

        for a, b in zip(opt.variables(), weights, strict=True):
            if not np.allclose(a.numpy(), b):
                raise ValueError("Optimizer weights do not match")

    def save_optim(self, path: StrPath) -> None:
        """Save optimizer to a file.

        Important:
            use `freeze_batch_norm` before saving or loading the optimizer.

        Args:
            path: Path to the file.
        """
        opt: tf.keras.optimizers.Optimizer = self.model.optimizer
        config = opt.get_config()
        weights = [x.numpy() for x in opt.variables()]
        bytes_content = pickle.dumps((config, weights))
        Path(path).write_bytes(bytes_content)

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
        return self.model(inputs, training=training)

    def __getattr__(self, key: str) -> Any:
        """Get attribute from the underlying model.

        Example:
            >>> clm = BaseCLM(...)
            >>> lstm_clm.summary()
            # is equivalent to lstm_clm.model.summary()
            ...

        Args:
            key: Attribute name.

        Returns:
            Attribute value.
        """
        return getattr(self.model, key)

    if TYPE_CHECKING:  # pragma: no cover
        # type hints some wrapped methods
        # could be much longer - only the most used methods are included

        def load_weights(self, path: StrPath) -> None:
            """Load the model weights from a file.

            Args:
                path: Path to the file.
            """

        def save_weights(self, path: StrPath) -> None:
            """Save the model weights to a file.

            Args:
                path: Path to the file.
            """


__all__ = ["BaseCLM"]
