# %%
"""Implementation of the Vocabulary class with support for :class:`tf.Tensor`.

Includes tensorflow dependencies. They are imported on call.
"""

from __future__ import annotations

import logging
import string
from typing import TYPE_CHECKING, Any

import numpy as np

from lstm_clm.vocab.decode import build_decode_map, build_full_decode_func, decode_multi
from lstm_clm.vocab.encode import REGEX_PATTERNS, build_full_encode_func, encode_multi

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    import tensorflow as tf

    from lstm_clm.vocab.proto import (
        DecodableBatch,
        EncodableBatch,
        Int32Array,
        NDArray,
        TokenizedBatch,
        VocabProto,
    )

logger = logging.getLogger(__package__)


def _get_unused_char(decode_map: dict[int, str]) -> str:
    """Returns a unused char to mark unknown characters.

    Args:
        decode_map: Mapping from indices to tokens.

    Returns:
        A character which is not in the decode map.

    Raises:
        IndexError: If all lowercase and uppercase letters are used.
    """
    all_chars = set(string.ascii_lowercase + string.ascii_uppercase)
    ununused_chars = sorted(all_chars - set(decode_map.values()))
    return ununused_chars[0]


def _create_hash_table(
    lookup_map: dict, key_dtype: tf.DType, value_dtype: tf.DType, default_value: Any
) -> tf.lookup.StaticHashTable:
    """Create a static hash table from a dictionary.

    Args:
        lookup_map: The lookup dictionary.
        key_dtype: Key type.
        value_dtype: Value type.
        default_value: Value if key not found, same type as values.

    Returns:
        The static hash table
    """
    import tensorflow as tf

    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            list(lookup_map.keys()),
            list(lookup_map.values()),
            key_dtype=key_dtype,
            value_dtype=value_dtype,
        ),
        default_value,
    )


def _tensor_lookup(inputs: tf.Tensor, hash_table: tf.lookup.StaticHashTable) -> NDArray:
    """Batch lookups inputs from a hash table.

    Args:
        inputs: Inputs to lookup (1D if encode, 2D if decode)
        hash_table: Hash table (types depending on mode)

    Returns:
        Array with either strings or indices
    """
    # Split inputs into chunks of 500k inputs
    import tensorflow as tf

    arrays: list[NDArray] = []

    # Lookup each chunk and append to output
    for start_index in range(0, len(inputs), int(5 * 1e4)):
        stop_index = start_index + int(5 * 1e4)
        batch = inputs[start_index:stop_index]
        batch = tf.convert_to_tensor(batch, dtype=hash_table.key_dtype)
        batch = hash_table.lookup(batch)

        # If decode, join the strings and convert to np.str_
        if hash_table.key_dtype == tf.int32:
            if tf.size(tf.where(batch == hash_table.default_value)) > 0:
                raise KeyError("Unknown character in input")
            batch = tf.strings.reduce_join(batch, axis=1).numpy().astype(str)
        else:
            batch = batch.numpy()

        arrays.append(batch)

    # Concatenate output and return
    return np.concatenate(arrays)


def _encode_tensor(
    inputs: tf.Tensor, encode_map: dict[str, int], unk: str
) -> Int32Array:
    import tensorflow as tf

    unk_ix = encode_map[unk]
    hash_table = _create_hash_table(
        encode_map, tf.string, tf.int32, unk_ix
    )  # TODO: reuse?
    return _tensor_lookup(inputs, hash_table)


def _encode_multi(
    inputs: TokenizedBatch, encode_map: dict[str, int], unk: str
) -> Int32Array:
    import tensorflow as tf

    if isinstance(inputs, tf.Tensor):
        return _encode_tensor(inputs, encode_map, unk)
    return encode_multi(inputs, encode_map, unk)


def _decode_tensor(inputs: tf.Tensor, decode_map: dict[int, str]) -> list[str]:
    # Find a char which is not in the decode map
    import tensorflow as tf

    new_char = _get_unused_char(decode_map)
    hash_table = _create_hash_table(
        decode_map, tf.int32, tf.string, new_char
    )  # TODO: reuse?
    decoded = _tensor_lookup(inputs, hash_table)
    return decoded.tolist()  # type: ignore[no-any-return]


def _decode_multi(inputs: DecodableBatch, decode_map: dict[int, str]) -> list[str]:
    import tensorflow as tf

    if isinstance(inputs, tf.Tensor):
        return _decode_tensor(inputs, decode_map)

    return decode_multi(inputs, decode_map)


class Vocabulary:
    """Vocabulary class for tokenizing and encoding SMILES strings.

    Follows the :class:`VocabProto` protocol.

    With different padding directions, other tokenization methods and more.
    We have additional options in comparison to just the protocol.

    Attributes:
        tokens (list[str]): Vocabulary to use for tokenization and encoding
        max_len (int): Maximum length of the model
        pad (str): Padding character
        bos (str): Start character
        eos (str | None): End character.
        unk (str | None): Unknown character. If None, `pad` is used.
        encode_map (dict[str, int]): Mapping from tokens to indices
        decode_map (dict[int, str]): Mapping from indices to tokens
        special_tokens (list[str]): List of special tokens
        right_pad (bool): If True, pads to the right, else pads to the left.
        only_two_char_tokens (bool): If True, only uses tokenization
            for two character tokens ([Se] -> '[', 'Se' ']').
        assert_length (bool): If True, raises an error if
            a SMILES string is longer than max_len_model.
        assert_known (bool): If True, raises an error if a token is unknown.
    """

    def __init__(
        self,
        tokens: Sequence[str],
        max_len: int,
        pad: str,
        bos: str,
        eos: str | None = None,
        unk: str | None = None,
        right_pad: bool = True,
        only_two_char_tokens: bool = False,
        assert_length: bool = False,
        assert_known: bool = False,
    ) -> None:
        """Initialize the vocabulary.

        Args:
            tokens: Vocabulary to use for tokenization and encoding
            max_len: Maximum length of the model
            pad: Padding character.
            bos: Start character.
            eos: End character. Defaults to None.
            unk: Unknown character. If None and assert_known is False, `pad` is used.
                Defaults to None.
            right_pad: If True, pads to the right, else pads to the left.
                Defaults to True.
            only_two_char_tokens: If True, only uses tokenization for two
                character tokens ([Se] -> '[', 'Se' ']'). Defaults to False.
            assert_length: If True, raise if a SMILES string is longer than max_len.
                Defaults to False.
            assert_known: If True, raise if a token is unknown. Defaults to False.

        Raises:
            ValueError: If tokens are not unique
            ValueError: If special tokens are not in the set of tokens
        """
        if len(tokens) != len(set(tokens)):
            raise ValueError("Tokens must be unique")

        self.tokens = list(tokens)
        """@private"""

        self.max_len = max_len
        """@private"""

        if unk is None and not assert_known:
            unk = pad

        self.pad = pad
        """@private"""

        self.bos = bos
        """@private"""
        self.eos = eos
        """@private"""
        self.unk = unk
        """@private"""

        self.special_tokens = [x for x in (pad, bos, eos, unk) if x is not None]
        """@private"""
        self.decode_map = dict(enumerate(self.tokens))
        """@private"""
        self.encode_map = {tok: ix for ix, tok in self.decode_map.items()}
        """@private"""

        if not set(self.special_tokens).issubset(set(self.tokens)):
            raise ValueError("Special tokens must be in the set of tokens")

        self.right_pad = right_pad
        """@private"""
        self.only_two_char_tokens = only_two_char_tokens
        """@private"""
        self.assert_length = assert_length
        """@private"""
        self.assert_known = assert_known
        """@private"""

        # Function methods are not well supported for parallelization
        # so we use partial functions instead
        # we hide it behind a property to avoid confusion

        pattern = REGEX_PATTERNS[only_two_char_tokens]

        self.encode = build_full_encode_func(  # type: ignore[method-assign, assignment]
            self.encode_map,
            self.pad,
            self.bos,
            self.eos,
            self.unk,
            pattern,
            self.max_len,
            self.right_pad,
            self.assert_length,
        )

        adjusted_decode_map = build_decode_map(
            self.decode_map, self.encode_map, self.special_tokens
        )

        self.decode = build_full_decode_func(  # type: ignore[method-assign, assignment]
            adjusted_decode_map, decode_func=_decode_multi
        )

    def __len__(self) -> int:
        """@public Returns number of vocab tokens."""
        return len(self.tokens)

    def encode(self, inputs: EncodableBatch) -> Int32Array:  # pylint: disable=method-hidden
        """Encode a list of tokens to indices.

        Args:
            inputs: Token inputs. Can be a list/np.ndarray/tf.Tensor of strings.

        Returns:
            Encoded indices. [Input Length x Sequence Length]
        """
        logger.warning("Called as method, thus not pickable")
        return self.encode(inputs)

    def decode(self, inputs: DecodableBatch) -> list[str]:  # pylint: disable=method-hidden
        """Decode a list of indices to tokens.

        Args:
            inputs: Indices inputs.
                Can be a list of integer lists or a np.ndarray/tf.Tensor (int32).

        Returns:
            Decoded tokens. [Input Length]
        """
        logger.warning("Called as method, thus not pickable")
        return self.decode(inputs)


def get_data_from_vocab(vocab: VocabProto) -> tuple[int, int, int, int]:
    """Get the settings stored in a vocabulary.

    Args:
        vocab: Vocabulary.

    Returns:
        A tuple with the vocabulary size (length of the tokens),
        defined maximum length,
        start index (index of the BOS token),
        and sequence length append (1 if EOS token is present, else 0).
    """
    start_index = vocab.encode_map[vocab.bos]
    seq_len_append = 1 if vocab.eos is not None else 0
    return len(vocab), vocab.max_len, start_index, seq_len_append


__all__ = ["Vocabulary", "get_data_from_vocab"]
