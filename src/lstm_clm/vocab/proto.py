"""Protocol defining a vocabulary.

Additionally, the types used in the package are defined here.

Implementation in `lstm_clm.vocab.tensor_impl`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    import tensorflow as tf


Float32Array: TypeAlias = NDArray[np.float32]
Int32Array: TypeAlias = NDArray[np.int32]
StrArray: TypeAlias = NDArray[np.str_]

# union as str to avoid tensorflow import
EncodableBatch: TypeAlias = "StrArray | Sequence[str]"
"""Types, which can be encoded to indices. [Batch] {str}"""

TokenizedBatch: TypeAlias = "StrArray | Sequence[Sequence[str]]"
"""Tokenized inputs for encoding. [Batch, Length] {str}"""

DecodableBatch: TypeAlias = "tf.Tensor | Int32Array | Sequence[Sequence[int]]"
"""Encoded inputs for decoding. [Batch, Length] {int32}"""


@runtime_checkable
class VocabProto(Protocol):
    """Protocol for a vocabulary.

    Must implement the following attributes and methods:

    Attributes:
        tokens (list[str]): List of unique tokens in the vocabulary.
        max_len (int): Maximum length of generated samples.
        pad (str): Padding token.
        bos (str): Beginning of sentence token.
        eos (str | None): End of sentence token. Optional.
        unk (str | None): Unknown token. Optional.
        encode_map (dict[str, int]): Token to index mapping.
        decode_map (dict[int, str]): Index to token mapping.
        special_tokens (list[str]): List of special tokens.

    Methods:
        - __len__: Returns number of vocab tokens.
        - encode: Encode a list of tokens to indices.
        - decode: Decode a list of indices to tokens.
    """

    # Arguments:
    # EncodableBatch: must be Sequence, NDArray or tf.Tensor. [Batch] {str}
    # DecodableBatch: must be Sequence[Sequence], NDArray or tf.Tensor.
    #     [Batch, Length] {int32}

    tokens: list[str]
    """@private List of unique tokens in the vocabulary."""
    max_len: int
    """@private Maximum length of generated samples."""
    pad: str
    """@private Padding token."""
    bos: str
    """@private Beginning of sentence token."""
    eos: str | None
    """@private End of sentence token. Optional."""
    unk: str | None
    """@private Unknown token. Optional."""
    encode_map: dict[str, int]
    """@private Token to index mapping."""
    decode_map: dict[int, str]
    """@private Token to index mapping."""
    special_tokens: list[str]
    """@private List of special tokens."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """@private Do not initizalize protocols."""
        super().__init__(*args, **kwargs)

    def __len__(self) -> int:
        """@public Returns number of vocab tokens."""

    def encode(self, inputs: EncodableBatch) -> Int32Array:
        """Encode a list of tokens to indices.

        Args:
            inputs: Token inputs. [Batch] {str}

        Returns:
            Encoded indices. [Batch, Length] {int32}
        """

    def decode(self, inputs: DecodableBatch) -> list[str]:
        """Decode a list of indices to tokens.

        Args:
            inputs: Indices inputs. [Batch, Length] {int32}

        Returns:
            Decoded tokens. [Batch] {str}
        """


__all__ = ["VocabProto"]
