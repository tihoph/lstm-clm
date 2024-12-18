"""Vocabulary for the lstm_clm.

Usage:
    >>> # Import the vocabulary functions (extended triggers tensorflow import)
    >>> from lstm_clm.vocab import Vocabulary
    >>> # Create a vocabulary instance
    >>> tokens = ["a", "b", "c", "d", "B", "E", "P"]
    >>> vocab = Vocabulary(tokens, 5, pad="P", "B", "E")
    >>> # Encode to np.ndarray
    >>> encoded = vocab.encode(["abc", "bcd"])
    >>> print(encoded.shape) # max len + 2 (for start and end token)
    (2, 7)
    >>> print(encoded)
    [[ 4 0 1 2 5 6 6
       4 1 2 3 5 6 6]]
    >>> # Decode to list of strings
    >>> decoded = vocab.decode(encoded)
    >>> print(decoded)
    ["abc", "bcd"]
"""

from __future__ import annotations

from lstm_clm.vocab import decode, encode
from lstm_clm.vocab.proto import VocabProto
from lstm_clm.vocab.tensor_impl import Vocabulary, get_data_from_vocab

__all__ = ["VocabProto", "Vocabulary", "decode", "encode", "get_data_from_vocab"]
