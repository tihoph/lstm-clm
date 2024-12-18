"""Encode functions for the vocabulary."""

from __future__ import annotations

import re
from functools import partial
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable, Sequence

    from lstm_clm.vocab.proto import Int32Array, StrArray


WHITESPACE = re.compile(r"\s")
"""Regex pattern for whitespace"""

REGEX_PATTERNS: dict[bool | None, SplitProto] = {
    True: re.compile(r"(Br|Cl|se|Se|Si|si|@@)"),
    False: re.compile(r"(\[[^]]*\]|%\\d{2}|Br|Cl|@@)"),
    None: WHITESPACE,
}
"""Regex patterns for tokenization

- True: Keep Br, Cl, se, Se, Si, si, @@ together
- False: Keep Br, Cl, @@, %{number} and everything in [] together
- None: Split character-wise
"""


class SplitProto(Protocol):
    """Protocol for splitting a string for tokenization.

    Main implementation is :class:`re.Pattern`.
    Custom implementation needs a `split` method
    which seperates tokens that should
    be kept together from parts that
    should be split character-wise.
    The tokens should be in every second
    position.

    Example:
        >>> splitter = re.compile("(Br|Cl|se|Se|Si|si|@@)")
        >>> splitter.split("BrCCCCl")
        ["Br", "CCC", "Cl"]
        >>> splitter.split("CCCClCCC")
        ["", "CCC, "Cl", "CCC"]

    """

    def split(self, x: str) -> Iterable[str]:
        """Split a string into tokens and characters.

        Args:
            x: String to split

        Returns:
            Iterable of tokens and characters
        """


def pad_tokens(
    tokens: Sequence[str],
    pad: str,
    max_len_model: int,
    right_pad: bool = True,
    assert_length: bool = False,
) -> list[str]:
    """Pads a tokenized string to a given length.

    Pads either before or after the input tokens.
    Cuts off the input tokens if they are longer than max_len.

    Args:
        tokens: List of tokens. [Any] {str}
        pad: Padding character.
        max_len_model: Maximum length of the model.
            Caveat: add 1 for each of start and/or end token.
        right_pad: If True, pads to the right, else pads to the left.
            Defaults to True.
        assert_length: If true, raises error on too long input.
            Else, cuts off the input. Defaults to False.

    Returns:
        Padded list of tokenized strings [Batch, Length] {str}

    Raises:
        ValueError: If `assert_length` is true and number
            of tokens is larger than `max_len_model`
    """
    if assert_length and len(tokens) > max_len_model:
        raise ValueError("Number of tokens too large")

    tokens = list(tokens[:max_len_model])
    padding_tokens = [pad] * (max_len_model - len(tokens))
    if right_pad:
        return tokens + padding_tokens
    return padding_tokens + tokens


def _unwrap_split(iterable: Iterable[str]) -> list[str]:
    """Unwrap a split seperated by a :class:`SplitProto` object.

    Every second element is a token,
    which should be kept together.
    Every other element is a character-wise split.

    Example:
        >>> inputs = ['', 'TX', 'HMZ', 'TX', 'SW']
        >>> _unwrap_split(inputs)
        ... ['TX', 'H', 'M', 'Z', 'TX', 'S', 'W']

    Args:
        iterable: Iterable to unwrap

    Returns:
        Unwrapped list of strings
    """
    unwrapped: list[str] = []
    for i, x in enumerate(iterable):
        if i % 2 == 0:
            unwrapped.extend(x)
        else:
            unwrapped.append(x)
    return unwrapped


def tokenize_string(
    inputs: str, pattern: SplitProto, bos: str, eos: str | None
) -> list[str]:
    r"""Split a string into tokens. Encloses them with BOS and EOS tokens, if present.

    Example:
        >>> pattern = re.compile(r"(\d+)")
        >>> bos, eos = "BOS", "EOS"
        >>> tokenize_string("t33st", pattern, bos, eos)
        ... ["BOS", "t", "33", "s", "t", "EOS"]

    Args:
        inputs: String to split. [1] {str}
        pattern: Pattern to split by (e.g. regex pattern)
            Every second element is a token.
            Every other element is character-wise split.
        bos: Start token
        eos: End token. Optional.

    Returns:
        List of tokens. [Any] {str}

    Raises:
        ValueError: If input contains whitespace
    """
    if WHITESPACE.search(inputs):
        raise ValueError("Input contains whitespace")

    splitted = pattern.split(inputs)
    tokens: list[str] = [bos, *_unwrap_split(splitted)]

    if eos is not None:
        tokens.append(eos)

    return tokens


def encode_single(
    tokens: Sequence[str], encode_map: dict[str, int], unk: str | None = None
) -> list[int]:
    """Encode a single padded token sequence to integers using a vocabulary.

    Args:
        tokens: List of padded tokens to encode. [Length] {str}
        encode_map: Mapping from tokens to indices.
        unk: Unknown character. If None, raises error on unknown characters.
            Defaults to None.

    Returns:
        Encoded list of tokens as list of ints. [Length] {int32}
    """
    if unk is None:
        try:
            return [encode_map[x] for x in tokens]
        except KeyError as e:
            raise KeyError(f"Unknown token: {e}") from e

    unk_ix = encode_map[unk]
    return [encode_map.get(x, unk_ix) for x in tokens]


def encode_multi(
    inputs: Sequence[Sequence[str]] | StrArray,
    encode_map: dict[str, int],
    unk: str | None = None,
) -> Int32Array:
    """Encode a list of padded tokens to integers using a vocabulary.

    Args:
        inputs: Tokenized and padded inputs for encoding. [Batch, Length] {str}
        encode_map: Mapping from tokens to indices
        unk: Unknown character. Defaults to None.

    Returns:
        Encoded list of strings as int array [Batch, Length] {int32}
    """
    return np.array([encode_single(x, encode_map, unk) for x in inputs], dtype=np.int32)


def build_tokenize_func(
    bos: str, eos: str | None, pattern: SplitProto
) -> Callable[[str], list[str]]:
    """Build a tokenization function.

    Args:
        bos: Beginning of sentence token.
        eos: End of sentence token. If None, no end token is added.
        pattern: Pattern to split by (e.g. regex pattern)

    Returns:
        Tokenization function.
    """
    return partial(tokenize_string, pattern=pattern, bos=bos, eos=eos)


def build_pad_func(
    max_len_model: int, pad: str, right_pad: bool = True, assert_length: bool = False
) -> Callable[[list[str]], list[str]]:
    """Build a padding function.

    Args:
        max_len_model: Maximum length of the model
        pad: Padding character
        right_pad: If True, pads to the right, else pads to the left.
        assert_length: If true, raises error instead of cutting. Defaults to False.

    Returns:
        Padding function.
    """
    # Set tokens to max_len_model # TODO: Might be causing problems
    return partial(
        pad_tokens,
        pad=pad,
        max_len_model=max_len_model,
        right_pad=right_pad,
        assert_length=assert_length,
    )


def build_encode_func(
    encode_map: dict[str, int], unk: str | None
) -> Callable[[Sequence[Sequence[str] | StrArray]], Int32Array]:
    """Build an encode function.

    Args:
        encode_map: Token to index mapping.
        unk: Unknown token.
    """
    return partial(encode_multi, encode_map=encode_map, unk=unk)


def encode_data(
    inputs: Sequence[str] | StrArray,
    tokenize_func: Callable[[str], list[str]],
    pad_func: Callable[[list[str]], list[str]],
    encode_func: Callable[[Sequence[Sequence[str] | StrArray]], Int32Array],
    max_len_model: int | None = None,
) -> Int32Array:
    """Encodes strings to integers using a defined tokenization, padding and encoding.

    Args:
        inputs: List of strings. [Batch] {str}
        tokenize_func: Tokenization function.
            Takes a single string and returns a list of tokens.
            Adjustments, like BOS and EOS tokens, should be done here.
            May use `build_tokenize_func` to build.
        pad_func: Padding function.
            Takes a list of tokens and returns a padded list of tokens.
            Pad direction and length must be defined in the function.
            May use `build_pad_func` to build.
        encode_func: Encoding function.
            Takes a list of padded tokens and returns an encoded list of tokens.
            Mapping from
            May use `build_encode_func` to build.
        max_len_model: Maximum length of the model.
            Has to include BOS and EOS tokens if present.
            If set and any input is longer, an error is raised.
            Else, the input is cut off.

    Returns:
        Encoded SMILES strings

    Raises:
        ValueError: If `max_len_model` is set
            and any input is longer than `max_len_model`.
    """
    tokens = [tokenize_func(x) for x in inputs]

    if (
        max_len_model is not None
        and (real_max := max(len(x) for x in tokens)) > max_len_model
    ):
        raise ValueError(
            f"Max len of {real_max} is greater than max_len_model of {max_len_model}"
        )

    padded_tokens = [pad_func(single_tokens) for single_tokens in tokens]

    return encode_func(padded_tokens)


def build_full_encode_func(
    encode_map: dict[str, int],
    pad: str,
    bos: str,
    eos: str | None,
    unk: str | None,
    pattern: SplitProto,
    max_len: int,
    right_pad: bool = True,
    assert_length: bool = False,
) -> Callable[[Sequence[str]], Int32Array]:
    """Build an encode function.

    Args:
        encode_map: Token to index mapping.
        pad: Padding character.
        bos: Beginning of sentence token.
        eos: End of sentence token. If None, no end token is added.
        unk: Unknown token.
        max_len: Maximum length of the model.
        pattern: Pattern to split by (e.g. regex pattern)
        right_pad: If True, pads to the right, else pads to the left.
        assert_length: If true, raises error instead of cutting. Defaults to False.
    """
    tokenize_func = build_tokenize_func(bos, eos, pattern)

    max_len_model = max_len + (2 if eos is not None else 1)

    pad_func = build_pad_func(max_len_model, pad, right_pad)
    encode_func = build_encode_func(encode_map, unk)

    return partial(
        encode_data,
        tokenize_func=tokenize_func,
        pad_func=pad_func,
        encode_func=encode_func,
        max_len_model=max_len_model if assert_length else None,
    )
