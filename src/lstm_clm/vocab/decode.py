"""Decode functions for the vocabulary."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, overload

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Sequence

    from lstm_clm.vocab.proto import Int32Array

T_contra = TypeVar("T_contra", contravariant=True)


def decode_single(indices: Sequence[int], decode_map: dict[int, str]) -> str:
    """Decode a single integer sequence to a string using a vocabulary.

    Args:
        indices: List of indices to decode. [Length] {int32}
        decode_map: Mapping from indices to tokens.

    Returns:
        Joined string of decoded tokens. [1] {str}
    """
    return "".join(decode_map[ix] for ix in indices)


def build_decode_map(
    decode_map: dict[int, str],
    encode_map: dict[str, int],
    exclude: Sequence[str] | None = None,
) -> dict[int, str]:
    """Adjusts the decode map if characters should be excluded.

    Args:
        decode_map: Mapping from indices to tokens.
        encode_map: Mapping from tokens to indices.
        exclude: List of values which are set to "" if found. Defaults to None.

    Returns:
        Adjusted decode map.
    """
    if exclude is None:
        return decode_map

    decode_map = decode_map.copy()  # Avoid modifying the original
    for token in exclude:
        ix = encode_map[token]
        decode_map[ix] = ""
    return decode_map


def decode_multi(
    inputs: Sequence[Sequence[int]] | Int32Array, decode_map: dict[int, str]
) -> list[str]:
    """Decode a nested integer sequence to a list of strings using a vocabulary.

    Args:
        inputs: List of integers to decode. [Batch, Length] {int32}
        decode_map: Mapping from tokens to indices
            with empty strings for excluded tokens.
            May use `build_decode_map` to exclude special tokens.

    Returns:
        Decoded list of integers as strings. [Batch] {str}

    Raises:
        ValueError: If unknown character is in input.
    """
    return [decode_single(x, decode_map) for x in inputs]


class DecodeFunc(Protocol, Generic[T_contra]):
    """Decode function protocol."""

    def __call__(self, inputs: T_contra, decode_map: dict[int, str]) -> list[str]:
        """Decode inputs to a list of strings.

        Inputs can be either a tensor/np.ndarray[np.str_]/list[list[int]].
        """


@overload
def build_full_decode_func(
    adjusted_decode_map: dict[int, str], decode_func: None = None
) -> Callable[[Sequence[Sequence[int]] | Int32Array], list[str]]: ...


@overload
def build_full_decode_func(
    adjusted_decode_map: dict[int, str], decode_func: DecodeFunc[T_contra]
) -> Callable[[T_contra], list[str]]: ...


def build_full_decode_func(
    adjusted_decode_map: dict[int, str], decode_func: DecodeFunc[T_contra] | None = None
) -> (
    Callable[[T_contra], list[str]]
    | Callable[[Sequence[Sequence[int]] | Int32Array], list[str]]
):
    """Build a decode function.

    Args:
        adjusted_decode_map: Decode map with empty strings for excluded tokens.
        decode_func: Decode function.
            Takes a decodable input and a decode map. Returns a list of strings.
            Defaults to `decode_multi`.
    """
    if decode_func is None:
        return partial(decode_multi, decode_map=adjusted_decode_map)
    return partial(decode_func, decode_map=adjusted_decode_map)
