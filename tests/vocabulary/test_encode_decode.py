"""Test the python-only parts of the vocabulary."""

# ruff: noqa: A001,A002
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest

from lstm_clm.vocab.decode import (
    build_decode_map,
    build_full_decode_func,
    decode_single,
)
from lstm_clm.vocab.encode import (
    _unwrap_split,
    build_full_encode_func,
    build_pad_func,
    build_tokenize_func,
    encode_multi,
    encode_single,
    pad_tokens,
    tokenize_string,
)

if TYPE_CHECKING:
    from lstm_clm.vocab.proto import Int32Array


def test_encode_single(sample_string: str, encode_map: dict[str, int]) -> None:
    expected_encoded = [encode_map[c] for c in sample_string]
    expected_unknown = [encode_map.get(c, encode_map["U"]) for c in sample_string + "x"]

    assert encode_single(list(sample_string), encode_map, "U") == expected_encoded
    assert encode_single(list(sample_string + "x"), encode_map, "U") == expected_unknown


def test_decode_single(
    sample_string: str, encode_map: dict[str, int], decode_map: dict[int, str]
) -> None:
    input = [encode_map.get(i, encode_map["U"]) for i in sample_string]
    assert decode_single(input, decode_map) == sample_string


@pytest.mark.parametrize(
    ("input", "expected"), [([0, 8], "h"), ([9, 0, 1, 2, 2, 3, 10], "hello")]
)
def test_adjusted_decode_map(
    adjusted_decode_map: dict[int, str], input: list[int], expected: str
) -> None:
    assert decode_single(input, adjusted_decode_map) == expected


@pytest.mark.parametrize(
    ("split_fixture", "input_str", "expected"),
    [
        ("split_on_every_char", "a10b", ["a", "1", "0", "b"]),
        ("split_on_numbers", "a10b", ["a", "10", "b"]),
    ],
)
def test_unwrap_split(
    request: pytest.FixtureRequest,
    split_fixture: str,
    input_str: str,
    expected: list[str],
) -> None:
    split: re.Pattern = request.getfixturevalue(split_fixture)
    assert _unwrap_split(split.split(input_str)) == expected


@pytest.mark.parametrize(
    ("right_pad", "assert_length", "expected"),
    [
        (True, False, ["h", "i", "A", "A", "A"]),
        (False, False, ["A", "A", "A", "h", "i"]),
        (True, True, ["h", "i", "A", "A", "A"]),
    ],
)
def test_pad_tokens(right_pad: bool, assert_length: bool, expected: list[str]) -> None:
    assert pad_tokens(list("hi"), "A", 5, right_pad, assert_length) == expected


def test_pad_tokens_raises() -> None:
    with pytest.raises(ValueError, match="Number of tokens too large"):
        pad_tokens(list("helloworld"), "A", 5, assert_length=True)


@pytest.mark.parametrize(
    ("input_str", "split_fixture", "bos", "eos", "expected"),
    [
        (
            "x12y1z",
            "split_on_every_char",
            "B",
            "E",
            ["B", "x", "1", "2", "y", "1", "z", "E"],
        ),
        ("x12y1z", "split_on_numbers", "B", "E", ["B", "x", "12", "y", "1", "z", "E"]),
        (
            "x12y1z",
            "split_on_every_char",
            "B",
            None,
            ["B", "x", "1", "2", "y", "1", "z"],
        ),
    ],
)
def test_tokenize_string(
    request: pytest.FixtureRequest,
    input_str: str,
    split_fixture: str,
    bos: str,
    eos: str | None,
    expected: list[str],
) -> None:
    split: re.Pattern = request.getfixturevalue(split_fixture)
    assert tokenize_string(input_str, split, bos, eos) == expected


def test_tokenize_string_raises() -> None:
    with pytest.raises(ValueError, match="Input contains whitespace"):
        tokenize_string("x12 1", re.compile(r"\s"), "B", None)


@pytest.mark.parametrize(
    ("bos", "eos", "expected"),
    [
        ("B", "E", ["B", "t", "e", "s", "t", "E"]),
        ("B", None, ["B", "t", "e", "s", "t"]),
    ],
)
def test_build_tokenize_func(
    split_on_every_char: re.Pattern, bos: str, eos: str | None, expected: list[str]
) -> None:
    tokenize = build_tokenize_func(bos, eos, split_on_every_char)
    assert tokenize("test") == expected


@pytest.mark.parametrize(
    ("max_len", "right_pad", "assert_length", "input_tokens", "expected"),
    [
        (5, True, False, ["H", "i"], ["H", "i", "A", "A", "A"]),
        (5, True, False, ["H", "i", "l", "l", "l", "o"], ["H", "i", "l", "l", "l"]),
        (5, False, False, ["H", "i"], ["A", "A", "A", "H", "i"]),
    ],
)
def test_build_pad_func(
    max_len: int,
    right_pad: bool,
    assert_length: bool,
    input_tokens: list[str],
    expected: list[str],
) -> None:
    pad = build_pad_func(max_len, "A", right_pad, assert_length)
    assert pad(input_tokens) == expected


def test_build_pad_func_raises() -> None:
    pad_assert = build_pad_func(5, "A", assert_length=True)
    with pytest.raises(ValueError, match="Number of tokens too large"):
        pad_assert(["H", "i", "l", "l", "l", "o"])


def test_build_decode_map(
    decode_map: dict[int, str],
    encode_map: dict[str, int],
    adjusted_decode_map: dict[int, str],
) -> None:
    assert build_decode_map(decode_map, encode_map, exclude=None) == decode_map
    assert (
        build_decode_map(decode_map, encode_map, exclude=["B", "E", "A", "U", *"UBEA"])
        == adjusted_decode_map
    )


@pytest.fixture
def encoded_decoded(encode_map: dict[str, int]) -> tuple[Int32Array, list[str]]:
    inputs = ["hello", "hi"]
    tokens = [["B", *x, "E"] for x in inputs]
    padded = [pad_tokens(x, "A", 10) for x in tokens]
    encoded = encode_multi(padded, encode_map, "U")
    return encoded, inputs


def test_build_full_encode_func(
    encoded_decoded: tuple[Int32Array, list[str]],
    encode_map: dict[str, int],
    split_on_every_char: re.Pattern,
) -> None:
    encoded, decoded = encoded_decoded
    encode = build_full_encode_func(
        encode_map, "A", "B", "E", "U", split_on_every_char, 8
    )
    np.testing.assert_equal(encode(decoded), encoded)

    inputs = [10 * "h"]
    expected = encode_single(["B", *9 * "h"], encode_map, "U")

    np.testing.assert_equal(encode(inputs), [expected])

    with_assert_length = build_full_encode_func(
        encode_map, "A", "B", "E", "U", split_on_every_char, 8, assert_length=True
    )

    with pytest.raises(
        ValueError, match=r"Max len of \d+ is greater than max_len_model of \d+"
    ):
        with_assert_length(inputs)


def test_build_full_decode_func(
    encoded_decoded: tuple[Int32Array, list[str]], adjusted_decode_map: dict[int, str]
) -> None:
    encoded, decoded = encoded_decoded

    decode = build_full_decode_func(adjusted_decode_map)
    assert decode(encoded) == decoded

    def custom_decode(inputs: list[list[int]], decode_map: dict[int, str]) -> list[str]:
        del decode_map  # unused
        return ["custom" for _ in inputs]

    encoded_aslist: list[list[int]] = encoded.tolist()
    decode_list = build_full_decode_func(adjusted_decode_map, custom_decode)
    assert decode_list(encoded_aslist) == ["custom", "custom"]
