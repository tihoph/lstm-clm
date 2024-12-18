"""Test the extended parts of the vocabulary.

This included the TensorFlow dependent parts.
"""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tensorflow as tf

from lstm_clm.vocab.decode import decode_multi
from lstm_clm.vocab.encode import encode_multi, pad_tokens
from lstm_clm.vocab.tensor_impl import (
    Vocabulary,
    _create_hash_table,
    _decode_multi,
    _decode_tensor,
    _encode_multi,
    _encode_tensor,
    _get_unused_char,
    _tensor_lookup,
    get_data_from_vocab,
)

if TYPE_CHECKING:
    from lstm_clm.vocab.proto import Int32Array


@pytest.mark.parametrize(
    ("inputs", "expected"), [({0: "C", 1: "B"}, "A"), ({0: "A", 1: "B", 2: "C"}, "D")]
)
def test_get_unused_char(inputs: dict[int, str], expected: str) -> None:
    assert _get_unused_char(inputs) == expected


@pytest.fixture
def chars_tokens(
    encode_map: dict[str, int],
) -> tuple[list[str], list[list[str]], Int32Array]:
    chars = ["hello", "hi"]
    inputs = [pad_tokens(["B", *x, "E"], "P", 8) for x in chars]
    return chars, inputs, encode_multi(inputs, encode_map, "U")


def test_create_hash_table(encode_map: dict[str, int]) -> None:
    hash_table = _create_hash_table(encode_map, tf.string, tf.int32, -1)
    assert hash_table.size() == len(encode_map)
    tensor = tf.convert_to_tensor(["h", "X"])
    np.testing.assert_equal(hash_table.lookup(tensor).numpy(), [0, -1])


def test_tensor_lookup(
    encode_map: dict[str, int], adjusted_decode_map: dict[int, str]
) -> None:
    hash_table = _create_hash_table(encode_map, tf.string, tf.int32, -1)
    data = [[*encode_map]]
    tensor = tf.convert_to_tensor(data)
    expected = tf.convert_to_tensor([[*encode_map.values()]])

    assert tensor.shape == (1, len(encode_map))
    np.testing.assert_equal(_tensor_lookup(tensor, hash_table), expected)

    hash_table = _create_hash_table(adjusted_decode_map, tf.int32, tf.string, "X")
    tensor = tf.convert_to_tensor([[0, 1], [1, 0]])
    np.testing.assert_equal(_tensor_lookup(tensor, hash_table), ["he", "eh"])

    tensor = tf.convert_to_tensor([[0, 22]])
    with pytest.raises(KeyError, match="Unknown character in input"):
        _tensor_lookup(tensor, hash_table)


def test_encode_tensor(
    encode_map: dict[str, int],
    chars_tokens: tuple[list[str], list[list[str]], list[list[int]]],
) -> None:
    _, inputs, tokens = chars_tokens
    tensor = tf.convert_to_tensor(inputs)
    np.testing.assert_equal(_encode_tensor(tensor, encode_map, "U"), tokens)


def test_decode_tensor(
    adjusted_decode_map: dict[int, str],
    chars_tokens: tuple[list[str], list[list[str]], list[list[int]]],
) -> None:
    chars, _, tokens = chars_tokens
    tensor = tf.convert_to_tensor(tokens)
    np.testing.assert_equal(
        _decode_tensor(tensor, adjusted_decode_map), ["".join(x) for x in chars]
    )


def test_decode_multi(
    adjusted_decode_map: dict[int, str],
    chars_tokens: tuple[list[str], list[list[str]], list[list[int]]],
) -> None:
    chars, _, tokens = chars_tokens
    assert decode_multi(tokens, adjusted_decode_map) == chars
    assert _decode_multi(tokens, adjusted_decode_map) == chars
    assert _decode_multi(tokens, adjusted_decode_map) == chars
    tensor = tf.convert_to_tensor(tokens)
    assert _decode_multi(tensor, adjusted_decode_map) == chars


def test_encode_multi(
    encode_map: dict[str, int],
    chars_tokens: tuple[list[str], list[list[str]], list[list[int]]],
) -> None:
    ragged_tokens = [["h"], ["h", "i"]]
    with pytest.raises(ValueError, match="setting an array element with a sequence"):
        encode_multi(ragged_tokens, encode_map, "U")

    padded_tokens = [pad_tokens(x, "A", 8) for x in ragged_tokens]
    expected = [[encode_map.get(t, encode_map["U"]) for t in x] for x in padded_tokens]
    encoded = encode_multi(padded_tokens, encode_map, "U")
    np.testing.assert_equal(encoded, expected)
    encoded = _encode_multi(padded_tokens, encode_map, "U")
    np.testing.assert_equal(encoded, expected)

    _, inputs, tokens = chars_tokens
    tensor = tf.convert_to_tensor(inputs)
    np.testing.assert_equal(_encode_multi(tensor, encode_map, "U"), tokens)


@pytest.fixture
def vocab() -> list[str]:
    return ["A", "B", "E", "U", "C", "c", "Cl", "O", "S", "1", "2", "[", "]", "[Cl]"]


def test_vocabulary(vocab: list[str], caplog: pytest.LogCaptureFixture) -> None:
    voc = Vocabulary(vocab, 4, "A", "B", "E", "U")

    assert len(voc) == len(vocab)

    expected = [[1, 4, 2, 0, 0, 0]]
    np.testing.assert_equal(voc.encode(["C"]), expected)

    with caplog.at_level(logging.WARNING):
        np.testing.assert_equal(Vocabulary.encode(voc, ["C"]), expected)
        assert "Called as method, thus not pickable" in caplog.text

    with caplog.at_level(logging.WARNING):
        assert Vocabulary.decode(voc, expected) == ["C"]
        assert "Called as method, thus not pickable" in caplog.text

    with pytest.raises(ValueError, match="Tokens must be unique"):
        Vocabulary([*vocab, "A"], 4, "A", "B", "E", "U")

    with pytest.raises(ValueError, match="Special tokens must be in the set of tokens"):
        Vocabulary(vocab, 4, "X", "B", "E", "U")


def test_end_token(vocab: list[str]) -> None:
    # no end token
    no_end = Vocabulary(vocab, 4, "A", "B", unk="U")
    expected = [[1, 4, 0, 0, 0]]
    np.testing.assert_equal(no_end.encode(["C"]), expected)

    with_unknown = [[1, 3, 0, 0, 0]]
    np.testing.assert_equal(no_end.encode(["X"]), with_unknown)


def test_unk_replacement(vocab: list[str]) -> None:
    no_unk = Vocabulary(vocab, 4, "A", "B", assert_known=False)  # it is replaced by PAD
    expected = [[1, 4, 0, 0, 0]]
    np.testing.assert_equal(no_unk.encode(["CK"]), expected)
    no_unk_safe = Vocabulary(vocab, 4, "A", "B", assert_known=True)

    with pytest.raises(KeyError, match="Unknown token"):
        no_unk_safe.encode(["CK"])


def test_assert_length(vocab: list[str]) -> None:
    # assert length
    assert_length = Vocabulary(vocab, 4, "A", "B", assert_length=True)
    with pytest.raises(
        ValueError, match=r"Max len of \d+ is greater than max_len_model of \d+"
    ):
        assert_length.encode(["C1CCCCC1"])

    no_assert_length = Vocabulary(vocab, 4, "A", "B", assert_length=False)
    np.testing.assert_equal(no_assert_length.encode(["C1CCCCC1"]), [[1, 4, 9, 4, 4]])


def test_only_two_char_tokens() -> None:
    vocab = ["A", "B", "C", "Cl", "[", "]", "[Cl]"]
    voc = Vocabulary(vocab, 4, "A", "B", only_two_char_tokens=True)
    np.testing.assert_equal(voc.encode(["C[Cl]"]), [[1, 2, 4, 3, 5]])  # C [ Cl ]
    voc = Vocabulary(vocab, 4, "A", "B", only_two_char_tokens=False)
    np.testing.assert_equal(voc.encode(["C[Cl]"]), [[1, 2, 6, 0, 0]])  # C [Cl] PAD PAD


params = itertools.product(["ABCDE", "EBACD"], [None, "E"], [2, 4, 5])


@pytest.mark.parametrize(("vocab_str", "eos", "max_len"), params)
def test_get_data_from_vocab(vocab_str: str, eos: str | None, max_len: int) -> None:
    vocab = Vocabulary(list(vocab_str), max_len, "A", "B", eos=eos)
    data = get_data_from_vocab(vocab)
    expected = (
        len(vocab_str),
        max_len,
        vocab_str.index("B"),
        1 if eos is not None else 0,
    )
    assert data == expected
