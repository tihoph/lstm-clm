"""Perplexity calculation for a lstm_clm."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lstm_clm.clm.utils import batch_tensor_slices
from lstm_clm.clm.wrapper import tf_dtype, tf_kw_func_wrapper, tf_tensor_spec
from lstm_clm.vocab import VocabProto
from lstm_clm.vocab.tensor_impl import _create_hash_table

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    import tensorflow as tf

    from lstm_clm.clm.proto import MultinomialModel, NDArray, VocabMultinomialModel


def build_default_prior_probs(
    vocab: Sequence[str], special_tokens: Sequence[str]
) -> tf.lookup.StaticHashTable:
    """Build a static hash table with prior probabilities for the vocabulary.

    Assigns equal prior probabilities to all characters
    in the vocabulary except for special tokens.
    """
    char_indices = [ix for ix, char in enumerate(vocab) if char not in special_tokens]
    prior_probs = {k: 1 / len(char_indices) for k in char_indices}
    return build_hash_table(prior_probs)


def build_hash_table(prior_probs: dict[int, float]) -> tf.lookup.StaticHashTable:
    """Build a static hash table with prior probabilities for the vocabulary.

    Args:
        prior_probs: Prior probabilities.

    Returns:
        Static hash table with prior probabilities.
    """
    import tensorflow as tf

    return _create_hash_table(prior_probs, tf.int32, tf.float32, 0.0)


@tf_kw_func_wrapper(
    input_signature=[
        tf_tensor_spec(shape=[None, None], dtype=tf_dtype.int32),
        tf_tensor_spec(shape=[None, None, None], dtype=tf_dtype.float32),
        tf_tensor_spec(shape=[None, None], dtype=tf_dtype.float32),
    ]
)
def calc_perplexity(
    samples: tf.Tensor, preds: tf.Tensor, weights: tf.Tensor
) -> tf.Tensor:
    """Calculate perplexity from samples and predictions (wrapped in a tf.function).

    Args:
        samples: Samples shifted (y_true). [Batch, Length] {int32}
        preds: Predictions. [Batch, Length[1:], Vocab] {float32}
        weights: Prior probabilities. [Batch, Length[1:]] {float32}

    Returns:
        Calculated Perplexity scores. [Batch] {float32}
    """
    import tensorflow as tf

    vocab_size = tf.shape(preds)[-1]

    y_true = samples[:, 1:]  # rename for clarity
    # Select the probabilities of the true tokens (sampled tokens) # Batch x Seq x Vocab
    y_true = tf.one_hot(y_true, depth=vocab_size)
    forced_pred = y_true * preds
    # Add the probabilities per complete timestep # Batch x Seq
    forced_pred = tf.reduce_sum(forced_pred, axis=-1)
    # Convert to log space # Batch x Seq
    logits = tf.math.log(forced_pred)
    # Multiply by the weights (set special tokens to 0) # Batch x Seq
    weighted_logits = logits * weights
    # Count the number of non-zero weights per sequence # Batch
    n_nonzero = tf.math.count_nonzero(weighted_logits, axis=-1, dtype=tf.float32)
    # Sum the weighted logits per sequence # Batch
    logits_sum = tf.reduce_sum(weighted_logits, axis=-1)
    # Calculate the perplexity # Batch
    neg_logits_sum = tf.math.negative(logits_sum)
    return tf.math.exp(neg_logits_sum / n_nonzero)


def perplexity_from_samples(
    model: MultinomialModel | VocabMultinomialModel,
    batch_size: int,
    samples: tf.Tensor | NDArray,
    preds: tf.Tensor | NDArray | None = None,
    prior_probs: tf.lookup.StaticHashTable | None = None,
    verbose: int = 0,
) -> tf.Tensor:
    """Calculate perplexity of samples.

    Note: samples (max_len + 2), preds (max_len + 1).

    Args:
        model: Model to calculate perplexity from.
        batch_size: Batch size.
        samples: Previously generated samples. [Batch, Length] {int32}
        preds: Predictions. [Batch, Length[1:], Vocab] {float32}
        prior_probs: Vocabulary prior probabilities.
            Defaults to equal probabilities for non-special tokens.
        verbose: Verbosity level. Defaults to 0.

    Returns:
        Perplexity scores. [Batch] {float32}

    Raises:
        TypeError: If prior_probs is not provided
            and model is not a VocabMultinomialModel.
    """
    import tensorflow as tf

    if prior_probs is None:
        if not hasattr(model, "vocab") or not isinstance(model.vocab, VocabProto):
            raise TypeError(
                "prior_probs is required or provide a VocabMultinomialModel"
            )

        prior_probs = build_default_prior_probs(
            model.vocab.tokens, model.vocab.special_tokens
        )

    perplexity_scores: list[tf.Tensor] = []

    if preds is None:
        for samples_batch in batch_tensor_slices(
            samples, batch_size, "Calculating perplexity (incl. predict)", verbose
        ):
            preds_batch = model(samples_batch[:, :-1])
            weights = prior_probs.lookup(samples_batch[:, 1:])
            perp_batch = calc_perplexity(samples_batch, preds_batch, weights)
            perplexity_scores.append(perp_batch)

    else:
        for samples_batch, preds_batch in batch_tensor_slices(
            (samples, preds), batch_size, "Calculating perplexity", verbose
        ):
            weights = prior_probs.lookup(samples_batch[:, 1:])
            perp_batch = calc_perplexity(samples_batch, preds_batch, weights)
            perplexity_scores.append(perp_batch)

    return tf.concat(perplexity_scores, axis=0)


def generate_with_perplexity(
    model: MultinomialModel | VocabMultinomialModel,
    n_batches: int,
    temp: float = 1.0,
    batch_size: int = 1024,
    prior_probs: tf.lookup.StaticHashTable | None = None,
    verbose: int = 0,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Generate samples with perplexity calculation.

    Args:
        model: Model to generate samples from.
        n_batches: Number of batches to generate.
        temp: Temperature factor. Defaults to 1.0.
        batch_size: Batch size. Defaults to 1024.
        prior_probs: Vocabulary prior probabilities.
            Required if model is not VocabMultinomialCLM.
        verbose: Verbosity level. Defaults to 0.

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            samples: Generated samples. [Batch, Length] {int32}
            preds: Predictions. [Batch, Length, Vocab] {float32}
            perplexity: Perplexity scores. [Batch] {float32}
    """
    import tensorflow as tf

    samples, preds = model.generate(n_batches, temp, batch_size, verbose=verbose)

    # if output is one-hot encoded, convert to indices
    perplexity_samples = (
        samples
        if samples.ndim == 2  # noqa: PLR2004
        else tf.argmax(samples, axis=-1, output_type=tf.int32)
    )
    perplexity = perplexity_from_samples(
        model, batch_size, perplexity_samples, preds, prior_probs, verbose
    )
    return samples, preds, perplexity


__all__ = [
    "build_default_prior_probs",
    "build_hash_table",
    "calc_perplexity",
    "generate_with_perplexity",
    "perplexity_from_samples",
]
