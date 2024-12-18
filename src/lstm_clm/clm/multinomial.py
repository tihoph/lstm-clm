# %%
"""Module for models generating samples with multinomial sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lstm_clm.clm.base import BaseCLM
from lstm_clm.clm.perplexity import generate_with_perplexity
from lstm_clm.clm.utils import init_hidden_states, init_tensor_array, multinomial_sample
from lstm_clm.clm.wrapper import (
    tf_dtype,
    tf_func_wrapper,
    tf_kw_func_wrapper,
    tf_tensor_spec,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    import tensorflow as tf

    from lstm_clm.clm.proto import CallFunc, SampleFunc
    from lstm_clm.vocab import VocabProto


class MultinomialCLM(BaseCLM):
    """Base class for models generating samples with multinomial sampling.

    Attributes:
        model (tf.keras.Model): Model to wrap with LSTM layers.
        vocab (Vocabulary | None): Vocabulary.
        seq_len (int): Sequence length.
        dims (list[int]): Number of neurons per LSTM layers.
        has_embedding (bool): Whether the model has an Embedding layer.
        start_token (tf.Tensor): Start token for generation.
    """

    @tf_func_wrapper
    def generate(
        self,
        n_batches: int = 1,
        temp: float = 1.0,
        batch_size: int = 1024,
        call_func: CallFunc | None = None,
        sample_func: SampleFunc | None = None,
        verbose: int = 0,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Generate multiple batches of samples (wrapped in a tf.function).

        Args:
            n_batches: Number of batches to generate.
            temp: Temperature factor. Defaults to 1.0.
            batch_size: Batch size for generation. Defaults to 1024.
            call_func: Function to call model for one time step.
                Defaults to instance's `call_cell` method.
            sample_func: Function to sample next tokens.
                Defaults to instance's `sample_next_tokens` method.
            verbose: Verbosity level. Unused. Defaults to 0.

        Returns:
            Tuple with generated samples [Batches*Batch, Length] {int32}
            and predictions [Batches*Batch, Length, Vocab] {float32}
        """
        del verbose  # Unused because no progress bar
        import tensorflow as tf
        # from tfpb import tfpb # noqa: ERA001

        safe_call_func: CallFunc = call_func or self.call_cell
        safe_sample_func: SampleFunc = sample_func or self.sample_next_tokens

        if n_batches == 1:
            samples, preds = self._generate_batch(
                temp, batch_size, safe_call_func, safe_sample_func
            )

        dtype = tf.int32 if self.has_embedding else tf.float32
        samples = init_tensor_array(dtype, n_batches)
        preds = init_tensor_array(tf.float32, n_batches)

        # Define recurrence function to generate batches

        # pb = tfpb(desc="Generating samples", total=n_batches, disable=verbose < 1) # noqa: ERA001,E501
        start = tf.timestamp()

        def _g_recurrence(
            ix: tf.Tensor,
            last: tf.Tensor,
            samples: tf.TensorArray,
            preds: tf.TensorArray,
        ) -> tuple[tf.Tensor, tf.Tensor, tf.TensorArray, tf.TensorArray]:
            # Generate batch of samples
            # if embedding: [Batch, Length] {int32}
            # else: [Batch, Length, Vocab] {float32} # noqa: ERA001

            # If return_preds is True, unpack samples and predictions and
            # append predictions to TensorArray
            # if return_preds: type of preds is tf.TensorArray and
            # type _samples is tuple[tf.Tensor, tf.Tensor]
            samples_batch, preds_batch = self._generate_batch(
                temp, batch_size, safe_call_func, safe_sample_func
            )
            preds = preds.write(ix, preds_batch)

            # Append samples to TensorArray
            samples = samples.write(ix, samples_batch)

            new_ix = ix + 1
            # new_ix, last = pb.update(ix, last, start) # noqa: ERA001

            return new_ix, last, samples, preds

        # Generate batches using recurrence function

        loop_output: tuple[tf.Tensor, tf.Tensor, tf.TensorArray, tf.TensorArray] = (
            tf.while_loop(
                cond=lambda i, *_: i < n_batches,
                body=_g_recurrence,
                loop_vars=(0, start, samples, preds),
            )
        )

        _, _, samples, preds = loop_output

        # Stack generated batches into final output
        # [Batches*Batch, Length] {int32} if embedding else {float32}
        joined_samples = samples.concat()

        # stack predictions into final output
        # [Batches*Batch, Length, Vocab] {float32}
        # and return samples and predictions

        joined_preds = preds.concat()
        return joined_samples, joined_preds

    @tf_kw_func_wrapper(
        input_signature=[
            tf_tensor_spec(shape=(), dtype=tf_dtype.int32),
            tf_tensor_spec(shape=[None, None], dtype=tf_dtype.float32),
            tf_tensor_spec(shape=(), dtype=tf_dtype.float64),
        ]
    )
    def sample_next_tokens(
        self, i: tf.Tensor, x_t: tf.Tensor, /, temp: float
    ) -> tf.Tensor:
        """Multinomially sample next tokens from the output (wrapped in a tf.function).

        Args:
            i: Current time step. [1] {int32}.
            x_t: Model output [Batch, Vocab] {float32}.
            temp: Temperature factor. [1]

        Returns:
            Sampled next tokens [Batch] {int32}.
        """
        del i  # Multinomial sampling does not depend on the current time step
        return multinomial_sample(x_t, temp=temp)

    @tf_func_wrapper
    def _generate_batch(
        self, temp: float, batch_size: int, call_func: CallFunc, sample_func: SampleFunc
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Generate one batch of samples (wrapped in a tf.function).

        Args:
            temp: Temperature factor
            batch_size: Batch size
            call_func: Function to call model for one time step.
            sample_func: Function to sample next tokens.

        Returns:
            Tuple with generated samples
            if embedding: [Batch, Length] {int32}
            else: [Batch, Length, Vocab] {float32}
            and predictions [Batch, Length, Vocab] {float32}
        """
        import tensorflow as tf

        # Initialize start tokens for generation,
        # if embedding: [Batch, 1] {int32}
        # else: [Batch, 1, Vocab] {float32} # noqa: ERA001
        start_tokens = tf.repeat(self.start_token, repeats=batch_size, axis=0)

        # Initialize hidden states for LSTM layers (GRU not supported)
        hidden_states = init_hidden_states(self.dims, batch_size)

        dtype = tf.int32 if self.has_embedding else tf.float32
        gen_x = init_tensor_array(dtype, self.seq_len)
        preds = init_tensor_array(tf.float32, self.seq_len)

        # Define recurrence function to generate samples
        def _g_recurrence(
            i: tf.Tensor,
            gen_x: tf.TensorArray,
            next_tokens: tf.Tensor,
            hidden_states: Sequence[Sequence[tf.Tensor]],
            preds: tf.TensorArray,
        ) -> tuple[
            tf.Tensor, tf.TensorArray, tf.Tensor, list[list[tf.Tensor]], tf.TensorArray
        ]:
            # Call model for one time step
            x_t, new_hidden_states, _ = call_func(next_tokens, hidden_states)

            # Sample next token (from logits or softmax distribution)

            next_tokens = sample_func(i, x_t, temp=temp)

            # Cast selected token to float if using one-hot encoding
            if not self.has_embedding:
                next_tokens = tf.one_hot(
                    next_tokens, depth=self.model.output_shape[-1], dtype=tf.float32
                )

            # Append next token to TensorArray
            # Shape: [Length[:i], Batch] # noqa: ERA001
            gen_x = gen_x.write(i, next_tokens)

            # Return updated variables
            return (i + 1, gen_x, next_tokens, new_hidden_states, preds.write(i, x_t))

        # Generate samples using recurrence function
        zero = tf.constant(0, dtype=tf.int32)

        loop_output: tuple[
            tf.Tensor, tf.TensorArray, tf.Tensor, list[list[tf.Tensor]], tf.TensorArray
        ] = tf.while_loop(
            cond=lambda i, *_: i < self.seq_len,
            body=_g_recurrence,
            loop_vars=(
                zero,  # Shape: []
                gen_x,  # Shape: [Length[:0], Batch]
                start_tokens,  # Shape: [Batch]
                hidden_states,  # Shape: [[Batch, Neurons], ...]
                preds,  # Shape: [Length[:0], Batch, Vocab]
            ),
        )
        _, gen_x, _, _, preds = loop_output

        # Stack generated samples into final output
        # Shape: [Length, Batch] # noqa: ERA001
        gen_x = gen_x.stack()  # TODO: Write an issue
        gen_x = tf.concat(  # pylint: disable=redefined-variable-type
            [tf.expand_dims(start_tokens, axis=0), gen_x], axis=0
        )  # Shape: [Length+1, Batch]

        permutation = [1, 0] if self.has_embedding else [1, 0, 2]

        outputs = tf.transpose(gen_x, perm=permutation)  # Shape: [Batch, Length+1]

        preds = (
            preds.stack()  # TODO: Write an issue
        )  # Shape: [Length, Batch, Vocab]
        preds = tf.transpose(preds, perm=[1, 0, 2])  # Shape: [Batch, Length, Vocab]
        return outputs, preds

    def generate_with_perplexity(
        self,
        n_batches: int = 1,
        temp: float = 1.0,
        batch_size: int = 1024,
        prior_probs: tf.lookup.StaticHashTable | None = None,
        verbose: int = 0,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Generate samples with perplexity calculation from a model.

        Args:
            n_batches: Number of batches to generate. Defaults to 1.
            temp: Temperature factor. Defaults to 1.0.
            batch_size: Batch size for generation. Defaults to 1024.
            prior_probs: Prior probabilities for each token.
                Defaults to equal probabilities for each non-special token.
            verbose: Verbosity level. Defaults to 0.

        Returns:
            Tuple with generated samples [Batches*Batch, Length] {int32} and
            predictions [Batches*Batch, Length, Vocab] {float32} and
            perplexity scores [Batches*Batch] {float32}

        Raises:
            ValueError: If vocabulary is not provided.
                Set on instance creation or set `vocab` attribute.
        """
        if prior_probs is None and self.vocab is None:
            raise ValueError("prior_probs or vocab is required for this model")

        return generate_with_perplexity(
            self, n_batches, temp, batch_size, prior_probs, verbose
        )


class VocabMultinomialCLM(MultinomialCLM):
    """Generating samples with multinomial sampling with vocabulary.

    Attributes:
        model (tf.keras.Model): Model to wrap with LSTM layers.
        vocab (Vocabulary): Vocabulary.
        seq_len (int): Sequence length.
        dims (list[int]): Number of neurons per LSTM layers.
        has_embedding (bool): Whether the model has an Embedding layer.
        start_token (tf.Tensor): Start token for generation.
    """

    ASSERT_VOCAB = True
    """Needs to be initialized with a vocabulary."""
    vocab: VocabProto
    """@private Vocabulary"""


__all__ = ["MultinomialCLM", "VocabMultinomialCLM"]
