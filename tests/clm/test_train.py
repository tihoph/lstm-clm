# %%
"""Test training."""

# ruff: noqa: ERA001
from __future__ import annotations

from pathlib import Path

MAX_LEN = 140
TEST_SIZE = 1000


def load_smis() -> list[str]:
    """Load 1000 SMILES."""
    smis = Path("speeds/wo.smi").read_text("utf-8").splitlines()
    smis = [smi for smi in smis if len(smi) <= MAX_LEN]
    return smis[:TEST_SIZE]


# %%


# r = RandomizedDataset(smis, vocab, n_precalc=10)
# v = RandomizedDataset(smis, vocab, n_precalc=10)

# R = r.build_dataset(128)
# V = v.build_dataset(128)

# iter_gen = iter(R.dataset)
# iter_val = iter(V.dataset)

# for epoch in range(10):
#     for step in range(R.steps):
#         x, y = next(iter_gen)
#         x_, y_ = next(iter_val)
#         print(x.shape)


# # %%
# # %%time
# epochs = 10

# all_smis = smis * epochs

# rnd_smis = auto_parallel(randomize_smiles, all_smis)


# def map_fn(x):
#     return (x[:, :-1], x[:, 1:])


# for epoch_smis in range(0, len(rnd_smis), len(smis)):
#     inputs = vocab.encode(rnd_smis[epoch_smis : epoch_smis + len(smis)])
#     dataset = (
#         tf.data.Dataset.from_tensor_slices(inputs)
#         .batch(128)
#         .map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
#         .prefetch(tf.data.AUTOTUNE)
#     )
#     model.fit(dataset, epochs=1, verbose=1)


# # %%
# dataset = (
#     tf.data.Dataset.from_tensor_slices(inputs)
#     .batch(128)
#     .map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
#     .prefetch(tf.data.AUTOTUNE)
# )
# for ix, x in enumerate(dataset):
#     print(ix)
# # %%
