import jax.numpy as jnp
from jax.random import PRNGKey

import haliax as hax

from levanter.utils.stat_utils import IndexCountHistogram, IndexCountUnique, MeanScalar


def test_topk_selected():
    Batch = hax.Axis("B", 2)
    Ax = hax.Axis("X", 4)
    logits1 = hax.NamedArray(jnp.array([[1, 1, 0, 0], [1, 0, 1, 0]]), axes=(Batch, Ax))
    logits2 = hax.NamedArray(jnp.array([[1, 0, 1, 0], [0, 1, 0, 1]]), axes=(Batch, Ax))

    acc1 = IndexCountHistogram.init(logits1.sum(Batch))
    assert acc1.hist.bucket_counts.tolist() == [2, 1, 1, 0]
    assert acc1.hist.bucket_limits.tolist() == list(range(Ax.size + 1))
    acc2 = IndexCountHistogram.init(logits2.sum(Batch))
    assert acc2.hist.bucket_counts.tolist() == [1, 1, 1, 1]
    acc3 = acc1 + acc2
    assert acc3.hist.bucket_counts.tolist() == [3, 2, 2, 1]

    _, inds1 = hax.top_k(logits1, Ax, 2)
    _, inds2 = hax.top_k(logits2, Ax, 2)

    acc1 = IndexCountUnique.init(inds1, Ax)
    assert acc1.item() == 3
    acc2 = IndexCountUnique.init(inds2, Ax)
    assert acc2.item() == 4
    acc3 = acc1 + acc2
    assert acc3.item() == 4


def test_mean_scalar():

    AccumStep = hax.Axis("Accum", 8)
    MicroBatch = hax.Axis("MB", 2)
    Ax = hax.Axis("X", 4)

    arr = hax.random.normal(PRNGKey(0), (AccumStep, MicroBatch, Ax))

    acc = MeanScalar.zero()

    for i in range(AccumStep.size):
        mean_scalar = MeanScalar.init(arr[AccumStep, i], where=None)
        acc += mean_scalar

    assert jnp.allclose(acc.item(), arr.mean().item())
