import jax.numpy as jnp
from jax.random import PRNGKey

import haliax as hax

from levanter.models.routed_qwen_model import create_expert_mask_from_acts
from levanter.utils.stat_utils import IndexCountHistogram, IndexCountUnique, MeanScalar, RunningMean


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

    TopK = hax.Axis("TopK", 2)
    _, inds1 = hax.top_k(logits1, Ax, TopK.size, TopK)
    mask1 = create_expert_mask_from_acts(TopK, Ax, inds1, logits1)
    _, inds2 = hax.top_k(logits2, Ax, TopK.size, TopK)
    mask2 = create_expert_mask_from_acts(TopK, Ax, inds2, logits2)

    acc1 = IndexCountUnique.init(mask1, Ax)
    assert acc1.item() == 3
    acc2 = IndexCountUnique.init(mask2, Ax)
    assert acc2.item() == 4
    acc3 = acc1 + acc2
    assert acc3.item() == 4


def test_mean_scalar():

    AccumStep = hax.Axis("Accum", 8)
    MicroBatch = hax.Axis("MB", 2)
    Ax = hax.Axis("X", 4)

    arr = hax.random.normal(PRNGKey(0), (AccumStep, MicroBatch, Ax))
    where = hax.random.bernoulli(PRNGKey(1), (AccumStep, MicroBatch, Ax), 0.5)

    acc = MeanScalar.zero()
    rmacc = RunningMean.zeros_like(jnp.array(0))

    for i in range(AccumStep.size):
        wi = where[AccumStep, i]
        mean_scalar = MeanScalar.init(arr[AccumStep, i], where=wi)
        acc += mean_scalar
        rmacc += RunningMean(arr[AccumStep, i].mean(where=wi), wi.sum())

    assert jnp.allclose(acc.item(), arr.mean(where=where).item())
    assert jnp.allclose(rmacc.item(), arr.mean(where=where).item())
