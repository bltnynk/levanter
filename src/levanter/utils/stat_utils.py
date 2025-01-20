from typing import Optional, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import Self

import haliax as hax

from levanter.tracker.histogram import Histogram, sharded_histogram
from levanter.utils.jax_utils import Zeroable


Arrayish: TypeAlias = hax.NamedArray | np.ndarray | jnp.ndarray


class SumScalar(eqx.Module):
    value: jnp.ndarray

    def item(self) -> float:
        return self.value.item()

    def __add__(self, other: Self) -> Self:
        return SumScalar(self.value + other.value)


class MeanScalar(eqx.Module):
    total: jnp.ndarray
    mean: jnp.ndarray

    @staticmethod
    def init(array: hax.NamedArray, where: Optional[hax.NamedArray] = None) -> "MeanScalar":
        total = where.sum() if where is not None else hax.named(array.size, ())
        mean = hax.mean(array, where=where).scalar()
        return MeanScalar(total.scalar().astype(jnp.float32), mean)

    def item(self) -> float:
        return self.mean.item()

    def __add__(self, other: Self) -> Self:
        delta = other.mean - self.mean
        # careful: total and self.total can be 0
        new_total = self.total + other.total
        ratio = jax.lax.select(new_total > 0, other.total / new_total, jnp.array(0.0))
        new_mean = self.mean + delta * ratio
        new_total = self.total + other.total
        return MeanScalar(new_total, new_mean)

    @staticmethod
    def zero() -> "MeanScalar":
        return MeanScalar(jnp.zeros(()), jnp.zeros(()))


class IndexCountUnique(eqx.Module):
    seen: jnp.ndarray

    @staticmethod
    def init(inds: hax.NamedArray, axis: hax.Axis) -> "IndexCountUnique":
        seen = hax.zeros(axis, dtype=jnp.bool).at[axis, inds.array.flatten()].set(True)
        return IndexCountUnique(seen)

    def item(self) -> int:
        return self.seen.sum().item()

    def __add__(self, other: Self) -> Self:
        return IndexCountUnique(self.seen | other.seen)


def combine_histograms(hist1: Histogram, hist2: Histogram) -> Histogram:
    return Histogram(
        min=jax.lax.min(hist1.min, hist2.min),
        max=jax.lax.max(hist1.max, hist2.max),
        num=hist1.num + hist2.num,
        sum=hist1.sum + hist2.sum,
        sum_squares=hist1.sum_squares + hist2.sum_squares,
        # this is a hack so that when microbatching, we don't carry
        # over 0 bucket limits from the zero init
        bucket_limits=hist2.bucket_limits,
        bucket_counts=hist1.bucket_counts + hist2.bucket_counts,
    )


def _logit_buckets():
    return jnp.concatenate([-jnp.logspace(6, -7, 64), jnp.array([0]), jnp.logspace(-7, 6, 64)])


class LogitHistogram(eqx.Module, Zeroable):
    hist: Histogram

    @staticmethod
    def init(
        data: Arrayish,
    ) -> "LogitHistogram":
        bins = _logit_buckets()
        counts, edges = sharded_histogram(data, edges=bins)
        return LogitHistogram(
            Histogram(
                min=data.min().scalar(),
                max=data.max().scalar(),
                num=data.size,
                sum=data.sum().scalar(),
                sum_squares=(data**2).sum().scalar(),
                bucket_limits=edges,
                bucket_counts=counts,
            )
        )

    def item(self) -> Histogram:
        return self.hist

    def __add__(self, other: Self) -> Self:
        return LogitHistogram(combine_histograms(self.hist, other.hist))

    def zeros_like(self) -> "LogitHistogram":
        return LogitHistogram(
            Histogram(
                min=jnp.zeros(()),
                max=jnp.zeros(()),
                num=jnp.zeros_like(self.hist.num),
                sum=jnp.zeros_like(self.hist.sum),
                sum_squares=jnp.zeros_like(self.hist.sum_squares),
                bucket_limits=_logit_buckets(),
                bucket_counts=jnp.zeros_like(self.hist.bucket_counts),
            )
        )


class IndexCountHistogram(eqx.Module, Zeroable):
    hist: Histogram

    @staticmethod
    def init(counts: Arrayish) -> "IndexCountHistogram":
        Bin = hax.Axis("bin", counts.size + 1)
        limits = hax.arange(Bin)
        sum = counts * hax.arange(counts.axes[0])
        sum2 = counts * (hax.arange(counts.axes[0]) ** 2)
        hist = Histogram(
            min=limits.min().scalar(),
            max=(limits.max() - 1).scalar(),
            num=jnp.array(counts.size),
            sum=sum.sum().scalar(),
            sum_squares=sum2.sum().scalar(),
            bucket_limits=limits.array,
            bucket_counts=counts.array,
        )
        return IndexCountHistogram(hist)

    def item(self) -> Histogram:
        return self.hist

    def __add__(self, other: Self) -> Self:
        return IndexCountHistogram(combine_histograms(self.hist, other.hist))

    def zeros_like(self) -> "IndexCountHistogram":
        return IndexCountHistogram(
            Histogram(
                min=jnp.zeros(()),
                max=jnp.array(self.hist.bucket_limits.size - 1),
                num=jnp.zeros_like(self.hist.num),
                sum=jnp.zeros_like(self.hist.sum),
                sum_squares=jnp.zeros_like(self.hist.sum_squares),
                bucket_limits=jnp.arange(self.hist.bucket_limits.size),
                bucket_counts=jnp.zeros_like(self.hist.bucket_counts),
            )
        )


class RunningMean(eqx.Module):
    mean: Arrayish
    total: Arrayish

    @staticmethod
    def zeros_like(x: Arrayish) -> "RunningMean":
        return RunningMean(x * 0.0, x * 0.0)

    def add(self, x: Arrayish, total: Arrayish) -> "RunningMean":
        delta = x - self.mean
        # careful: total and self.total can be 0
        new_total = self.total + total
        ratio = hax.where(new_total, total / new_total, 0.0)
        new_mean = self.mean + delta * ratio
        new_total = self.total + total
        return RunningMean(new_mean, new_total)

    def item(self) -> float:
        return self.mean.item()

    def __add__(self, other: "RunningMean"):
        return self.add(other.mean, other.total)

    def __str__(self):
        return f"RunningMean(mean={self.mean}, total={self.total})"
