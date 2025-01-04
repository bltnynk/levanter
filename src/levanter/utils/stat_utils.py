from typing import Optional, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import Self

import haliax as hax

from levanter.tracker.histogram import Histogram
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


class IndexCountHistogram(eqx.Module, Zeroable):
    hist: Histogram

    @staticmethod
    def init(inds: Arrayish, axis: hax.Axis) -> "IndexCountHistogram":
        counts = hax.zeros(axis, dtype=jnp.int32).at[axis, inds].add(1)
        Bin = hax.Axis("bin", axis.size + 1)
        limits = hax.arange(Bin)
        hist = Histogram(
            min=limits.min().scalar(),
            max=(limits.max() - 1).scalar(),
            num=jnp.array(inds.size),
            sum=inds.sum().scalar(),
            sum_squares=(inds**2).sum().scalar(),
            bucket_limits=limits.array,
            bucket_counts=counts.array,
        )
        return IndexCountHistogram(hist)

    def item(self) -> Histogram:
        return self.hist

    def __add__(self, other: Self) -> Self:
        new_hist = Histogram(
            min=jax.lax.min(self.hist.min, other.hist.min),
            max=jax.lax.max(self.hist.max, other.hist.max),
            num=self.hist.num + other.hist.num,
            sum=self.hist.sum + other.hist.sum,
            sum_squares=self.hist.sum_squares + other.hist.sum_squares,
            # this is a hack so that when microbatching, we don't carry
            # over 0 bucket limits from the zero init
            bucket_limits=other.hist.bucket_limits,
            bucket_counts=self.hist.bucket_counts + other.hist.bucket_counts,
        )
        return IndexCountHistogram(new_hist)

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
