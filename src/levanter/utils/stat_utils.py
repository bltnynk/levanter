from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from typing_extensions import Self

import haliax as hax

from levanter.tracker.histogram import Histogram


Arrayish: TypeAlias = hax.NamedArray | np.ndarray | jnp.ndarray


class SumScalar(eqx.Module):
    value: jnp.ndarray

    def item(self) -> float:
        return self.value.item()

    def __add__(self, other: Self) -> Self:
        return SumScalar(self.value + other.value)


class MeanScalar(eqx.Module):
    total: jnp.ndarray
    sum: jnp.ndarray

    @staticmethod
    def init(asum: Arrayish, total: Arrayish) -> "MeanScalar":
        return MeanScalar(asum.astype(jnp.float32), total.astype(jnp.float32))

    def item(self) -> float:
        return self.sum.item() / self.total.item()

    def __add__(self, other: Self) -> Self:
        return MeanScalar(self.sum + other.sum, self.total + other.total)


class IndexCountHistogram(eqx.Module):
    hist: Histogram

    @staticmethod
    def init(inds: Arrayish, axis: hax.Axis) -> "IndexCountHistogram":
        counts = hax.zeros(axis, dtype=jnp.int32).at[axis, inds].add(1)
        Bin = hax.Axis("bin", axis.size + 1)
        limits = hax.arange(Bin)
        hist = Histogram(
            min=limits.min(),
            max=limits.max() - 1,
            num=inds.size,
            sum=inds.sum(),
            sum_squares=(inds**2).sum(),
            bucket_limits=limits,
            bucket_counts=counts,
        )
        return IndexCountHistogram(hist)

    def item(self) -> Histogram:
        return self.hist

    def __add__(self, other: Self) -> Self:
        new_hist = Histogram(
            min=self.hist.min,
            max=self.hist.max,
            num=self.hist.num + other.hist.num,
            sum=self.hist.sum + other.hist.sum,
            sum_squares=self.hist.sum_squares + other.hist.sum_squares,
            bucket_limits=self.hist.bucket_limits,
            bucket_counts=self.hist.bucket_counts + other.hist.bucket_counts,
        )
        return IndexCountHistogram(new_hist)


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
