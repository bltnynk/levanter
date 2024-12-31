from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from typing_extensions import Self

import haliax as hax

from levanter.utils.types import Accumulatable


Arrayish: TypeAlias = hax.NamedArray | np.ndarray | jnp.ndarray


class SumScalar(Accumulatable):
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
