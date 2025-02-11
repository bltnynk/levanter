from dataclasses import field
from typing import Any, Callable, Dict, Protocol, Tuple, TypeAlias, TypeVar, Union

import equinox as eqx
import jax
from jaxtyping import PyTree

import haliax as hax
from haliax.types import Scalar


M = TypeVar("M")  # Model
M_con = TypeVar("M_con", contravariant=True)  # Model
X = TypeVar("X", contravariant=True)  # Input


ExtraData: TypeAlias = Dict[str, jax.Array | eqx.Module]

try:
    from haliax.nn.scan import BlockFoldable
except ImportError:

    class BlockFoldable(Protocol[M]):  # type: ignore
        def fold(self, *args, **kwargs):
            ...

        def scan(self, *args, **kwargs):
            ...


class ValAndGradFn(Protocol[M, X]):
    def __call__(self, model: M, *inputs: X, **input_kwargs) -> Tuple[Scalar, M]:
        ...


class ValFn(Protocol[M_con, X]):
    def __call__(self, model: M_con, *inputs: X, **input_kwargs) -> Scalar:
        ...


FilterSpec = Union[bool, Callable[[Any], bool]]
"""
A filter specification. Typically used on a pytree to filter out certain subtrees. Boolean values are
treated as-is, while callables are called on each element of the pytree. If the callable returns True, the element
is kept, otherwise it is filtered out.
"""

FilterTree = FilterSpec | PyTree[FilterSpec]


def add_merge(res: ExtraData, input: ExtraData):
    for k, v in input.items():
        if k not in res:
            res[k] = v
        else:
            res[k] = res[k] + v


class Extras(eqx.Module):
    loggable: ExtraData = field(default_factory=dict)
    aux: ExtraData = field(default_factory=dict)

    def merge(self, other: "Extras") -> "Extras":
        res = Extras()
        add_merge(res.loggable, self.loggable)
        add_merge(res.loggable, other.loggable)
        add_merge(res.aux, self.aux)
        add_merge(res.aux, other.aux)
        return res


class ComputeLossFunction(Protocol[M_con, X]):
    """
    Function signature for "compute_loss" functions in Levanter: these
    couple the computation of the logits and the evaluation of the loss
    """

    def __call__(
        self,
        model: M_con,
        input: X,
        **kwargs,
    ) -> tuple[hax.NamedArray, hax.NamedArray, Extras]:
        ...
