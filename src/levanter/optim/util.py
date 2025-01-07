import dataclasses

import equinox as eqx
import jax
import optax

import haliax as hax
import haliax.nn as hnn

from levanter.utils.jax_utils import is_inexact_arrayish


def hvp(f, x, v):
    """Compute the Hessian-vector product of a function."""
    return eqx.filter_jvp(eqx.filter_grad(f), (x,), (v,))[1]


def tree_gaussian_like(key, tree):
    """
    Samples a tree of gaussian noise with the same structure as `tree`, except for leaves which are not inexact arrays,
    for which it returns None
    """
    leaves, structure = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    rand_n = lambda x, key: jax.random.normal(key, x.shape) if is_inexact_arrayish(x) else None
    g = jax.tree_util.tree_map(rand_n, leaves, list(keys))
    g = jax.tree_util.tree_unflatten(structure, g)

    return g


def filter_embedding_grads(Embed, Vocab, token_mask: hax.NamedArray):
    def replace_fn(x):
        if isinstance(x, hnn.Embedding):
            assert x.weight is not None, "No embedding updates, is embedding_ft True?"
            new_grads = x.weight * token_mask.broadcast_to((Vocab, Embed)).astype(x.weight.dtype)
            return dataclasses.replace(x, weight=new_grads)
        return x

    def update_fn(updates, state, params=None):
        del params
        updates = jax.tree.map(replace_fn, updates, is_leaf=lambda x: isinstance(x, hnn.Embedding))
        return updates, state

    mask_transform = optax.GradientTransformation(lambda _: optax.EmptyState(), update_fn)

    return mask_transform
