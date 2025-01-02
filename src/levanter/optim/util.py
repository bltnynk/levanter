import equinox as eqx
import jax
import optax
import haliax as hax
import haliax.nn as hnn
import dataclasses

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

def filter_embedding_grads(optimizer: optax.GradientTransformation, Embed, Vocab, token_mask: hax.NamedArray):
    def where(m):
        return [m.embeddings]

    def replace_fn(x):
        assert hasattr(x, 'token_embeddings') and isinstance(x.token_embeddings, hnn.Embedding) and x.token_embeddings.weight is not None
        new_grads = x.token_embeddings.weight * token_mask.broadcast_to((Vocab, Embed))
        new_token_embeddings = dataclasses.replace(x.token_embeddings, weight=new_grads)
        return dataclasses.replace(x, token_embeddings=new_token_embeddings)

    def update_fn(updates, state, params=None):
        del params
        updates = eqx.tree_at(where, updates, replace_fn=replace_fn)
        return updates, state

    mask_transform = optax.GradientTransformation(lambda _: optax.EmptyState(), update_fn)

    return optax.chain(optimizer, mask_transform)
