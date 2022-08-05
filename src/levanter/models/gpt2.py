import functools
from dataclasses import dataclass
from typing import Optional, List, Callable

import equinox as eqx
import equinox.nn as nn
from equinox.custom_types import Array

import levanter.nn as pnn
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from einops import rearrange

from haliax import Axis, NamedArray
import haliax as hax
from levanter import jax_utils
from levanter.modeling_utils import ACT2FN


@dataclass(frozen=True)
class Gpt2Config:
    seq_len: int = 512
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12

    initializer_range: float = 0.02
    embed_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu_new"

    gradient_checkpointing: bool = False
    gradient_checkpointing_block_size: int = 5

    # Axes
    @property
    def seqlen(self) -> Axis:
        return Axis(name="seqlen", size=self.seq_len)

    @property
    def hidden(self) -> Axis:
        return Axis(name="hidden", size=self.hidden_dim)


class NamedLinear(eqx.Module):
    weight: NamedArray
    bias: Optional[NamedArray]

    in_axis: Axis = eqx.static_field()
    out_axis: Axis = eqx.static_field()

    def __init__(self, in_axis: Axis, out_axis: Axis, *, key, include_bias=True):
        self.weight = hax.random.normal(key, (in_axis, out_axis)) * 0.02
        if include_bias:
            self.bias = hax.zeros(out_axis)
        else:
            self.bias = None

        self.in_axis = in_axis
        self.out_axis = out_axis

    def __call__(self, inputs):
        # out = inputs.dot(self.in_axis, self.weight)

        kernel = self.weight.array
        return inputs @ kernel + self.bias.array


class Gpt2Mlp(eqx.Module):
    act: Callable = eqx.static_field()
    c_fc: NamedLinear
    c_proj: NamedLinear

    def __init__(self, hidden: Axis, intermediate: Axis, activation_fn, *, key):

        k_fc, k_proj = jrandom.split(key, 2)
        self.c_fc = NamedLinear(out_axis=intermediate, in_axis=hidden, key=k_fc)
        self.c_proj = NamedLinear(out_axis=hidden, in_axis=intermediate, key=k_proj)
        self.act = ACT2FN[activation_fn]

    def __call__(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class Gpt2Attention(eqx.Module):
    causal: bool = eqx.static_field()
    head_dim: Axis = eqx.static_field()
    num_heads: int = eqx.static_field()

    c_attn: NamedLinear
    c_proj: NamedLinear
    dropout: pnn.Dropout

    @property
    def total_head_dim(self):
        return self.head_dim.size * self.num_heads

    def __init__(self, in_dim: Axis, num_heads: int, head_dim: Axis, dropout_prob: float, *, key, causal: bool = True):
        self.causal = causal
        self.num_heads = num_heads
        self.head_dim = head_dim

        k_c, k_proj = jrandom.split(key, 2)

        qkv = hax.Axis(name="qkv", size=3 * self.total_head_dim)
        total_head_dim = Axis(name="total_head_dim", size=self.total_head_dim)
        self.c_attn = NamedLinear(out_axis=qkv, in_axis=in_dim, key=k_c)
        self.c_proj = NamedLinear(out_axis=in_dim, in_axis=total_head_dim, key=k_proj)
        self.dropout = pnn.Dropout(dropout_prob)

    # TODO: cross-attention
    # TODO: reorder_and_upcast_attn
    # TODO: scale_attn_by_inverse_layer_idx
    # @eqx.filter_jit
    def __call__(self, hidden_states: Array, inference: bool = True, *, key):
        # hidden_states has shape [seq_len, embed_dim]
        rng_key = key

        qkv_out = self.c_attn(hidden_states)  # [seq_len, 3 * embed_dim]
        # TODO(haliax): split for named
        query, key, value = jnp.split(qkv_out, 3, axis=-1)  # [seq_len, embed_dim]

        query = self._split_heads(query)  # [seq_len, num_heads, head_dim]
        key = self._split_heads(key)
        value = self._split_heads(value)

        # must use negative indexing to please the pmap gods
        query_length, key_length = query.shape[-3], key.shape[-3]

        attn_weights = jnp.einsum('... n h d, ... m h d -> ... h n m', query, key)  # [heads, seq_len, seq_len]
        attn_weights = attn_weights * lax.rsqrt(float(value.shape[-1]))

        if self.causal is not None:
            seq_len = hidden_states.shape[-2]  # TODO(haliax): fix for named arrays
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            causal_mask = causal_mask[:query_length, :key_length]

            attn_weights = jnp.where(causal_mask, attn_weights, -1E9)
            # causal_mask = causal_mask.astype(jnp.bfloat16)
            # mask = jnp.broadcast_to(attention_mask, w.shape)
            # w = jnp.where(mask > 0, w, -1E9)
            # attn_weights = attn_weights + attention_mask

        attn_weights = jnn.softmax(attn_weights)  # heads, seqlen, seqlen
        attn_weights = self.dropout(attn_weights, key=rng_key, inference=inference)

        attn_output = jnp.einsum('... h n m, ... m h d -> ... n h d', attn_weights, value)  # [seq_len, head, head_dim]

        attn_output = self._merge_heads(attn_output)  # [seq_len, total_head_dim]
        attn_output = self.c_proj(attn_output)

        return attn_output

    def _split_heads(self, hidden_states: Array["seq_len", "embed_dim"]) -> Array["seq_len", "num_heads", "head_dim"]:
        return rearrange(hidden_states, '... n (h d) -> ... n h d', h=self.num_heads)

    @staticmethod
    def _merge_heads(hidden_states: Array["seq_len", "num_heads", "head_dim"]) -> Array["seq_len", "num_heads", "embed_dim"]:
        return rearrange(hidden_states, '... n h d -> ... n (h d)')


class Gpt2Block(eqx.Module):
    ln_1: nn.LayerNorm
    attn: Gpt2Attention
    ln_2: nn.LayerNorm
    mlp: Gpt2Mlp
    resid_dropout: pnn.Dropout

    def __init__(self, config: Gpt2Config, *, key):
        k_attn, k_cross, k_mlp = jrandom.split(key, 3)

        hidden = config.hidden
        inner_dim = Axis("mlp", 4 * hidden.size)
        head_dim = Axis("head", hidden.size // config.num_heads)

        assert hidden.size % config.num_heads == 0, \
            f"embed_dim={hidden} must be divisible by num_heads={config.num_heads}"

        self.ln_1 = nn.LayerNorm(hidden.size, eps=config.layer_norm_epsilon)
        self.attn = Gpt2Attention(hidden, num_heads=config.num_heads, head_dim=head_dim,
                                  dropout_prob=config.attn_pdrop, key=k_attn, causal=True)
        self.resid_dropout = pnn.Dropout(p=config.resid_pdrop)
        self.ln_2 = nn.LayerNorm(hidden.size, eps=config.layer_norm_epsilon)

        self.mlp = Gpt2Mlp(hidden=hidden, intermediate=inner_dim, activation_fn=config.activation_function, key=k_mlp)

    # @eqx.filter_jit
    def __call__(self, hidden_states: Array, inference=True, *, key):
        k1, k2, k3 = jax_utils.maybe_rng_split(key, 3)

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, inference=inference, key=k1)
        attn_output = self.resid_dropout(attn_output, key=k2, inference=inference)
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        ff_output = self.mlp(hidden_states)
        ff_output = self.resid_dropout(ff_output, inference=inference, key=k3)

        hidden_states = ff_output + residual

        return hidden_states


class Gpt2Transformer(eqx.Module):
    config: Gpt2Config = eqx.static_field()
    blocks: List[Gpt2Block]
    ln_f: nn.LayerNorm

    def __init__(self, config: Gpt2Config, *, key):
        super().__init__()
        self.config = config

        self.blocks = [
            Gpt2Block(config, key=k) for i, k in enumerate(jrandom.split(key, config.num_layers))
        ]
        self.ln_f = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)

    # @eqx.filter_jit
    def __call__(self, hidden_states: Array, inference=True, *, key) -> Array:
        keys = jax_utils.maybe_rng_split(key, len(self.blocks))

        if not self.config.gradient_checkpointing:
            for block, k_block, i in zip(self.blocks, keys, range(len(self.blocks))):
                hidden_states = block(hidden_states, inference=inference, key=k_block)
        else:
            for block, k_block, i in zip(self.blocks, keys, range(len(self.blocks))):
                hidden_states = jax.remat(functools.partial(block, inference=inference))(hidden_states, key=k_block)
            # hidden_states = recursive_checkpoint(
            #     [ functools.partial(block, inference=inference, key=k_block) for block, k_block in zip(self.blocks, keys)],
            #     threshold=self.config.gradient_checkpointing_block_size)(hidden_states)

        hidden_states = self.ln_f(hidden_states)

        return hidden_states


# from https://github.com/google/jax/issues/4285
def recursive_checkpoint(funs, threshold = 2):
    if len(funs) == 1:
        return funs[0]
    elif len(funs) == 2:
        f1, f2 = funs
        return lambda x: f2(f1(x))
    elif len(funs) <= threshold:
        return functools.reduce(lambda f, g: lambda x: g(f(x)), funs)
    else:
        f1 = recursive_checkpoint(funs[:len(funs)//2])
        f2 = recursive_checkpoint(funs[len(funs)//2:])
        return lambda x: f2(jax.remat(f1)(x))


class Gpt2Embeddings(eqx.Module):
    token_embeddings: NamedArray
    position_embeddings: NamedArray
    token_out_embeddings: Optional[NamedArray]
    dropout: pnn.Dropout

    # axes
    vocab: Axis = eqx.static_field()
    seqlen: Axis = eqx.static_field()
    hidden: Axis = eqx.static_field()

    def __init__(self,
                 embed: Axis,
                 vocab: Axis,
                 seqlen: Axis,
                 initializer_range: float,
                 tie_word_embeddings: bool,
                 dropout_prob: float, *, key):
        super().__init__()
        k_wte, k_wpe, k_out = jrandom.split(key, 3)

        self.vocab = vocab
        self.seqlen = seqlen
        self.hidden = embed

        self.token_embeddings = hax.random.normal(key=k_wte,
                                                  shape=(vocab, embed)) * initializer_range
        self.position_embeddings = hax.random.normal(key=k_wpe,
                                                     shape=(seqlen, embed)) * (initializer_range / 2)
        self.dropout = pnn.Dropout(p=dropout_prob)

        if tie_word_embeddings:
            self.token_out_embeddings = None
        else:
            self.token_out_embeddings = hax.random.normal(key=k_out,
                                                          shape=(vocab, embed)) * initializer_range

    def embed(self, input_ids, inference, *, key):
        # TODO: select
        # input_embeds = self.token_embeddings.select(self.vocab, input_ids)
        # position_embeds = self.position_embeddings.select(self.seqlen, jnp.arange(input_ids.shape[-1], dtype="i4"))
        input_embeds = self.token_embeddings.array[input_ids]
        position_embeds = self.position_embeddings.array[jnp.arange(input_ids.shape[-1], dtype="i4")]
        hidden_states = input_embeds + position_embeds
        hidden_states = self.dropout(hidden_states, inference=inference, key=key)

        return hidden_states

    def unembed(self, hidden_states: Array):
        embeddings = self.token_out_embeddings or self.token_embeddings
        # return hax.dot(self.hidden, hidden_states, embeddings)
        return jnp.einsum('... l h, ... v h -> ... l v', hidden_states, embeddings.array)


class Gpt2LMHeadModel(eqx.Module):
    transformer: Gpt2Transformer
    embeddings: Gpt2Embeddings

    @property
    def config(self):
        return self.transformer.config

    def __init__(self, vocab: Axis, config: Gpt2Config, *, key):
        k_t, k_embeddings = jrandom.split(key, 2)
        self.transformer = Gpt2Transformer(config, key=k_t)
        self.embeddings = Gpt2Embeddings(vocab=vocab,
                                         embed=config.hidden,
                                         seqlen=config.seqlen,
                                         initializer_range=config.initializer_range,
                                         tie_word_embeddings=True,
                                         dropout_prob=config.embed_pdrop,
                                         key=k_embeddings)

    def __call__(self, input_ids, key):
        k_embed, k_transformer = jax_utils.maybe_rng_split(key, 2)
        hidden_states = self.embeddings.embed(input_ids, inference=key is None, key=k_embed)
        hidden_states = self.transformer(hidden_states, inference=key is None, key=k_transformer)
        lm_logits = self.embeddings.unembed(hidden_states)

        return lm_logits