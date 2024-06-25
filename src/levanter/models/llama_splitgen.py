import dataclasses
import functools
import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, Type, TypeVar, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray, PyTree

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import Stacked

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.compat.torch_serialization import (
    StateDict,
    StateDictSerializationMixin,
    apply_prefix,
    flatten_linear_layers,
    stack_state_dict,
    unflatten_linear_layers,
    unstack_state_dict,
)
from levanter.logging import silence_transformer_nag
from levanter.models.attention import AttentionBackend, AttentionMask, dot_product_attention
from levanter.models.gpt2 import ACT2FN
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.jax_utils import key_iterator, leaf_key_paths


silence_transformer_nag()
from transformers import LlamaConfig as HfLlamaConfig  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402


M = TypeVar("M", bound=PyTree)

LORA_R = "LORA_R"


class LowRankLinear(eqx.Module):
    """
    A linear layer with a low-rank factorization. Used by LoRA.

    A LowRankLinear is basically 2 matrices that have a common dimension
    """

    lora_A: hnn.Linear
    lora_B: hnn.Linear
    scale: float = eqx.field(static=True)

    def __call__(self, x, key=None):
        z = self.lora_A(x)
        z = self.lora_B(z)
        return z * self.scale

    @staticmethod
    def init(In: hax.Axis, Out: Axis, r: int, *, key, alpha: float):
        """
        Initializes a LoraLinear module.
        """
        _R = hax.Axis(LORA_R, r)
        key_A, key_B = jax.random.split(key)
        # Peft always uses out_first=True (i.e. normal Torch convention) for linear, even for gpt2-style Conv1d
        lora_A = hnn.Linear.init(In, _R, key=key_A, use_bias=False, out_first=True)
        in_size = hax.axis_size(In)
        lora_A = dataclasses.replace(
            lora_A, weight=hax.random.truncated_normal(key_A, lora_A.weight.axes, -3, 3) / math.sqrt(in_size)
        )
        lora_B = hnn.Linear.init(_R, Out, key=key_B, use_bias=False, out_first=True)
        lora_B = dataclasses.replace(lora_B, weight=hax.zeros_like(lora_B.weight))
        return LowRankLinear(lora_A, lora_B, alpha / r)

    def merge(self) -> hax.NamedArray:
        return hax.dot(self.lora_A.weight, self.lora_B.weight, axis=LORA_R) * self.scale


class SplitLoraLinear(eqx.Module, StateDictSerializationMixin):
    """
    Linear layer with LoRA transform.
    """

    wrapped: hnn.Linear
    lora: LowRankLinear
    seq_axis: Axis = eqx.static_field()
    weight_axis: Axis = eqx.static_field()
    weight_lora_axis: Axis = eqx.static_field()

    def __call__(self, x: NamedArray, key=None):
        x1, x2 = x.split(self.seq_axis, (self.weight_axis, self.weight_lora_axis))
        if key is not None:
            k1, k2, k3 = jax.random.split(key, 3)
            p1 = self.wrapped(x1, key=k1)
            p2 = self.lora(x2, key=k2) + self.wrapped(x2, key=k3)
            return hax.concatenate(self.seq_axis, (p1, p2))
        else:
            p1 = self.wrapped(x1)
            p2 = self.lora(x2) + self.wrapped(x2)
            return hax.concatenate(self.seq_axis, (p1, p2))

    def merge(self):
        weight = self.lora.merge() + self.wrapped.weight
        return dataclasses.replace(self.wrapped, weight=weight)

    @staticmethod
    def init(wrapped: hnn.Linear, r: int, n_split: int, seq_axis: Axis, alpha: float, *, key):
        """
        Initializes a LoraLinear module.
        """
        weight_axis = Axis(seq_axis.name, n_split)
        weight_lora_axis = Axis(seq_axis.name, seq_axis.size - n_split)
        lora = LowRankLinear.init(wrapped.In, wrapped.Out, r, alpha=alpha, key=key)
        return SplitLoraLinear(wrapped, lora, seq_axis, weight_axis, weight_lora_axis)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"wrapped": None, "lora": None}


@LmConfig.register_subclass("llamasg")
@dataclass(frozen=True)
class SplitLlamaConfig(HFCompatConfig):
    """Config for LlamaModel

    Args:
        seq_len (int, optional): maximum length of the input sequence. Defaults to 2048.
        hidden_dim (int, optional): dimension of the hidden state. Defaults to 4096.
        intermediate_dim (int, optional): dimension of the intermediate state. Defaults to 11008.
        num_layers (int, optional): number of hidden layers in the Transformer encoder. Defaults to 32.
        num_heads (int, optional): number of attention heads for each attention layer. Defaults to 32.
        num_kv_heads (int, optional): number of attention heads for keys and values in each attention layer.
            Setting to 1 means MQA. Setting to num_heads means MHA. Otherwise GQA.
            Note that num_heads must be divisible by this number. Defaults to 32.
        activation_function (str, optional): activation function for the hidden layer. Defaults to "silu".
        rope_scaling (Dict, optional): dict containing the scaling configuration for the Rotary Positional Embedding.
    """

    seq_len: int = 2048
    hidden_dim: int = 4096
    intermediate_dim: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32
    activation_function: str = "silu"
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-5
    target_modules: Optional[Sequence[str]] = ("q_proj", "v_proj", "k_proj", "o_proj")
    """modules to loraize. can either be a regex or a list of strings of module names, or None, meaning all linear modules"""
    r: int = 8  # rank of LoRA transform
    alpha: Optional[float] = None  # scaling factor for LoRA transform

    skip_indices: Sequence[int] = (24, 25, 26, 27, 28, 29)
    skip_after_k_tokens: int = 256

    # Attention-related config
    upcast_attn: bool = False
    use_flash_attention: Optional[bool] = True
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = None

    gradient_checkpointing: bool = True
    gradient_checkpointing_block_size: int = 5
    scan_layers: bool = True

    use_bias: bool = False
    use_layer_norm_weight: bool = True
    rope_scaling: Optional[dict] = None
    rope_base: float = 10000.0

    reference_checkpoint: str = "meta-llama/Llama-2-7b-hf"
    tokenizer: Optional[str] = None

    # Axis
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
    Heads = property(lambda self: Axis(name="heads", size=self.num_heads))
    KVHeads = property(lambda self: Axis(name="kv_heads", size=self.num_kv_heads))
    Layers = property(lambda self: Axis(name="layers", size=self.num_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))
    HeadSize = property(lambda self: Axis(name="head_size", size=self.hidden_dim // self.num_heads))

    def __post_init__(self):
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), f"num_heads={self.num_heads} not divisible by num_kv_heads={self.num_kv_heads}."
        assert self.skip_after_k_tokens < self.seq_len, "skip_after_k_tokens must be less than seq_len."

    def matches_target(self, key_path):
        if isinstance(self.target_modules, str):
            compiled = re.compile(self.target_modules)
            return compiled.match(key_path) is not None
        elif self.target_modules is None:
            return True
        else:
            return any(key_path.endswith(target) for target in self.target_modules)

    def hf_checkpoint_converter(self) -> HFCheckpointConverter["SplitLlamaConfig"]:  # type: ignore
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint,
            trust_remote_code=True,
            tokenizer=self.tokenizer if self.tokenizer else self.reference_checkpoint,
            HfConfigClass=HfLlamaConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        return SplitLlamaConfig(
            seq_len=hf_config.max_position_embeddings,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            activation_function=hf_config.hidden_act,
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            rope_scaling=hf_config.rope_scaling,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfLlamaConfig:
        """Convert to HuggingFace's LlamaConfig

        Args:
            vocab_size (int, optional): Vocabulary size of the tokenizer. Defaults to 32000.
            config_overrides (dict, optional): Overrides for the config. Defaults to None.

        Returns:
            HfLlamaConfig: HuggingFace's LlamaConfig
        """
        if config_overrides is None:
            config_overrides = {}

        return HfLlamaConfig(
            max_position_embeddings=self.seq_len,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            hidden_act=self.activation_function,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.layer_norm_epsilon,
            rope_scaling=self.rope_scaling,
            vocab_size=vocab_size,
            **config_overrides,
        )

    @property
    def model_type(self) -> Type["LlamaLMHeadModel"]:
        return LlamaLMHeadModel

    def mk_LayerNorm(self, axis: Axis) -> "LlamaRMSNorm":
        return LlamaRMSNorm.init(
            axis, eps=self.layer_norm_epsilon, use_weight=self.use_layer_norm_weight, use_bias=self.use_bias
        )

    def flops_per_token(self, vocab_size: int):
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            vocab_size=vocab_size,
            glu=True,
        )

    def loraize(self, model: M, key: jax.random.PRNGKey, prefix: str = "", batch_dims: Tuple[Axis, ...] = ()) -> M:
        """
        This implementation is mostly straightforward, with one major wrinkle: scan layers like Stacked, which
        add an extra batch dimension and thus require vmap, and thus require a vmap'ed LoRA transform.

        As an example, the GPT-2 Model has a Stacked[Gpt2Block] member, which means that all members of the Gpt2Block
        have an extra Layers dimension. E.g. the c_attn will have shape weight=NamedArray(['layers', 'embed', 'qkv', 'heads', 'head_size']
        even though it's defined as just Linear(In=Embed, Out=(Qkv, Heads, HeadSize)). The Stacked adds the initial Layers dimension.

        There are two ways we can approach scan layers: one is to ask implementors of lora layers to handle
        this themselves, and the other is to handle it here. The former is more flexible, but the latter is
        more convenient, even if it runs the risk of being a leaky abstraction. We choose the latter.
        """
        key_iter = key_iterator(key)

        def _is_special_module(module):
            return isinstance(module, hnn.Linear)

        def _batchify_ctor(ctor):
            # this is gross but it basically just vmaps the ctor over each batch dimension
            return functools.reduce(lambda ctor, batch_axis: hax.vmap(ctor, batch_axis), reversed(batch_dims), ctor)

        def _loraize_module(module, key_path):
            # we don't want to loraize layers that are skipped
            is_skipped_layer = any(f".{idx}." in key_path for idx in self.skip_indices)
            if self.matches_target(key_path) and isinstance(module, hnn.Linear) and not is_skipped_layer:
                my_key = next(key_iter)
                batched_key = shaped_rng_split(my_key, [axis.size for axis in batch_dims])
                alpha = self.alpha if self.alpha is not None else float(self.r)
                return _batchify_ctor(SplitLoraLinear.init)(
                    module, self.r, self.skip_after_k_tokens, self.Pos, alpha=alpha, key=batched_key
                )
            else:
                return module

        return jax.tree_util.tree_map(
            _loraize_module,
            model,
            leaf_key_paths(model, is_leaf=_is_special_module, prefix=prefix),
            is_leaf=_is_special_module,
        )

    def splitize(self, model: M) -> M:
        """
        Splits the decoder layer into two parts: one that is loraized and one that is not.
        """

        def _is_decoder_layer(module):
            return isinstance(module, LlamaDecoderLayer)

        def _split_decoder_layer(module, key_path):
            if _is_decoder_layer(module) and any(
                f"transformer.layers.blocks.{idx}" == key_path for idx in self.skip_indices
            ):
                return SplitDecoderWrapper.init(module)
            else:
                return module

        return jax.tree_util.tree_map(
            _split_decoder_layer,
            model,
            leaf_key_paths(model, is_leaf=_is_decoder_layer),
            is_leaf=_is_decoder_layer,
        )

    def is_trainable_filter(self, model: M) -> M:
        """
        Creates a filter tree suitable for passing to Trainer.is_trainable marking which parameters are trainable and which
        are not.

        Returns:
        (PyTree) A filter tree marking which parameters are trainable and which are not. This filter is the same as the model,
        except every LoRA param is replaced with True and every other leaf (really, every array) is replaced with False.
        """

        # We only want to train on the lora params. The way to do this in Equinox is generally with
        # a filter tree (cf https://docs.kidger.site/equinox/examples/frozen_layer/),
        # which is a tree with the same structure (or a "tree prefix" thereof) as the model, but with
        # bools or Callable[..., bool] at the leaves. We can then pass this tree to the trainer and it
        # will only train the parameters that are True in the tree.
        # Levanter defines `is_lora_param` for this purpose, but we need to be careful about how we use it.
        # Equinox's primitives don't really have a "match all tree nodes matching a predicate" function (just
        # a "match all tree leaves matching a predicate" function), so we need to be just a bit careful.
        # Basically, we want to halt recursion in the tree whenever we hit a node that is a lora param.
        def is_lora_param(node):
            return isinstance(node, LowRankLinear)

        return jax.tree_util.tree_map(is_lora_param, model, is_leaf=is_lora_param)


class LlamaMlp(eqx.Module, StateDictSerializationMixin):
    """Multi-layer Perceptron
    In comparison with GPT2, LlamaMlp adds an up-proj that multiplies with activated gate_proj,
    before down-proj.
    """

    gate_proj: hnn.Linear  # projection from Embed to Mlp
    up_proj: hnn.Linear  # projection from Embed to Mlp
    down_proj: hnn.Linear  # projection from Mlp to Embed
    act: Callable = eqx.static_field()

    @staticmethod
    def init(
        Embed: Axis, Mlp: Axis, activation_fn: Union[str, Callable], *, key, use_bias: bool = False
    ) -> "LlamaMlp":
        k_fc, k_up_proj, k_down_proj = jrandom.split(key, 3)
        gate_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_fc, use_bias=use_bias, out_first=True)
        up_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_up_proj, use_bias=use_bias, out_first=True)
        down_proj = hnn.Linear.init(Out=Embed, In=Mlp, key=k_down_proj, use_bias=use_bias, out_first=True)
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        act = activation_fn  # type: ignore
        return LlamaMlp(gate_proj, up_proj, down_proj, act)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        k_gate, k_up, k_down = maybe_rng_split(key, 3)
        hidden_states = self.gate_proj(x, key=k_gate)
        hidden_states = self.act(hidden_states)
        hidden_states = hidden_states * self.up_proj(x, key=k_up)
        outputs = self.down_proj(hidden_states, key=k_down)
        return outputs

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        # unflatten the linear layers of HF state_dict to match the shape of LlamaMlp
        d = {}
        d.update(
            unflatten_linear_layers(
                apply_prefix(prefix, "gate_proj"), state_dict, self.gate_proj, out_dims_first_in_dict=True
            )
        )
        d.update(
            unflatten_linear_layers(
                apply_prefix(prefix, "up_proj"), state_dict, self.up_proj, out_dims_first_in_dict=True
            )
        )
        d.update(
            unflatten_linear_layers(
                apply_prefix(prefix, "down_proj"), state_dict, self.down_proj, out_dims_first_in_dict=True
            )
        )

        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix=prefix)

        my_dict.update(
            flatten_linear_layers(apply_prefix(prefix, "gate_proj"), self.gate_proj, out_dims_first_in_dict=True)
        )
        my_dict.update(
            flatten_linear_layers(apply_prefix(prefix, "up_proj"), self.up_proj, out_dims_first_in_dict=True)
        )
        my_dict.update(
            flatten_linear_layers(apply_prefix(prefix, "down_proj"), self.down_proj, out_dims_first_in_dict=True)
        )

        state_dict.update(my_dict)
        return state_dict


class LlamaAttention(StateDictSerializationMixin, eqx.Module):
    config: SplitLlamaConfig = eqx.static_field()
    q_proj: hnn.Linear  # projection from Embed to query
    k_proj: hnn.Linear  # projection from Embed to key
    v_proj: hnn.Linear  # projection from Embed to value
    o_proj: hnn.Linear  # projection from Heads to output

    @staticmethod
    def init(config: SplitLlamaConfig, *, key) -> "LlamaAttention":
        use_bias = config.use_bias
        Embed = config.Embed
        QHeadsPerGroup = hax.Axis("q_heads_per_group", config.num_heads // config.num_kv_heads)

        k_q, k_k, k_v, k_o = jrandom.split(key, 4)
        q_proj = hnn.Linear.init(
            In=Embed, Out=(config.KVHeads, QHeadsPerGroup, config.HeadSize), key=k_q, use_bias=use_bias, out_first=True
        )
        k_proj = hnn.Linear.init(
            In=Embed, Out=(config.KVHeads, config.HeadSize), key=k_k, use_bias=use_bias, out_first=True
        )
        v_proj = hnn.Linear.init(
            In=Embed, Out=(config.KVHeads, config.HeadSize), key=k_v, use_bias=use_bias, out_first=True
        )
        o_proj = hnn.Linear.init(
            In=(config.Heads, config.HeadSize), Out=Embed, key=k_o, use_bias=use_bias, out_first=True
        )

        return LlamaAttention(config, q_proj, k_proj, v_proj, o_proj)

    def _rope_scale_factor(self) -> float:
        # hasattr for gemma and I'm feeling lazy
        if hasattr(self.config, "rope_scaling") and self.config.rope_scaling is not None:
            assert self.config.rope_scaling["type"] == "linear"
            return self.config.rope_scaling["factor"]
        return 1.0

    @named_call
    def __call__(self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None) -> NamedArray:
        key_q, key_k, key_v, key_o = maybe_rng_split(key, 4)

        # reorder heads and position for better training throughput
        q = self.q_proj(x, key=key_q).rearrange((..., "kv_heads", "q_heads_per_group", "position", "head_size"))
        k = self.k_proj(x, key=key_k).rearrange((..., "kv_heads", "position", "head_size"))
        v = self.v_proj(x, key=key_v).rearrange((..., "kv_heads", "position", "head_size"))

        cos, sin = llama_rotary_pos_emb(
            self.config.HeadSize,
            x.resolve_axis("position"),
            base=self.config.rope_base,
            scale=self._rope_scale_factor(),
        )
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        c = self.config
        attn_output = dot_product_attention(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask,
            attention_dtype=jnp.float32 if self.config.upcast_attn else x.dtype,
            use_flash=c.use_flash_attention,
            attn_backend=self.config.attn_backend,
            flash_block_size=c.flash_attention_block_size,
        )

        attn_output = attn_output.flatten_axes(("kv_heads", "q_heads_per_group"), "heads")
        attn_output = attn_output.astype(x.dtype)

        attn_output = self.o_proj(attn_output, key=key_o)
        return attn_output

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        # unflatten the linear layers of HF state_dict to match the shape of LlamaAttention
        d = {}
        d.update(unflatten_linear_layers(apply_prefix(prefix, "q_proj"), state_dict, self.q_proj, True))
        d.update(unflatten_linear_layers(apply_prefix(prefix, "k_proj"), state_dict, self.k_proj, True))
        d.update(unflatten_linear_layers(apply_prefix(prefix, "v_proj"), state_dict, self.v_proj, True))
        d.update(unflatten_linear_layers(apply_prefix(prefix, "o_proj"), state_dict, self.o_proj, True))

        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        # flatten the linear layers of LlamaAttention to match the shape of HF state_dict
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix)

        my_dict.update(flatten_linear_layers(apply_prefix(prefix, "q_proj"), self.q_proj, True))
        my_dict.update(flatten_linear_layers(apply_prefix(prefix, "k_proj"), self.k_proj, True))
        my_dict.update(flatten_linear_layers(apply_prefix(prefix, "v_proj"), self.v_proj, True))
        my_dict.update(flatten_linear_layers(apply_prefix(prefix, "o_proj"), self.o_proj, True))

        state_dict.update(my_dict)
        return state_dict


class LlamaRMSNorm(eqx.Module):
    """
    Similar to LayerNorm, but uses the RMS of the input along the specified axis (or axes) instead of variance.
    """

    axis: AxisSpec = eqx.static_field()
    weight: Optional[NamedArray]
    bias: Optional[NamedArray]

    eps: float = eqx.static_field(default=1e-5)
    dtype: Optional[jnp.dtype] = eqx.static_field(default=jnp.float32)

    @staticmethod
    def init(axis: AxisSpec, eps: float = 1e-6, use_weight: bool = True, use_bias: bool = True, dtype=jnp.float32):
        if use_weight:
            weight = hax.ones(axis)
        else:
            weight = None
        if use_bias:
            bias = hax.zeros(axis)
        else:
            bias = None

        return LlamaRMSNorm(axis, weight, bias, eps, dtype)

    def __call__(self, x: NamedArray) -> NamedArray:
        # This gives a different result than jnp.var(), which is
        # defined as the average of the squared deviations from the mean
        in_dtype = x.dtype
        x = x.astype(self.dtype)
        var = hax.mean(hax.square(x), axis=self.axis)
        inv = hax.rsqrt(var + self.eps)
        out = x * inv
        out = out.astype(in_dtype)

        if self.weight is not None:
            out = self.weight * out
        if self.bias is not None:
            out = out + self.bias

        # second cast in case params are in float32
        return out.astype(in_dtype)


class SplitDecoderWrapper(eqx.Module, StateDictSerializationMixin):
    config: SplitLlamaConfig = eqx.static_field()
    wrapped: "LlamaDecoderLayer"

    @staticmethod
    def init(wrapped: "LlamaDecoderLayer") -> "SplitDecoderWrapper":
        return SplitDecoderWrapper(wrapped.config, wrapped)

    @named_call
    def __call__(self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None) -> NamedArray:
        new_x_axis = Axis(self.config.Pos.name, self.config.skip_after_k_tokens)
        skip_x_axis = Axis(self.config.Pos.name, self.config.Pos.size - new_x_axis.size)
        x, holdout = x.split(self.config.Pos, (new_x_axis, skip_x_axis))
        x = self.wrapped(x, mask, key=key)
        return hax.concatenate(self.config.Pos, (x, holdout))


class LlamaDecoderLayer(StateDictSerializationMixin, eqx.Module):
    config: SplitLlamaConfig = eqx.static_field()
    self_attn: LlamaAttention
    mlp: LlamaMlp
    input_layernorm: LlamaRMSNorm
    post_attention_layernorm: LlamaRMSNorm

    @staticmethod
    def init(config: SplitLlamaConfig, *, key, layer_idx) -> "LlamaDecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)

        attn = LlamaAttention.init(config, key=k_attn)
        mlp = LlamaMlp.init(
            config.Embed,
            config.Mlp,
            config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )
        ln_1 = config.mk_LayerNorm(config.Embed)
        ln_2 = config.mk_LayerNorm(config.Embed)

        return LlamaDecoderLayer(config, attn, mlp, ln_1, ln_2)

    @named_call
    def __call__(self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)
        # self attention and skip connection
        residual = x
        x = self.input_layernorm(x)
        attn_output = self.self_attn(x=x, mask=mask, key=k_attn)
        x = residual + attn_output

        # MLP and skip connection
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x, key=k_mlp)
        output = residual + mlp_output
        return output


class LlamaTransformer(StateDictSerializationMixin, eqx.Module):
    config: SplitLlamaConfig = eqx.static_field()
    layers: hnn.BlockSeq[LlamaDecoderLayer]
    norm: LlamaRMSNorm

    @staticmethod
    def init(config: SplitLlamaConfig, *, key) -> "LlamaTransformer":
        layers = hnn.BlockSeq.init(
            config.Layers, LlamaDecoderLayer, gradient_checkpointing=config.gradient_checkpointing
        )(
            config,
            key=shaped_rng_split(key, config.num_layers),
            layer_idx=hax.arange(config.Layers),
        )
        ln_f = config.mk_LayerNorm(config.Embed)

        return LlamaTransformer(config, layers, ln_f)

    @named_call
    def __call__(self, x: NamedArray, attn_mask: Optional[NamedArray | AttentionMask], *, key) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.layers.fold(x, mask=attn_mask, key=keys)
        x = self.norm(x)

        return x

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        if isinstance(self.layers, Stacked):
            state_dict = stack_state_dict(state_dict, prefix=apply_prefix(prefix, "layers"))

        out = super().from_state_dict(state_dict, prefix=prefix)
        return out

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        my_state_dict: StateDict = {}
        super().update_state_dict(my_state_dict, prefix=prefix)

        if isinstance(self.layers, Stacked):
            stacked_dict = unstack_state_dict(my_state_dict, prefix=apply_prefix(prefix, "layers"))
            state_dict.update(stacked_dict)
        else:
            state_dict.update(my_state_dict)

        return state_dict


class LlamaEmbedding(StateDictSerializationMixin, eqx.Module):
    """Similar to GPT2 Embedding, except that:
    - Llama doesn't have position embedding in the Embedding layer.
    - Llama doesn't use dropout.
    """

    Vocab: Axis = eqx.static_field()
    token_embeddings: hnn.Embedding

    @staticmethod
    def init(Vocab: Axis, config: SplitLlamaConfig, *, key) -> "LlamaEmbedding":
        return LlamaEmbedding(Vocab, hnn.Embedding.init(Vocab, config.Embed, key=key))

    @named_call
    def embed(self, input_ids, *args):
        input_embeds = self.token_embeddings(input_ids)
        x = input_embeds
        return x

    def unembed(self, x: NamedArray):
        return self.token_embeddings.unembed(x)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"token_embeddings": "model.embed_tokens"}

    def resize_embeddings(self, new_size: int, key: Optional[PRNGKeyArray] = None):
        new_weights = self.token_embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, Vocab=self.Vocab.resize(new_size), token_embeddings=new_weights)


class LlamaLMHeadModel(eqx.Module, LmHeadModel[SplitLlamaConfig], StateDictSerializationMixin):
    transformer: LlamaTransformer
    embeddings: LlamaEmbedding
    lm_head: hnn.Linear

    @property
    def config(self):
        return self.transformer.config

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @classmethod
    def init(cls, Vocab: Axis, config: SplitLlamaConfig, *, key) -> "LlamaLMHeadModel":
        k_t, k_emb = jrandom.split(key, 2)
        transformer = LlamaTransformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)
        return LlamaLMHeadModel(transformer, embeddings, lm_head)

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[Union[NamedArray, AttentionMask]] = None,
        *,
        key=None,
    ) -> NamedArray:
        """
        Args:
            input_ids (NamedArray): [batch, position]
                Indices of input sequence tokens in the vocabulary.
            attn_mask (Union[NamedArray, AttentionMask], optional): [batch, position]
                Mask to avoid performing attention on the padding token indices of the encoder input.
                The attn_mask from training pipeline may be an AttentionMask object instead of NamedArray
        """
        k_t, k_head = maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask=attn_mask, key=k_t)
        lm_logits = self.lm_head(x, key=k_head)
        return lm_logits

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[SplitLlamaConfig]":
        new_Vocab = self.Vocab.resize(new_size)
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=k1)
        new_lm_matrix = hax.tree_util.resize_axis(self.lm_head.weight, self.Vocab, new_size, key=k2)
        new_lm_head = dataclasses.replace(self.lm_head, Out=new_Vocab, weight=new_lm_matrix)

        return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": "model", "embeddings": None}

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        # unflatten the linear layers of HF state_dict to match the shape of LlamaMlp
        d = state_dict.copy()
        d.update(
            unflatten_linear_layers(
                apply_prefix(prefix, "lm_head"), state_dict, self.lm_head, out_dims_first_in_dict=True
            )
        )
        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix=prefix)

        my_dict.update(
            flatten_linear_layers(apply_prefix(prefix, "lm_head"), self.lm_head, out_dims_first_in_dict=True)
        )

        state_dict.update(my_dict)
        return state_dict


def _rotate_half(x: NamedArray) -> NamedArray:
    """Rotates half of the hidden dims of the input and concatenates them."""
    HeadSize = x.axes[-1]
    x1 = x[HeadSize, : HeadSize.size // 2]
    x2 = x[HeadSize, HeadSize.size // 2 :]
    out = hax.concatenate(HeadSize, (-x2, x1))
    return out


def _apply_rotary_pos_emb(
    q: NamedArray,  # [batch, position, kv_heads, q_heads_per_group, head_size]
    k: NamedArray,  # [batch, position, kv_heads, head_size]
    cos: NamedArray,  # [position, head_size]
    sin: NamedArray,  # [position, head_size]
) -> Tuple[NamedArray, NamedArray]:
    """Applies rotary position embedding to q and k."""
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


def llama_rotary_pos_emb(
    HeadSize: Axis, Pos: Axis, base: float = 10000, scale: float = 1.0
) -> Tuple[NamedArray, NamedArray]:
    with jax.ensure_compile_time_eval():
        HeadHalfSize = HeadSize.resize(HeadSize.size // 2)
        inv_freq: NamedArray = 1.0 / (base ** (hax.arange(HeadHalfSize, step=2) / HeadSize.size))

        position_ids: NamedArray = hax.arange(Pos) / scale

        freqs = position_ids * inv_freq.broadcast_axis(Pos)
        # This is different from the paper but aligns with HF implementation:
        # It uses a different permutation in order to obtain the same calculation
        emb = hax.concatenate(HeadSize, (freqs, freqs))
        cos = hax.cos(emb)
        sin = hax.sin(emb)
        # This is different from the paper but aligns with HF implementation:
        return cos, sin
