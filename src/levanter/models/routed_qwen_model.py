import dataclasses
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Type, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from chex import PRNGKey
from jaxtyping import PyTree

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from haliax.quantization import DotGeneralOp
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.models.attention import AttentionMask, dot_product_attention
from levanter.models.gpt2 import ACT2FN
from levanter.models.llama import LlamaConfig, LlamaEmbedding, LlamaRMSNorm
from levanter.models.lm_model import Extras, LmConfig, LmHeadModel, RoutableLmExample
from levanter.models.rotary import RotaryEmbeddingsConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.jax_utils import key_iterator
from levanter.utils.logging import silence_transformer_nag
from levanter.utils.stat_utils import IndexCountHistogram, IndexCountUnique, LogitHistogram


silence_transformer_nag()
from transformers import PretrainedConfig as HfConfig  # noqa: E402
from transformers import Qwen2Config as HfQwenConfig  # noqa: E402


class ExpertType(Enum):
    """Model routing options"""

    LORA = "lora"
    MLP = "mlp"
    MLP_GLU = "mlp_glu"


class LowRankLinear(ModuleWithStateDictSerialization):
    """Low-rank linear layer for Lora"""

    lora_a: hnn.Linear
    lora_b: hnn.Linear
    scale: float = eqx.field(static=True)

    @staticmethod
    def init(
        In: AxisSpec,
        Inter: AxisSpec,
        Out: AxisSpec,
        *,
        key: PRNGKey,
        scale: float = 1.0,
        out_first: bool = True,
        dot_general: Optional[DotGeneralOp] = None,
    ) -> "LowRankLinear":
        k_a, k_b = jrandom.split(key, 2)
        kwargs = dict(
            out_first=out_first,
            dot_general=dot_general,
        )
        lora_a = hnn.Linear.init(In=In, Out=Inter, key=k_a, use_bias=False, init_scale=1.0, **kwargs)
        lora_b = hnn.Linear.init(In=Inter, Out=Out, key=k_b, use_bias=False, init_scale=0.0, **kwargs)
        return LowRankLinear(lora_a, lora_b, scale)

    @named_call
    def __call__(self, x: NamedArray, expert_mask: Optional[NamedArray], *, key=None) -> NamedArray:
        k_a, k_b = maybe_rng_split(key, 2)
        lora_a = self.lora_a(x, key=k_a)
        if expert_mask is not None:
            lora_a *= expert_mask
        lora_b = self.lora_b(lora_a, key=k_b)
        return lora_b * self.scale


def is_routed_experts_param(x):
    return isinstance(x, (RQwenMlpExperts, Router, LowRankLinear))


def routed_experts_trainable_params_filter(model: eqx.Module) -> Dict[str, jnp.ndarray]:
    return jax.tree_util.tree_map(is_routed_experts_param, model, is_leaf=is_routed_experts_param)


def routed_experts_mask(model_shape: PyTree) -> PyTree:
    return jax.tree.map(
        is_routed_experts_param,
        model_shape,
        is_leaf=lambda x: is_routed_experts_param(x) or isinstance(x, hax.NamedArray),
    )


def base_weights_mask(model_shape: PyTree) -> PyTree:
    return jax.tree.map(
        lambda x: not is_routed_experts_param(x),
        model_shape,
        is_leaf=lambda x: is_routed_experts_param(x) or isinstance(x, hax.NamedArray),
    )


def re_init_linear(x: hnn.Linear, init_scale=1.0, *, key):
    weight = init_scale * hax.random.normal(key, x.weight.axes, dtype=x.weight.dtype)
    return dataclasses.replace(x, weight=weight)


def reinit_expert_weights(config: "RQwenConfig", model: eqx.Module, *, key: jax.random.PRNGKey) -> eqx.Module:
    """Re-initialize all LoRA weights in the model while preserving other weights."""

    keys = key_iterator(key)
    init_scale = config.expert_init_scale
    init = config.expert_init

    def replace_fn(x: eqx.Module):
        if isinstance(x, LowRankLinear):
            lora_a = re_init_linear(
                x.lora_a, init_scale=0.0 if init == ExpertInit.LORA_ZERO_A else init_scale, key=next(keys)
            )
            lora_b = re_init_linear(
                x.lora_b, init_scale=0.0 if init == ExpertInit.LORA_ZERO_B else init_scale, key=next(keys)
            )
            return LowRankLinear(lora_a, lora_b, x.scale)
        elif isinstance(x, Router):
            return re_init_linear(x, init_scale=config.router_init_scale, key=key)
        elif isinstance(x, RQwenMlpExperts):
            gate_proj = None
            if x.gate_proj is not None:
                gate_proj = re_init_linear(
                    x.gate_proj, init_scale=0.0 if init == ExpertInit.MLP_ZERO_GATE else init_scale, key=next(keys)
                )
            up_proj = re_init_linear(
                x.up_proj, init_scale=0.0 if init == ExpertInit.MLP_ZERO_UP else init_scale, key=next(keys)
            )
            down_proj = re_init_linear(
                x.down_proj, init_scale=0.0 if init == ExpertInit.MLP_ZERO_DOWN else init_scale, key=next(keys)
            )
            return RQwenMlpExperts(gate_proj, up_proj, down_proj, x.act)
        else:
            return x

    def is_leaf(x):
        return isinstance(x, (LowRankLinear, Router, RQwenMlpExperts))

    return jax.tree.map(replace_fn, model, is_leaf=is_leaf)


class RLoraLinear(ModuleWithStateDictSerialization):
    """Linear layer with routing to Lora"""

    low_rank_linear: LowRankLinear
    linear: hnn.Linear

    @staticmethod
    def init(
        In: AxisSpec,
        Out: AxisSpec,
        Inter: AxisSpec,
        *,
        key: PRNGKey,
        scale: float = 1.0,
        use_bias: bool = True,
        out_first: bool = True,
        dot_general: Optional[DotGeneralOp] = None,
        init_scale: float = 1.0,
    ) -> "RLoraLinear":
        k_low_rank, k_linear = jrandom.split(key, 2)
        linear = hnn.Linear.init(
            In=In,
            Out=Out,
            key=k_linear,
            use_bias=use_bias,
            init_scale=init_scale,
            out_first=out_first,
            dot_general=dot_general,
        )
        low_rank_linear = LowRankLinear.init(
            In=In, Inter=Inter, Out=Out, scale=scale, key=k_low_rank, out_first=out_first, dot_general=dot_general
        )
        return RLoraLinear(low_rank_linear, linear)

    @named_call
    def __call__(self, x: NamedArray, expert_mask: Optional[NamedArray], *, key=None) -> NamedArray:
        k_linear, k_low_rank = maybe_rng_split(key, 2)
        linear_out = self.linear(x, key=k_linear)
        if expert_mask is not None:
            lora_out = self.low_rank_linear(x, expert_mask, key=k_low_rank)
            output = linear_out + lora_out
        else:
            output = linear_out
        return output

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"linear": None}

    def from_state_dict(self, state_dict, prefix=None):
        fsd = ModuleWithStateDictSerialization.from_state_dict
        if prefix is not None and prefix + ".low_rank_linear.lora_a" not in state_dict:
            print("Skipping lora load")
            low_rank_linear = self.low_rank_linear
            self = dataclasses.replace(self, low_rank_linear=None)
            self = fsd(self, state_dict, prefix)
            self = dataclasses.replace(self, low_rank_linear=low_rank_linear)
            return self
        return fsd(self, state_dict, prefix)


class Linear(eqx.Module):
    linear: hnn.Linear

    def __call__(self, x: NamedArray, expert_mask: Optional[NamedArray], *, key=None) -> NamedArray:
        return self.linear(x, key=key)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"linear": None}


MaybeRoutedLinear = Union[RLoraLinear, Linear]


def make_linear(
    config: "RQwenConfig",
    In: AxisSpec,
    Out: AxisSpec,
    Inter: AxisSpec,
    *,
    key: PRNGKey,
    scale: float = 1.0,
    use_bias: bool = True,
    out_first: bool = True,
    dot_general: Optional[DotGeneralOp] = None,
    init_scale: float = 1.0,
) -> MaybeRoutedLinear:
    if config.expert_type == ExpertType.LORA:
        return RLoraLinear.init(
            In=In,
            Out=Out,
            Inter=Inter,
            key=key,
            scale=scale,
            use_bias=use_bias,
            out_first=out_first,
            dot_general=dot_general,
            init_scale=init_scale,
        )
    else:
        return Linear(
            hnn.Linear.init(
                In=In,
                Out=Out,
                key=key,
                use_bias=use_bias,
                out_first=out_first,
                dot_general=dot_general,
                init_scale=init_scale,
            )
        )


class ExpertInit(Enum):
    """Expert initialization options"""

    LORA_ZERO_A = "lora_zero_a"
    LORA_ZERO_B = "lora_zero_b"
    MLP_ZERO_UP = "mlp_zero_up"
    MLP_ZERO_DOWN = "mlp_zero_down"
    MLP_ZERO_GATE = "mlp_zero_gate"
    NONZERO = "nonzero"


@LmConfig.register_subclass("rlora_qwen")
@dataclass(frozen=True)
class RQwenConfig(LlamaConfig):
    """Extends LlamaConfig with Qwen specific features"""

    use_sliding_window: bool = False
    sliding_window: Optional[int] = None
    max_window_layers: int = 0  # Only apply sliding window beyond this layer

    num_experts: int = 64
    expert_rank: int = 16
    top_k: int = 4
    disable_expert_mask: bool = False
    ident_expert_mask: bool = False
    scale: float = 1.0
    expert_type: ExpertType = ExpertType.LORA
    expert_init: ExpertInit = ExpertInit.LORA_ZERO_B
    expert_init_scale: float = 0.02
    router_init_scale: float = 0.02
    mult_by_topk: bool = False
    prefill_expert: bool = False
    route_each_layer: bool = False

    router_activation: str = "softmax"

    router_act_before_topk: bool = False
    expert_bias_update_rate: Optional[float] = None

    ExpertRank = property(lambda self: Axis("expert_rank", self.expert_rank))
    TopK = property(lambda self: Axis("top_k", self.top_k))

    @property
    def Experts(self) -> Axis:
        num_experts = self.num_experts
        if self.prefill_expert:
            num_experts += self.top_k  # Add prefill experts
        return Axis("experts", num_experts)

    @property
    def RouterOut(self) -> tuple[hax.Axis, ...]:
        return (self.Layers, self.Experts) if self.route_each_layer else (self.Experts,)

    def mask_axes(self, Batch: hax.Axis) -> tuple[hax.Axis, ...]:
        return (Batch, self.Pos) + self.RouterOut

    def __post_init__(self):
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), f"num_heads={self.num_heads} not divisible by num_kv_heads={self.num_kv_heads}."

        if self.expert_type == ExpertType.LORA:
            assert self.expert_init in [
                ExpertInit.LORA_ZERO_A,
                ExpertInit.LORA_ZERO_B,
                ExpertInit.NONZERO,
            ], self.expert_init
        elif self.expert_type == ExpertType.MLP:
            assert self.expert_init in [
                ExpertInit.MLP_ZERO_UP,
                ExpertInit.MLP_ZERO_DOWN,
                ExpertInit.NONZERO,
            ], self.expert_init
        elif self.expert_type == ExpertType.MLP_GLU:
            assert self.expert_init in [
                ExpertInit.MLP_ZERO_UP,
                ExpertInit.MLP_ZERO_DOWN,
                ExpertInit.MLP_ZERO_GATE,
                ExpertInit.NONZERO,
            ], self.expert_init

        if self.prefill_expert:
            assert (not self.ident_expert_mask) and (
                not self.disable_expert_mask
            ), "Can only use prefill experts when using the expert mask"

        assert self.router_activation in [
            "softmax",
            "sigmoid",
            "sigmoid_norm",
        ], f"Invalid router activation: {self.router_activation}"
        if self.expert_bias_update_rate is not None:
            assert self.router_act_before_topk, "Lossless expert bias update requires router_act_before_topk"

    def hf_checkpoint_converter(self) -> HFCheckpointConverter["RQwenConfig"]:  # type: ignore
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint,
            trust_remote_code=True,
            tokenizer=self.tokenizer if self.tokenizer else self.reference_checkpoint,
            HfConfigClass=HfQwenConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        rope_theta = hf_config.rope_theta
        rope_config = RotaryEmbeddingsConfig.from_hf_config(rope_theta, hf_config.rope_scaling)
        return RQwenConfig(
            seq_len=hf_config.max_position_embeddings,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            use_sliding_window=getattr(hf_config, "use_sliding_window", False),
            sliding_window=getattr(hf_config, "sliding_window", None),
            max_window_layers=getattr(hf_config, "max_window_layers", 0),
            activation_function=hf_config.hidden_act,
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            rope=rope_config,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfQwenConfig:
        if config_overrides is None:
            config_overrides = {}

        rope_theta, rope_scaling = self.rope.to_hf_config()

        return HfQwenConfig(
            max_position_embeddings=self.seq_len,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            use_sliding_window=self.use_sliding_window,
            sliding_window=self.sliding_window,
            max_window_layers=self.max_window_layers,
            hidden_act=self.activation_function,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.layer_norm_epsilon,
            tie_word_embeddings=self.tie_word_embeddings,
            vocab_size=vocab_size,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            **config_overrides,
        )

    @property
    def model_type(self) -> Type["RQwenLMHeadModel"]:
        return RQwenLMHeadModel

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


# Modified attention class for Qwen
class RQwenAttention(eqx.Module):
    config: RQwenConfig = eqx.static_field()
    q_proj: MaybeRoutedLinear
    k_proj: MaybeRoutedLinear
    v_proj: MaybeRoutedLinear
    o_proj: MaybeRoutedLinear

    @staticmethod
    def init(config: RQwenConfig, *, key) -> "RQwenAttention":
        Embed = config.Embed
        QHeadsPerGroup = hax.Axis("q_heads_per_group", config.num_heads // config.num_kv_heads)

        k_q, k_k, k_v, k_o = jrandom.split(key, 4)
        q_proj = make_linear(
            config,
            In=Embed,
            Out=(config.KVHeads, QHeadsPerGroup, config.HeadSize),
            Inter=(config.Experts, config.ExpertRank),
            scale=config.scale,
            key=k_q,
            use_bias=True,  # Qwen always uses bias in attention
            out_first=True,
        )
        k_proj = make_linear(
            config,
            In=Embed,
            Out=(config.KVHeads, config.HeadSize),
            Inter=(config.Experts, config.ExpertRank),
            scale=config.scale,
            key=k_k,
            use_bias=True,
            out_first=True,
        )
        v_proj = make_linear(
            config,
            In=Embed,
            Out=(config.KVHeads, config.HeadSize),
            Inter=(config.Experts, config.ExpertRank),
            scale=config.scale,
            key=k_v,
            use_bias=True,
            out_first=True,
        )
        o_proj = make_linear(
            config,
            In=(config.Heads, config.HeadSize),
            Out=Embed,
            Inter=(config.Experts, config.ExpertRank),
            scale=config.scale,
            key=k_o,
            use_bias=False,  # Qwen doesn't use bias in o_proj
            out_first=True,
        )
        return RQwenAttention(config, q_proj, k_proj, v_proj, o_proj)

    @named_call
    def __call__(
        self,
        x: NamedArray,
        mask: Optional[NamedArray | AttentionMask],
        expert_mask=Optional[NamedArray],
        layer_idx: int = 0,
        *,
        key=None,
    ) -> NamedArray:
        key_q, key_k, key_v, key_o = maybe_rng_split(key, 4)

        # QKV projections
        q = self.q_proj(x, expert_mask, key=key_q).rearrange(
            (..., "kv_heads", "q_heads_per_group", "position", "head_size")
        )
        k = self.k_proj(x, expert_mask, key=key_k).rearrange((..., "kv_heads", "position", "head_size"))
        v = self.v_proj(x, expert_mask, key=key_v).rearrange((..., "kv_heads", "position", "head_size"))

        # Apply rotary embeddings
        rot_embs = self.config.rope.build(self.config.HeadSize, q.resolve_axis("position"))
        q, k = rot_embs(self.config.HeadSize, q, k)

        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        # Apply sliding window attention if configured and past max_window_layers
        if (
            self.config.use_sliding_window
            and self.config.sliding_window is not None
            and layer_idx >= self.config.max_window_layers
        ):
            raise ValueError("Sliding Window Attention is not currently supported.")

        # Perform attention
        attn_output = dot_product_attention(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask,
            attention_dtype=jnp.float32 if self.config.upcast_attn else x.dtype,
            use_flash=self.config.use_flash_attention,
            attn_backend=self.config.attn_backend,
            flash_block_size=self.config.flash_attention_block_size,
        )

        attn_output = attn_output.flatten_axes(("kv_heads", "q_heads_per_group"), "heads")
        attn_output = attn_output.astype(x.dtype)

        attn_output = self.o_proj(attn_output, expert_mask, key=key_o)
        return attn_output


class RQwenMlpExperts(eqx.Module):
    gate_proj: Optional[hnn.Linear]
    up_proj: hnn.Linear
    down_proj: hnn.Linear
    act: Callable = eqx.static_field()

    @staticmethod
    def init(
        config: RQwenConfig,
        *,
        key,
    ) -> "RQwenMlpExperts":
        assert config.expert_type in [ExpertType.MLP, ExpertType.MLP_GLU]
        k_gate, k_up_proj, k_down_proj = jrandom.split(key, 3)
        Inter = (config.Experts, config.ExpertRank)
        gate_proj = None
        if config.expert_type == ExpertType.MLP_GLU:
            gate_proj = hnn.Linear.init(
                In=config.Embed,
                Out=Inter,
                key=k_gate,
                use_bias=False,
                out_first=True,
            )
        up_proj = hnn.Linear.init(
            In=config.Embed,
            Out=Inter,
            key=k_up_proj,
            use_bias=False,
            out_first=True,
        )
        down_proj = hnn.Linear.init(
            In=Inter,
            Out=config.Embed,
            key=k_down_proj,
            use_bias=False,
            out_first=True,
        )
        act = ACT2FN[config.activation_function]
        return RQwenMlpExperts(gate_proj, up_proj, down_proj, act)

    @named_call
    def __call__(self, x: NamedArray, expert_mask: NamedArray, *, key=None) -> NamedArray:
        k_gate, k_up, k_down = maybe_rng_split(key, 3)
        assert expert_mask is not None
        hidden_states = x
        hidden_states = self.act(self.up_proj(hidden_states, key=k_up))
        if self.gate_proj is not None:
            hidden_states *= self.gate_proj(hidden_states, key=k_gate)
        hidden_states *= expert_mask
        outputs = self.down_proj(hidden_states, key=k_down)
        return outputs


class RQwenMlp(ModuleWithStateDictSerialization):
    """Multi-layer Perceptron
    In comparison with GPT2, LlamaMlp adds an up-proj that multiplies with activated gate_proj,
    before down-proj.
    """

    gate_proj: MaybeRoutedLinear  # projection from Embed to Mlp
    up_proj: MaybeRoutedLinear  # projection from Embed to Mlp
    down_proj: MaybeRoutedLinear  # projection from Mlp to Embed
    experts: Optional[RQwenMlpExperts]
    act: Callable = eqx.static_field()

    @staticmethod
    def init(
        config: RQwenConfig,
        Embed: Axis,
        Mlp: Axis,
        Inter: AxisSpec,
        activation_fn: Union[str, Callable],
        *,
        key,
        scale: float = 1.0,
        use_bias: bool = False,
    ) -> "RLoraLinear":
        k_fc, k_up_proj, k_down_proj, k_mlp_exp = jrandom.split(key, 4)
        gate_proj = make_linear(
            config, Out=Mlp, In=Embed, Inter=Inter, scale=scale, key=k_fc, use_bias=use_bias, out_first=True
        )
        up_proj = make_linear(
            config, Out=Mlp, In=Embed, Inter=Inter, scale=scale, key=k_up_proj, use_bias=use_bias, out_first=True
        )
        down_proj = make_linear(
            config, Out=Embed, In=Mlp, Inter=Inter, scale=scale, key=k_down_proj, use_bias=use_bias, out_first=True
        )
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        act = activation_fn  # type: ignore
        experts = None
        if config.expert_type in [ExpertType.MLP, ExpertType.MLP_GLU]:
            experts = RQwenMlpExperts.init(config, key=k_mlp_exp)

        return RQwenMlp(gate_proj, up_proj, down_proj, experts, act)

    @named_call
    def __call__(self, x: NamedArray, expert_mask: Optional[NamedArray], *, key=None) -> NamedArray:
        k_gate, k_up, k_down = maybe_rng_split(key, 3)
        hidden_states = self.gate_proj(x, expert_mask, key=k_gate)
        hidden_states = self.act(hidden_states)
        hidden_states = hidden_states * self.up_proj(x, expert_mask, key=k_up)
        outputs = self.down_proj(hidden_states, expert_mask, key=k_down)
        if self.experts is not None and expert_mask is not None:
            outputs += self.experts(x, expert_mask, key=key)
        return outputs

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"expert_mlp": None}

    def from_state_dict(self, state_dict, prefix=None):
        fsd = ModuleWithStateDictSerialization.from_state_dict
        if self.experts is not None and prefix is not None and prefix + ".experts.up_proj" not in state_dict:
            print("Skipping expert_mlp load")
            experts = self.experts
            self = dataclasses.replace(self, experts=None)
            self = fsd(self, state_dict, prefix)
            self = dataclasses.replace(self, experts=experts)
            return self
        return fsd(self, state_dict, prefix)


# Modified decoder layer for Qwen
class RQwenDecoderLayer(eqx.Module):
    config: RQwenConfig = eqx.static_field()
    self_attn: RQwenAttention
    mlp: RQwenMlp
    input_layernorm: LlamaRMSNorm
    post_attention_layernorm: LlamaRMSNorm

    @staticmethod
    def init(config: RQwenConfig, *, key) -> "RQwenDecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)

        attn = RQwenAttention.init(config, key=k_attn)
        mlp = RQwenMlp.init(
            config,
            config.Embed,
            config.Mlp,
            (config.Experts, config.ExpertRank),
            config.activation_function,
            scale=config.scale,
            key=k_mlp,
            use_bias=config.use_bias,
        )
        ln_1 = config.mk_LayerNorm(config.Embed)
        ln_2 = config.mk_LayerNorm(config.Embed)

        return RQwenDecoderLayer(config, attn, mlp, ln_1, ln_2)

    @named_call
    def __call__(
        self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], expert_mask: Optional[NamedArray], *, key=None
    ) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)

        residual = x
        x = self.input_layernorm(x)
        attn_output = self.self_attn(x=x, mask=mask, expert_mask=expert_mask, key=k_attn)
        x = residual + attn_output

        residual = x
        x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x, expert_mask=expert_mask, key=k_mlp)
        output = residual + mlp_output
        return output


# Modified transformer for Qwen
class RQwenTransformer(eqx.Module):
    config: RQwenConfig = eqx.static_field()
    layers: Stacked[RQwenDecoderLayer]
    norm: LlamaRMSNorm

    @staticmethod
    def init(config: RQwenConfig, *, key) -> "RQwenTransformer":
        S = Stacked

        # Initialize layers with their indices
        layers = S.init(config.Layers, RQwenDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )

        ln_f = config.mk_LayerNorm(config.Embed)
        return RQwenTransformer(config, layers, ln_f)

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: Optional[NamedArray | AttentionMask], expert_mask: Optional[NamedArray], *, key
    ) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.layers.fold(x, mask=attn_mask, expert_mask=expert_mask, key=keys)
        x = self.norm(x)

        return x


# Modified LM head model for Qwen
class Router(hnn.Linear):
    @staticmethod
    def init(*args, **kwargs):
        linear = hnn.Linear.init(*args, **kwargs)
        return Router(linear.weight, linear.bias, linear.In, linear.Out, linear.dot_general)


class ExpertBiasTracker(eqx.Module):
    prev_bias: NamedArray
    """The bias used for the previous batch"""
    prev_load: NamedArray
    """The load on the previous batch"""

    def __add__(self, other: "ExpertBiasTracker") -> "ExpertBiasTracker":
        """This is aggregated by the microbatching logic, summing up the loads for the previous batch"""
        return ExpertBiasTracker(self.prev_bias, self.prev_load + other.prev_load)

    @staticmethod
    def zero(config: RQwenConfig):
        return ExpertBiasTracker(hax.zeros(config.Experts), hax.zeros(config.Experts))

    def curr_bias(self, config: RQwenConfig) -> NamedArray:
        """This computes the bias for the current batch based on the previous bias/load. XLA should optimize this."""
        assert config.expert_bias_update_rate is not None, "Lossless expert bias update requires a rate"
        mask = hax.ones(config.RouterOut, dtype=bool)
        if config.prefill_expert:
            mask = mask.at[config.Experts, : config.top_k].set(False)
        prev_avg_load = self.prev_load.mean(config.Experts, where=mask)
        update = hax.where(
            (self.prev_load > prev_avg_load) & mask, -config.expert_bias_update_rate, config.expert_bias_update_rate
        )
        return self.prev_bias + update


class RQwenLMHeadModel(LmHeadModel[RQwenConfig], ModuleWithStateDictSerialization):
    transformer: RQwenTransformer
    embeddings: LlamaEmbedding  # Can reuse Llama embeddings
    lm_head: Optional[hnn.Linear]
    router: Router

    @classmethod
    def init(cls, Vocab: Axis, config: RQwenConfig, *, key) -> "RQwenLMHeadModel":
        k_t, k_emb, k_rout = jrandom.split(key, 3)
        transformer = RQwenTransformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)

        router = Router.init(In=config.Embed, Out=config.RouterOut, key=k_rout, use_bias=False, out_first=True)

        return RQwenLMHeadModel(transformer, embeddings, lm_head, router)

    @property
    def config(self):
        return self.transformer.config

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[Union[NamedArray, AttentionMask]] = None,
        expert_mask: Optional[NamedArray] = None,
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
        x = self.transformer(x, attn_mask=attn_mask, expert_mask=expert_mask, key=k_t)
        if self.lm_head:
            lm_logits = self.lm_head(x, key=k_head)
        else:
            lm_logits = self.embeddings.unembed(x)
        return lm_logits

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        expert_mask: Optional[NamedArray] = None,
        *,
        key=None,
    ) -> NamedArray:
        """
        Compute the activations for the next token in a sequence.
        Args:
            input_ids: token IDs with shape {Pos}
            attn_mask: attention mask with shape {Pos, KeyPos}
            key: PRNGKey for random number generation

        Returns:
            NamedArray: activations with shape {Pos, Embed}

        """
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask=attn_mask, expert_mask=expert_mask, key=key)

        return x

    def router_activation(self, x: NamedArray, axis: hax.Axis) -> NamedArray:
        if self.config.router_activation == "softmax":
            return hax.nn.softmax(x, axis)
        elif self.config.router_activation == "sigmoid":
            return hax.nn.sigmoid(x)
        elif self.config.router_activation == "sigmoid_norm":
            sig = hax.nn.sigmoid(x)
            return sig / sig.sum(axis=axis).broadcast_to(sig.axes)
        else:
            raise ValueError(f"Invalid router activation: {self.config.router_activation}")

    def router_logits(
        self,
        Batch: hax.Axis,
        input_ids: NamedArray,
        router_hs_idxs: NamedArray,
        attn_mask: Optional[NamedArray],
        *,
        k_head=None,
        router_stop_grad: bool = True,
        k_rout=None,
    ) -> NamedArray:
        Experts, Pos, Embed, TopK = self.config.Experts, self.config.Pos, self.config.Embed, self.config.TopK
        compute_dtype = self.embeddings.token_embeddings.weight.dtype
        # Softmax, topk
        if Experts.size > 1:
            prefill_exp_mask = None
            if self.config.prefill_expert:
                assert not router_stop_grad, "Prefill expert requires router_stop_grad=False"
                prefill_exp_mask = hax.zeros(self.config.mask_axes(Batch), dtype=compute_dtype)
                prefill_exp_mask = prefill_exp_mask.at[Experts, : TopK.size].set(1.0)
            x = self.activations(input_ids, attn_mask=attn_mask, expert_mask=prefill_exp_mask, key=k_head)
            # Get the hidden states for the idxs we select
            router_inputs = fast_gather(Batch, Pos, Embed, x, router_hs_idxs)
            # Get the logits from the router
            if router_stop_grad:
                router_inputs = jax.lax.stop_gradient(router_inputs)
            router_logits = self.router(router_inputs, key=k_rout)
            router_logits = router_logits.astype(jnp.float32)
        else:
            router_logits = hax.zeros(self.config.mask_axes(Batch), dtype=jnp.float32)

        if self.config.prefill_expert:
            # The first top_k logical experts are just the prefill expert
            router_logits = router_logits.at[Experts, : TopK.size].set(-1e9)
        return router_logits

    def get_expert_mask(
        self,
        router_logits: hax.NamedArray,
        router_hs_idxs: hax.NamedArray,
        extras: Extras,
        compute_dtype: np.dtype,
        expert_bias: Optional[ExpertBiasTracker],
        example: RoutableLmExample,
    ) -> NamedArray | None:
        Batch = router_logits.resolve_axis("batch")
        Experts, TopK, Pos = self.config.Experts, self.config.TopK, self.config.Pos

        if self.config.disable_expert_mask:
            return None

        expert_mask: hax.NamedArray
        if self.config.ident_expert_mask:
            expert_mask = hax.ones(self.config.mask_axes(Batch), dtype=compute_dtype)
        if self.config.router_act_before_topk or self.config.top_k == 1:
            if self.config.top_k == 1 and not self.config.router_act_before_topk:
                warnings.warn("router_act_before_topk=False but top_k=1, doing router_act_before_topk anyways")
            router_acts = self.router_activation(router_logits, Experts)
            topk_inp = router_acts + expert_bias.curr_bias(self.config) if expert_bias is not None else router_acts
            _, top_k_indices = hax.top_k(topk_inp, Experts, TopK.size, TopK)
            expert_mask = create_expert_mask_from_acts(TopK, Experts, top_k_indices, router_acts.astype(compute_dtype))
        else:
            assert expert_bias is None, "Expert bias only supported with router_act_before_topk"
            elems, top_k_indices = hax.top_k(router_logits, Experts, TopK.size, TopK)
            elems = self.router_activation(elems, TopK)
            expert_mask = create_expert_mask(TopK, Experts, top_k_indices, elems.astype(compute_dtype))

        if self.config.mult_by_topk:
            expert_mask *= TopK.size

        if self.config.prefill_expert:
            # For prefill tokens, enable only the prefill expert
            only_exp0 = hax.zeros_like(expert_mask).at[Experts, : TopK.size].set(1.0)
            expert_mask = hax.where(router_hs_idxs < 0, only_exp0, expert_mask).astype(compute_dtype)
        else:
            expert_mask = hax.where(router_hs_idxs < 0, 0.0, expert_mask).astype(compute_dtype)

        assert example.completion_first_token_mask is not None, "Need completion_first_token_mask for expert mask"
        first_token_expert_mask = hax.where(
            example.completion_first_token_mask.broadcast_to(expert_mask.axes),
            expert_mask,
            0.0,
        )
        expert_mask_used = first_token_expert_mask > 0

        if self.config.route_each_layer:
            for i in range(self.config.Layers.size):
                extras.loggable[f"router/index_hist_{i:0>2}"] = IndexCountHistogram.init(
                    expert_mask_used[self.config.Layers, i].sum(axis=(Batch, Pos))
                )
                extras.loggable[f"router/used_count_{i:0>2}"] = IndexCountUnique.init(expert_mask_used, Experts)
        else:
            # TODO: can we do this by sequence rather than by token? I.e. adjust for the seq length?
            # Maybe what we want is a 'first token mask' that is 1 for the first token in the completion of sequence
            # and zero otherwise. That would get rid of a lot of hacking stuff around.
            extras.loggable["router/index_hist"] = IndexCountHistogram.init(expert_mask_used.sum(axis=(Batch, Pos)))
            extras.loggable["router/used_count"] = IndexCountUnique.init(expert_mask_used, Experts)

        if expert_bias is not None:
            per_expert_load = expert_mask_used.sum(axis=(Batch, Pos))
            # Put the new bias in, per_expert_load will get aggregated across microbatches
            extras.aux["expert_bias"] = ExpertBiasTracker(expert_bias.curr_bias(self.config), per_expert_load)

        return expert_mask

    def routed_forward(
        self,
        example: RoutableLmExample,
        *,
        key=None,
        router_stop_grad: bool = True,
        activations: bool = False,
        expert_bias: Optional[ExpertBiasTracker] = None,
    ) -> tuple[NamedArray, NamedArray, Extras]:
        k_head, k_rout = maybe_rng_split(key, 2)
        Batch = example.tokens.resolve_axis("batch")
        compute_dtype = self.embeddings.token_embeddings.weight.dtype
        extras: Extras = Extras()
        input_ids, attn_mask, router_hs_idxs = example.tokens, example.attn_mask, example.router_hs_idxs
        assert router_hs_idxs is not None

        # Softmax, topk
        router_logits = self.router_logits(
            Batch,
            input_ids,
            router_hs_idxs,
            attn_mask,
            k_head=k_head,
            router_stop_grad=router_stop_grad,
            k_rout=k_rout,
        )
        extras.loggable["router/logits"] = LogitHistogram.init(router_logits)

        expert_mask = self.get_expert_mask(router_logits, router_hs_idxs, extras, compute_dtype, expert_bias, example)

        if activations:
            res = self.activations(input_ids, attn_mask=attn_mask, key=k_head, expert_mask=expert_mask)
        else:
            res = self(input_ids, attn_mask=attn_mask, key=k_head, expert_mask=expert_mask)

        return (res, router_logits.astype(compute_dtype), extras)

    def get_lm_head(self) -> hax.NamedArray:
        if self.lm_head is None:
            return self.embeddings.token_embeddings.weight
        else:
            return self.lm_head.weight

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[LlamaConfig]":
        new_Vocab = self.Vocab.resize(new_size)
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=k1)
        if self.lm_head is not None:
            new_lm_matrix = hax.tree_util.resize_axis(self.lm_head.weight, self.Vocab, new_size, key=k2)
            new_lm_head = dataclasses.replace(self.lm_head, Out=new_Vocab, weight=new_lm_matrix)
            return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)
        else:
            return dataclasses.replace(self, embeddings=new_embeddings)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": "model", "embeddings": None}

    def from_state_dict(self, state_dict, prefix=None):
        fsd = ModuleWithStateDictSerialization.from_state_dict
        if "router.weight" not in state_dict:
            print("Skipping router init")
            router = self.router
            self = dataclasses.replace(self, router=None)
            self = fsd(self, state_dict, prefix)
            self = dataclasses.replace(self, router=router)
            return self
        return fsd(self, state_dict, prefix)


def fast_gather(
    Batch: hax.Axis,
    Pos: hax.Axis,
    Embed: hax.Axis,
    hidden_states: hax.NamedArray,
    indices: hax.NamedArray,
):
    def fast_gather_jax(hidden_states, indices):
        """
        Args:
            hidden_states: [batch, seq, embed]
            indices: [batch, seq] with values in [0, seq-1]
        Returns:
            selected: [batch, seq, embed]
        """
        # Create one-hot: [batch, seq, seq]
        one_hot = jax.nn.one_hot(indices, hidden_states.shape[1])

        # Matmul to select the right hidden states
        # [batch, seq, seq] @ [batch, seq, embed] -> [batch, seq, embed]
        return jnp.einsum("bst,bte->bse", one_hot, hidden_states)

    x = hidden_states.rearrange((Batch, Pos, Embed)).array
    inds = indices.rearrange((Batch, Pos)).array
    jres = fast_gather_jax(x, inds)
    return hax.named(jres, (Batch, Pos, Embed))


def create_expert_mask(
    TopK: hax.Axis,
    Experts: hax.Axis,
    indices: hax.NamedArray,
    values: hax.NamedArray,
):
    assert TopK in indices.axes, f"TopK must be in indices: {indices.axes}"
    # [..., TopK, Experts]
    one_hot = hax.nn.one_hot(indices, Experts, dtype=bool)
    # [..., TopK, 1]
    values_expanded = values.rearrange(indices.axes).array[..., None]
    # [..., TopK, Experts] where experts is one_hot
    one_hot = one_hot.rearrange(indices.axes + (Experts,)).array
    # [..., TopK, Experts]
    weighted = one_hot * values_expanded  # hax doesn't like broadcasting sometimes
    # [..., Experts] by summing out TopK
    return hax.NamedArray(weighted, indices.axes + (Experts,)).sum(TopK)


def create_expert_mask_from_acts(
    TopK: hax.Axis,
    Experts: hax.Axis,
    inds: hax.NamedArray,
    activations: hax.NamedArray,
):
    activation_mask = hax.nn.one_hot(inds, Experts, dtype=bool).sum(axis=TopK)
    return activations * activation_mask
