import abc
import dataclasses
import warnings
from enum import Enum
from typing import Callable, Dict, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call
from haliax.quantization import DotGeneralOp
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.models.attention import AttentionMask
from levanter.models.gpt2 import ACT2FN
from levanter.models.lm_model import RoutableLmExample
from levanter.utils.jax_utils import key_iterator
from levanter.utils.stat_utils import IndexCountHistogram, IndexCountUnique, LogitHistogram
from levanter.utils.types import Extras


class ExpertInit(Enum):
    """Expert initialization options"""

    LORA_ZERO_A = "lora_zero_a"
    LORA_ZERO_B = "lora_zero_b"
    MLP_ZERO_UP = "mlp_zero_up"
    MLP_ZERO_DOWN = "mlp_zero_down"
    MLP_ZERO_GATE = "mlp_zero_gate"
    NONZERO = "nonzero"


class ExpertType(Enum):
    """Model routing options"""

    LORA = "lora"
    MLP = "mlp"
    MLP_GLU = "mlp_glu"


class RoutableLmConfigMixin(abc.ABC):
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

    @abc.abstractmethod
    @property
    def Layers(self) -> Axis:
        pass

    @property
    @abc.abstractmethod
    def Pos(self) -> Axis:
        pass

    @property
    @abc.abstractmethod
    def Embed(self) -> Axis:
        pass

    @property
    @abc.abstractmethod
    def activation_function(self) -> str:
        pass

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


class RoutableLmModel(abc.ABC, eqx.Module):
    router: "Router"

    @property
    @abc.abstractmethod
    def config(self) -> RoutableLmConfigMixin:
        pass

    @property
    @abc.abstractmethod
    def compute_dtype(self) -> jnp.dtype:
        pass

    @abc.abstractmethod
    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        expert_mask: Optional[NamedArray] = None,
        *,
        key=None,
    ) -> NamedArray:
        pass

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
        # Softmax, topk
        if Experts.size > 1:
            prefill_exp_mask = None
            if self.config.prefill_expert:
                assert not router_stop_grad, "Prefill expert requires router_stop_grad=False"
                prefill_exp_mask = hax.zeros(self.config.mask_axes(Batch), dtype=self.compute_dtype)
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
        expert_bias: Optional["ExpertBiasTracker"],
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
        expert_bias: Optional["ExpertBiasTracker"] = None,
    ) -> tuple[NamedArray, NamedArray, NamedArray | None, Extras]:
        k_head, k_rout = maybe_rng_split(key, 2)
        Batch = example.tokens.resolve_axis("batch")
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

        expert_mask = self.get_expert_mask(
            router_logits, router_hs_idxs, extras, self.compute_dtype, expert_bias, example
        )

        if activations:
            res = self.activations(input_ids, attn_mask=attn_mask, key=k_head, expert_mask=expert_mask)
        else:
            res = self(input_ids, attn_mask=attn_mask, key=k_head, expert_mask=expert_mask)

        return (res, router_logits.astype(self.compute_dtype), expert_mask, extras)


# Modified LM head model for Qwen
class Router(hax.nn.Linear):
    @staticmethod
    def init(*args, **kwargs):
        linear = hax.nn.Linear.init(*args, **kwargs)
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
    def zero(config: RoutableLmConfigMixin):
        return ExpertBiasTracker(hax.zeros(config.RouterOut), hax.zeros(config.RouterOut))

    def curr_bias(self, config: RoutableLmConfigMixin) -> NamedArray:
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


class RoutedMlpExperts(eqx.Module):
    gate_proj: Optional[hnn.Linear]
    up_proj: hnn.Linear
    down_proj: hnn.Linear
    act: Callable = eqx.static_field()

    @staticmethod
    def init(
        config: RoutableLmConfigMixin,
        *,
        key,
    ) -> "RoutedMlpExperts":
        assert config.expert_type in [ExpertType.MLP, ExpertType.MLP_GLU]
        k_gate, k_up_proj, k_down_proj = jax.random.split(key, 3)
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
        return RoutedMlpExperts(gate_proj, up_proj, down_proj, act)

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
        key: jax.random.PRNGKey,
        scale: float = 1.0,
        out_first: bool = True,
        dot_general: Optional[DotGeneralOp] = None,
    ) -> "LowRankLinear":
        k_a, k_b = jax.random.split(key, 2)
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
        key: jax.random.PRNGKey,
        scale: float = 1.0,
        use_bias: bool = True,
        out_first: bool = True,
        dot_general: Optional[DotGeneralOp] = None,
        init_scale: float = 1.0,
    ) -> "RLoraLinear":
        k_low_rank, k_linear = jax.random.split(key, 2)
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


def is_routed_experts_param(x):
    return isinstance(x, (RoutedMlpExperts, Router, LowRankLinear))


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


def reinit_expert_weights(config: RoutableLmConfigMixin, model: eqx.Module, *, key: jax.random.PRNGKey) -> eqx.Module:
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
        elif isinstance(x, RoutedMlpExperts):
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
            return RoutedMlpExperts(gate_proj, up_proj, down_proj, x.act)
        else:
            return x

    def is_leaf(x):
        return isinstance(x, (LowRankLinear, Router, RoutedMlpExperts))

    return jax.tree.map(replace_fn, model, is_leaf=is_leaf)


def make_linear(
    config: RoutableLmConfigMixin,
    In: AxisSpec,
    Out: AxisSpec,
    Inter: AxisSpec,
    *,
    key: jax.random.PRNGKey,
    scale: float = 1.0,
    use_bias: bool = True,
    out_first: bool = True,
    dot_general: Optional[DotGeneralOp] = None,
    init_scale: float = 1.0,
) -> "MaybeRoutedLinear":
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
