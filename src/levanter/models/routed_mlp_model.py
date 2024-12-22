import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from chex import PRNGKey

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
from levanter.models.lm_model import LmConfig, LmHeadModel, LmExample, RoutableLmExample
from levanter.models.rotary import RotaryEmbeddingsConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag
from levanter.models.loss import maybe_fused_next_token_loss
from levanter.utils.jax_utils import key_iterator


silence_transformer_nag()
from transformers import PretrainedConfig as HfConfig  # noqa: E402
from transformers import Qwen2Config as HfQwenConfig  # noqa: E402

import math


class RoutedLowRankLayer(ModuleWithStateDictSerialization):
    """Similar to LowRankLinear, with a non-linearity between two low-rank matrices mlp_a and mlp_b
    Also contains a set of experts, each of which is a pair of (column, row) that could be routed to."""

    mlp_a: hnn.Linear
    mlp_b: hnn.Linear
    expert_cols: hnn.Linear
    expert_rows: hnn.Linear
    scale: float = eqx.field(static=True)
    act: Callable = eqx.static_field()

    @staticmethod
    def init(
        In: AxisSpec,
        Inter: AxisSpec,
        Expert: AxisSpec,
        Out: AxisSpec,
        activation_fn: Union[str, Callable],
        *,
        key: PRNGKey,
        scale: float = 1.0,
        out_first: bool = True,
        dot_general: Optional[DotGeneralOp] = None,
    ) -> "RoutedLowRankLayer":
        k_a, k_b, k_cols, k_rows = jrandom.split(key, 4)
        kwargs = dict(
            out_first=out_first,
            dot_general=dot_general,
        )
        mlp_a = hnn.Linear.init(In=In, Out=Inter, key=k_a, use_bias=False, init_scale=1.0, **kwargs)
        mlp_b = hnn.Linear.init(In=Inter, Out=Out, key=k_b, use_bias=False, init_scale=0.0, **kwargs)
        expert_cols = hnn.Linear.init(In=In, Out=Expert, key=k_cols, use_bias=False, init_scale=1.0, **kwargs)
        expert_rows = hnn.Linear.init(In=Expert, Out=Out, key=k_rows, use_bias=False, init_scale=0.0, **kwargs)
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        act = activation_fn
        return RoutedLowRankLayer(mlp_a, mlp_b, expert_cols, expert_rows, scale, act)

    @named_call
    def __call__(self, x: NamedArray, expert_mask: Optional[NamedArray], *, key=None) -> NamedArray:
        z_mlp = self.mlp_a(x)
        z_mlp = self.act(z_mlp)
        z_mlp = self.mlp_b(z_mlp)
        
        z_routed = self.expert_cols(x)
        z_routed = self.act(z_routed)
        if expert_mask is not None:
            z_routed *= expert_mask
        z_routed = self.expert_rows(z_routed)

        return z_mlp * self.scale + z_routed

def routed_mlp_trainable_params_filter(model: eqx.Module) -> Dict[str, jnp.ndarray]:
    def is_routed_mlp_param(x):
        return isinstance(x, RoutedLowRankLayer) or isinstance(x, Router)

    return jax.tree_util.tree_map(is_routed_mlp_param, model, is_leaf=is_routed_mlp_param)


class RoutedMlpLayer(ModuleWithStateDictSerialization):
    """Layer with a base Linear and a low-rank MLP with routing to column-row experts"""

    routed_low_rank_layer: RoutedLowRankLayer
    linear: hnn.Linear

    @staticmethod
    def init(
        In: AxisSpec,
        Out: AxisSpec,
        Inter: AxisSpec,
        Expert: AxisSpec,
        *,
        key: PRNGKey,
        scale: float = 1.0,
        use_bias: bool = True,
        out_first: bool = True,
        dot_general: Optional[DotGeneralOp] = None,
        init_scale: float = 1.0,
    ) -> "RoutedMlpLayer":
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
        routed_low_rank_layer = RoutedMlpLayer.init(
            In=In, Inter=Inter, Expert=Expert, Out=Out, scale=scale, key=k_low_rank, out_first=out_first, dot_general=dot_general
        )
        return RoutedMlpLayer(routed_low_rank_layer, linear)

    @named_call
    def __call__(self, x: NamedArray, expert_mask: Optional[NamedArray], *, key=None) -> NamedArray:
        k_linear, k_low_rank = maybe_rng_split(key, 2)
        linear_out = self.linear(x, key=k_linear)
        if expert_mask is not None:
            routed_mlp_out = self.routed_low_rank_layer(x, expert_mask, key=k_low_rank)
            output = linear_out + routed_mlp_out
        else:
            output = linear_out
        return output

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"linear": None}

    def from_state_dict(self, state_dict, prefix=None):
        fsd = ModuleWithStateDictSerialization.from_state_dict
        if prefix is not None and prefix + ".routed_low_rank_layer.mlp_a" not in state_dict:
            print("Skipping lora load")
            routed_low_rank_layer = self.routed_low_rank_layer
            self = dataclasses.replace(self, routed_low_rank_layer=None)
            self = fsd(self, state_dict, prefix)
            self = dataclasses.replace(self, routed_low_rank_layer=routed_low_rank_layer)
            return self
        return fsd(self, state_dict, prefix)


@LmConfig.register_subclass("routed_qwen")
@dataclass(frozen=True)
class RoutedQwenConfig(LlamaConfig):
    """Extends LlamaConfig with Qwen specific features"""

    use_sliding_window: bool = False
    sliding_window: Optional[int] = None
    max_window_layers: int = 0  # Only apply sliding window beyond this layer

    num_experts: int = 64
    lora_rank: int = 16
    top_k: int = 4
    disable_expert_mask: bool = False
    ident_expert_mask: bool = False
    scale: float = 1.0

    Expert = property(lambda self: Axis("expert", self.num_experts))
    LoraRank = property(lambda self: Axis("lora_rank", self.lora_rank))

    def __post_init__(self):
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), f"num_heads={self.num_heads} not divisible by num_kv_heads={self.num_kv_heads}."

    def hf_checkpoint_converter(self) -> HFCheckpointConverter["RoutedQwenConfig"]:  # type: ignore
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
        return RoutedQwenConfig(
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
    def model_type(self) -> Type["RoutedQwenLMHeadModel"]:
        return RoutedQwenLMHeadModel

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
class RoutedQwenAttention(eqx.Module):
    config: RoutedQwenConfig = eqx.static_field()
    q_proj: RoutedMlpLayer
    k_proj: RoutedMlpLayer
    v_proj: RoutedMlpLayer
    o_proj: RoutedMlpLayer

    @staticmethod
    def init(config: RoutedQwenConfig, *, key) -> "RoutedQwenAttention":
        Embed = config.Embed
        QHeadsPerGroup = hax.Axis("q_heads_per_group", config.num_heads // config.num_kv_heads)

        k_q, k_k, k_v, k_o = jrandom.split(key, 4)
        q_proj = RoutedMlpLayer.init(
            In=Embed,
            Out=(config.KVHeads, QHeadsPerGroup, config.HeadSize),
            Inter=config.LoraRank,
            Expert=config.Expert,
            scale=config.scale,
            key=k_q,
            use_bias=True,  # Qwen always uses bias in attention
            out_first=True,
        )
        k_proj = RoutedMlpLayer.init(
            In=Embed,
            Out=(config.KVHeads, config.HeadSize),
            Inter=config.LoraRank,
            Expert=config.Expert,
            scale=config.scale,
            key=k_k,
            use_bias=True,
            out_first=True,
        )
        v_proj = RoutedMlpLayer.init(
            In=Embed,
            Out=(config.KVHeads, config.HeadSize),
            Inter=config.LoraRank,
            Expert=config.Expert,
            scale=config.scale,
            key=k_v,
            use_bias=True,
            out_first=True,
        )
        o_proj = RoutedMlpLayer.init(
            In=(config.Heads, config.HeadSize),
            Out=Embed,
            Inter=config.LoraRank,
            Expert=config.Expert,
            scale=config.scale,
            key=k_o,
            use_bias=False,  # Qwen doesn't use bias in o_proj
            out_first=True,
        )
        return RoutedQwenAttention(config, q_proj, k_proj, v_proj, o_proj)

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


class RoutedQwenMlp(eqx.Module):
    """Multi-layer Perceptron
    In comparison with GPT2, LlamaMlp adds an up-proj that multiplies with activated gate_proj,
    before down-proj.
    """

    gate_proj: RoutedMlpLayer  # projection from Embed to Mlp
    up_proj: RoutedMlpLayer  # projection from Embed to Mlp
    down_proj: RoutedMlpLayer  # projection from Mlp to Embed
    act: Callable = eqx.static_field()

    @staticmethod
    def init(
        Embed: Axis,
        Mlp: Axis,
        Inter: AxisSpec,
        Expert: AxisSpec,
        activation_fn: Union[str, Callable],
        *,
        key,
        scale: float = 1.0,
        use_bias: bool = False,
    ) -> "RoutedMlpLayer":
        k_fc, k_up_proj, k_down_proj = jrandom.split(key, 3)
        gate_proj = RoutedMlpLayer.init(
            Out=Mlp, In=Embed, Inter=Inter, Expert=Expert, scale=scale, key=k_fc, use_bias=use_bias, out_first=True
        )
        up_proj = RoutedMlpLayer.init(
            Out=Mlp, In=Embed, Inter=Inter, Expert=Expert, scale=scale, key=k_up_proj, use_bias=use_bias, out_first=True
        )
        down_proj = RoutedMlpLayer.init(
            Out=Embed, In=Mlp, Inter=Inter, Expert=Expert, scale=scale, key=k_down_proj, use_bias=use_bias, out_first=True
        )
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        act = activation_fn  # type: ignore
        return RoutedQwenMlp(gate_proj, up_proj, down_proj, act)

    @named_call
    def __call__(self, x: NamedArray, expert_mask: Optional[NamedArray], *, key=None) -> NamedArray:
        k_gate, k_up, k_down = maybe_rng_split(key, 3)
        hidden_states = self.gate_proj(x, expert_mask, key=k_gate)
        hidden_states = self.act(hidden_states)
        hidden_states = hidden_states * self.up_proj(x, expert_mask, key=k_up)
        outputs = self.down_proj(hidden_states, expert_mask, key=k_down)
        return outputs


# Modified decoder layer for Qwen
class RoutedQwenDecoderLayer(eqx.Module):
    config: RoutedQwenConfig = eqx.static_field()
    self_attn: RoutedQwenAttention
    mlp: RoutedQwenMlp  # Can reuse Llama MLP as structure is similar
    input_layernorm: LlamaRMSNorm
    post_attention_layernorm: LlamaRMSNorm

    @staticmethod
    def init(config: RoutedQwenConfig, *, key) -> "RoutedQwenDecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)

        attn = RoutedQwenAttention.init(config, key=k_attn)
        mlp = RoutedQwenMlp.init(
            config.Embed,
            config.Mlp,
            config.LoraRank,
            config.Expert,
            config.activation_function,
            scale=config.scale,
            key=k_mlp,
            use_bias=config.use_bias,
        )
        ln_1 = config.mk_LayerNorm(config.Embed)
        ln_2 = config.mk_LayerNorm(config.Embed)

        return RoutedQwenDecoderLayer(config, attn, mlp, ln_1, ln_2)

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
class RoutedQwenTransformer(eqx.Module):
    config: RoutedQwenConfig = eqx.static_field()
    layers: Stacked[RoutedQwenDecoderLayer]
    norm: LlamaRMSNorm

    @staticmethod
    def init(config: RoutedQwenConfig, *, key) -> "RoutedQwenTransformer":
        S = Stacked

        # Initialize layers with their indices
        layers = S.init(config.Layers, RoutedQwenDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )

        ln_f = config.mk_LayerNorm(config.Embed)
        return RoutedQwenTransformer(config, layers, ln_f)

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


class RoutedQwenLMHeadModel(LmHeadModel[RoutedQwenConfig], ModuleWithStateDictSerialization):
    transformer: RoutedQwenTransformer
    embeddings: LlamaEmbedding  # Can reuse Llama embeddings
    lm_head: Optional[hnn.Linear]
    router: Router

    @classmethod
    def init(cls, Vocab: Axis, config: RoutedQwenConfig, *, key) -> "RoutedQwenLMHeadModel":
        k_t, k_emb, k_rout = jrandom.split(key, 3)
        transformer = RoutedQwenTransformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)

        router = Router.init(In=config.Embed, Out=config.Expert, key=k_rout, use_bias=False, out_first=True)

        return RoutedQwenLMHeadModel(transformer, embeddings, lm_head, router)

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

    def routed_forward(
        self,
        Batch: hax.Axis,
        input_ids: NamedArray,
        router_hs_idxs: NamedArray,
        attn_mask: Optional[NamedArray] = None,
        *,
        key=None,
        router_stop_grad: bool = True,
        activations: bool = False,
    ) -> tuple[NamedArray, NamedArray, dict]:
        k_head, k_rout = maybe_rng_split(key, 2)
        Expert, Pos = self.config.Expert, self.config.Pos
        TopK = hax.Axis("top_k", self.config.top_k)
        compute_dtype = self.embeddings.token_embeddings.weight.dtype

        # Softmax, topk
        if Expert.size > 1:
            x = self.activations(input_ids, attn_mask=attn_mask, key=k_head)
            # Get the hidden states for the idxs we select
            router_inputs = x.take(Pos, router_hs_idxs)
            # Get the logits from the router
            if router_stop_grad:
                router_inputs = jax.lax.stop_gradient(router_inputs)
            router_logits = self.router(router_inputs, key=k_rout)
            router_logits = router_logits.astype(jnp.float32)
        else:
            router_logits = hax.zeros((Batch, Expert), dtype=jnp.float32)

        sm = hax.nn.softmax(router_logits, Expert)
        ent = hax.sum(-sm * hax.log2(sm), Expert)
        mean_ent = hax.mean(ent, Batch)
        std_ent = hax.std(ent, Batch)

        batch_sm = sm.mean(Batch)
        batch_ent = hax.sum(-batch_sm * hax.log2(batch_sm), Expert)

        elems, inds = hax.top_k(sm, Expert, TopK.size, TopK)
        # Create a mask
        expert_mask = hax.zeros((Batch, Expert), dtype=compute_dtype)
        # Arrange batch indicies for a .at
        batch_indices = hax.arange(Batch).broadcast_to((Batch, TopK))
        expert_mask = expert_mask.array.at[batch_indices.array, inds.array].set(elems.array.astype(compute_dtype))
        expert_mask = hax.NamedArray(expert_mask, [Batch, Expert])

        # Broadcast the mask to the almost full shape
        expert_mask = expert_mask.broadcast_to([Batch, Pos, Expert])
        seq_mask = hax.arange(Pos).broadcast_to((Batch, Pos)) > router_hs_idxs.broadcast_to((Batch, Pos))
        seq_mask = seq_mask.broadcast_to([Batch, Pos, Expert])
        expert_mask = expert_mask * seq_mask

        if self.config.disable_expert_mask:
            expert_mask = None
        elif self.config.ident_expert_mask:
            expert_mask = hax.ones((Batch, Pos, Expert), dtype=compute_dtype)
        if activations:
            res = self.activations(input_ids, attn_mask=attn_mask, expert_mask=expert_mask, key=k_head)
        else:
            res = self(input_ids, attn_mask=attn_mask, expert_mask=expert_mask, key=k_head)
        return (
            res,
            router_logits,
            {"router/mean_ent": mean_ent, "router/std_ent": std_ent, "router/batch_ent": batch_ent},
        )

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

def compute_next_token_loss_with_routing(
    model: RoutedQwenLMHeadModel,
    example: LmExample,
    *,
    key=None,
    reduction: Optional[hax.ReductionFunction] = hax.mean,
    reduction_axis: Optional[hax.AxisSelection] = None,
    batch_num_elements: Optional[int] = None,
    batch_completion_num_elements: Optional[int] = None,
    logsumexp_weight: Optional[float] = None,
    loss_dtype: Optional[Type[jnp.dtype]] = jnp.float32,
    router_zloss_weight: float = 0.0,
    stop_grad: bool = True,
) -> tuple[jnp.ndarray | NamedArray, dict]:
    """
    Computes the cross-entropy loss for a language modeling example. If reduction is not None, the loss is reduced
    across the reduction axis (with reduction_axis=None meaning all axes). If reduction is None, the loss is not
    reduced, and the result is a named array with axes (*batch axes, sequence_length).
    """
    assert isinstance(example, RoutableLmExample)
    # This is problematic, we don't get correctly batched ones so...
    idxs = jnp.squeeze(example.router_hs_idxs, axis=1)
    batch_axis = example.tokens.resolve_axis("batch")
    idxs = hax.NamedArray(idxs, (batch_axis,))
    example = dataclasses.replace(example, router_hs_idxs=idxs)
    activations, rlogits, extras = model.routed_forward(
        batch_axis,
        example.tokens,
        example.router_hs_idxs,
        example.attn_mask,
        key=key,
        activations=True,
        router_stop_grad=stop_grad,
    )

    loss = maybe_fused_next_token_loss(
        model.Pos,
        model.Embed,
        model.Vocab,
        activations,
        model.get_lm_head(),
        example.tokens,
        loss_mask=example.loss_mask,
        batch_num_elements=batch_num_elements,
        reduction=reduction,
        reduction_axis=reduction_axis,
        logsumexp_weight=logsumexp_weight,
        dtype=loss_dtype,
        block_size=model.config.cross_entropy_block_size,
    )

    completion_loss = maybe_fused_next_token_loss(
        model.Pos,
        model.Embed,
        model.Vocab,
        activations,
        model.get_lm_head(),
        example.tokens,
        loss_mask=example.completion_mask,  # only looking at completion
        reduction=reduction,
        reduction_axis=reduction_axis,
        batch_num_elements=batch_completion_num_elements,
        logsumexp_weight=logsumexp_weight,
        dtype=loss_dtype,
        block_size=model.config.cross_entropy_block_size,
    )

    extras["all_lm_loss"] = loss
    extras["lm_loss"] = completion_loss
    if router_zloss_weight > 0.0:
        z_loss = hax.nn.logsumexp(rlogits, model.config.Loras)
        z_loss = hax.mean(hax.square(z_loss), batch_axis)
        loss += router_zloss_weight * z_loss
        extras["router/z_loss"] = z_loss

    return loss, extras

def reinit_routed_mlp_weights(model: eqx.Module, *, key: jax.random.PRNGKey) -> eqx.Module:
    """Re-initialize all LoRA weights in the model while preserving other weights."""

    def where(m: RoutedQwenLMHeadModel):
        return [
            m.transformer.layers.stacked.mlp.down_proj.routed_low_rank_layer,
            m.transformer.layers.stacked.mlp.up_proj.routed_low_rank_layer,
            m.transformer.layers.stacked.mlp.gate_proj.routed_low_rank_layer,
            m.transformer.layers.stacked.self_attn.q_proj.routed_low_rank_layer,
            m.transformer.layers.stacked.self_attn.k_proj.routed_low_rank_layer,
            m.transformer.layers.stacked.self_attn.v_proj.routed_low_rank_layer,
            m.transformer.layers.stacked.self_attn.o_proj.routed_low_rank_layer,
            m.router,
        ]

    def re_init_linear(x: hnn.Linear, init_scale=1.0, *, key):
        input_size = hax.axis_size(x.In)
        weight = hax.random.truncated_normal(key, x.weight.axes, -3, 3) * (init_scale / math.sqrt(input_size))
        return dataclasses.replace(x, weight=weight)

    keys = key_iterator(key)

    def replace_fn(x: eqx.Module):
        if isinstance(x, RoutedLowRankLayer):
            mlp_a = re_init_linear(x.mlp_a, init_scale=1.0, key=next(keys))
            mlp_b = re_init_linear(x.mlp_b, init_scale=0.0, key=next(keys))
            return RoutedLowRankLayer(mlp_a, mlp_b, x.scale)
        elif isinstance(x, Router):
            return re_init_linear(x, init_scale=1.0, key=next(keys))
        else:
            return x

    return eqx.tree_at(where, model, replace_fn=replace_fn)
