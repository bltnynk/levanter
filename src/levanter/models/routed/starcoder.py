import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.models.attention import AttentionMask, dot_product_attention
from levanter.models.gpt2 import ACT2FN
from levanter.models.llama import LlamaConfig, LlamaEmbedding
from levanter.models.lm_model import LmHeadModel
from levanter.models.rotary import RotaryEmbeddingsConfig
from levanter.models.routed.comon import (
    ExpertType,
    MaybeRoutedLinear,
    RLoraLinear,
    RoutableLmConfig,
    RoutableLmHeadModel,
    RoutedMlpExperts,
    Router,
    make_linear,
)
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag


silence_transformer_nag()
from transformers import PretrainedConfig as HfConfig  # noqa: E402
from transformers import Starcoder2Config as HfStarcoderConfig  # noqa: E402


@RoutableLmConfig.register_subclass("rstarcoder")
@dataclass(frozen=True)
class RStarcoderConfig(LlamaConfig, RoutableLmConfig):
    """Extends LlamaConfig with Starcoder specific features"""

    tie_word_embeddings: bool = False
    use_bias: bool = True

    def hf_checkpoint_converter(self) -> HFCheckpointConverter["RStarcoderConfig"]:  # type: ignore
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint,
            trust_remote_code=True,
            tokenizer=self.tokenizer if self.tokenizer else self.reference_checkpoint,
            HfConfigClass=HfStarcoderConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        rope_theta = hf_config.rope_theta
        rope_config = RotaryEmbeddingsConfig.from_hf_config(rope_theta, hf_config.rope_scaling)
        return RStarcoderConfig(
            seq_len=hf_config.max_position_embeddings,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            activation_function=hf_config.hidden_act,
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.norm_epsilon,
            tie_word_embeddings=False,
            rope=rope_config,
            use_bias=hf_config.use_bias,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfStarcoderConfig:
        if config_overrides is None:
            config_overrides = {}

        rope_theta, rope_scaling = self.rope.to_hf_config()

        return HfStarcoderConfig(
            max_position_embeddings=self.seq_len,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            hidden_act=self.activation_function,
            initializer_range=self.initializer_range,
            norm_epsilon=self.layer_norm_epsilon,
            tie_word_embeddings=self.tie_word_embeddings,
            vocab_size=vocab_size,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_bias=self.use_bias,
            **config_overrides,
        )

    @property
    def model_type(self) -> Type["RStarcoderLMHeadModel"]:
        return RStarcoderLMHeadModel

    def flops_per_token(self, vocab_size: int):
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            vocab_size=vocab_size,
            glu=False,
        )

    def __post_init__(self):
        assert self.expert_type != ExpertType.MLP_GLU, "MLP_GLU is not supported in Starcoder"
        assert (
            not self.tie_word_embeddings
        ), "Tied word embeddings are not supported in Starcodertests/test_routed_qwen.py"
        return super().__post_init__()


# Modified attention class for Starcoder
class RStarcoderAttention(eqx.Module):
    config: RStarcoderConfig = eqx.static_field()
    q_proj: MaybeRoutedLinear
    k_proj: MaybeRoutedLinear
    v_proj: MaybeRoutedLinear
    o_proj: MaybeRoutedLinear

    @staticmethod
    def init(config: RStarcoderConfig, *, key) -> "RStarcoderAttention":
        Embed = config.Embed
        QHeadsPerGroup = hax.Axis("q_heads_per_group", config.num_heads // config.num_kv_heads)
        use_bias = config.use_bias

        k_q, k_k, k_v, k_o = jrandom.split(key, 4)
        q_proj = make_linear(
            config,
            In=Embed,
            Out=(config.KVHeads, QHeadsPerGroup, config.HeadSize),
            Inter=(config.Experts, config.ExpertRank),
            scale=config.scale,
            key=k_q,
            use_bias=use_bias,  # Starcoder always uses bias in attention
            out_first=True,
        )
        k_proj = make_linear(
            config,
            In=Embed,
            Out=(config.KVHeads, config.HeadSize),
            Inter=(config.Experts, config.ExpertRank),
            scale=config.scale,
            key=k_k,
            use_bias=use_bias,
            out_first=True,
        )
        v_proj = make_linear(
            config,
            In=Embed,
            Out=(config.KVHeads, config.HeadSize),
            Inter=(config.Experts, config.ExpertRank),
            scale=config.scale,
            key=k_v,
            use_bias=use_bias,
            out_first=True,
        )
        o_proj = make_linear(
            config,
            In=(config.Heads, config.HeadSize),
            Out=Embed,
            Inter=(config.Experts, config.ExpertRank),
            scale=config.scale,
            key=k_o,
            use_bias=use_bias,  # Starcoder doesn't use bias in o_proj
            out_first=True,
        )
        return RStarcoderAttention(config, q_proj, k_proj, v_proj, o_proj)

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


class RStarcoderMlp(ModuleWithStateDictSerialization):
    """Multi-layer Perceptron
    In comparison with GPT2, LlamaMlp adds an up-proj that multiplies with activated gate_proj,
    before down-proj.
    """

    c_fc: MaybeRoutedLinear
    c_proj: MaybeRoutedLinear
    experts: Optional[RoutedMlpExperts]
    act: Callable = eqx.static_field()

    @staticmethod
    def init(
        config: RStarcoderConfig,
        Embed: Axis,
        Mlp: Axis,
        Inter: AxisSpec,
        activation_fn: Union[str, Callable],
        *,
        key,
        scale: float = 1.0,
        use_bias: bool = False,
    ) -> "RLoraLinear":
        k_fc, k_proj, k_mlp_exp = jrandom.split(key, 3)
        c_fc = make_linear(
            config, Out=Mlp, In=Embed, Inter=Inter, scale=scale, key=k_fc, use_bias=use_bias, out_first=True
        )
        c_proj = make_linear(
            config, Out=Embed, In=Mlp, Inter=Inter, scale=scale, key=k_proj, use_bias=use_bias, out_first=True
        )
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        act = activation_fn  # type: ignore
        experts = None
        if config.expert_type in [ExpertType.MLP, ExpertType.MLP_GLU]:
            experts = RoutedMlpExperts.init(config, key=k_mlp_exp)

        return RStarcoderMlp(c_fc, c_proj, experts, act)

    @named_call
    def __call__(self, x: NamedArray, expert_mask: Optional[NamedArray], *, key=None) -> NamedArray:
        k_gate, k_down = maybe_rng_split(key, 2)
        hidden_states = self.c_fc(x, expert_mask, key=k_gate)
        hidden_states = self.act(hidden_states)
        outputs = self.c_proj(hidden_states, expert_mask, key=k_down)
        if self.experts is not None and expert_mask is not None:
            outputs += self.experts(x, expert_mask, key=key)
        return outputs

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"expert_mlp": None}

    def from_state_dict(self, state_dict, prefix=None):
        fsd = ModuleWithStateDictSerialization.from_state_dict
        if self.experts is not None and prefix is not None and prefix + ".experts.c_fc" not in state_dict:
            print("Skipping expert_mlp load")
            experts = self.experts
            self = dataclasses.replace(self, experts=None)
            self = fsd(self, state_dict, prefix)
            self = dataclasses.replace(self, experts=experts)
            return self
        return fsd(self, state_dict, prefix)


# Modified decoder layer for Starcoder
class RStarcoderDecoderLayer(eqx.Module):
    config: RStarcoderConfig = eqx.static_field()
    self_attn: RStarcoderAttention
    mlp: RStarcoderMlp
    input_layernorm: hnn.LayerNorm
    post_attention_layernorm: hnn.LayerNorm

    @staticmethod
    def init(config: RStarcoderConfig, *, key) -> "RStarcoderDecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)

        attn = RStarcoderAttention.init(config, key=k_attn)
        mlp = RStarcoderMlp.init(
            config,
            config.Embed,
            config.Mlp,
            (config.Experts, config.ExpertRank),
            config.activation_function,
            scale=config.scale,
            key=k_mlp,
            use_bias=config.use_bias,
        )
        ln_1 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_weight=True, use_bias=True)
        ln_2 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_weight=True, use_bias=True)

        return RStarcoderDecoderLayer(config, attn, mlp, ln_1, ln_2)

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


# Modified transformer for Starcoder
class RStarcoderTransformer(eqx.Module):
    config: RStarcoderConfig = eqx.static_field()
    layers: Stacked[RStarcoderDecoderLayer]
    norm: hnn.LayerNorm

    @staticmethod
    def init(config: RStarcoderConfig, *, key) -> "RStarcoderTransformer":
        S = Stacked

        # Initialize layers with their indices
        layers = S.init(config.Layers, RStarcoderDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )

        ln_f = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_weight=True, use_bias=True)
        return RStarcoderTransformer(config, layers, ln_f)

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: Optional[NamedArray | AttentionMask], expert_mask: Optional[NamedArray], *, key
    ) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.layers.fold(x, mask=attn_mask, expert_mask=expert_mask, key=keys)
        x = self.norm(x)

        return x


class RStarcoderLMHeadModel(RoutableLmHeadModel, ModuleWithStateDictSerialization):
    router: Router
    transformer: RStarcoderTransformer
    embeddings: LlamaEmbedding  # Can reuse Llama embeddings
    lm_head: hnn.Linear

    @classmethod
    def init(cls, Vocab: Axis, config: RStarcoderConfig, *, key) -> "RStarcoderLMHeadModel":
        k_t, k_emb, k_rout = jrandom.split(key, 3)
        transformer = RStarcoderTransformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        assert not config.tie_word_embeddings
        lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)

        router = Router.init(In=config.Embed, Out=config.RouterOut, key=k_rout, use_bias=False, out_first=True)

        return RStarcoderLMHeadModel(router, transformer, embeddings, lm_head)

    def get_router(self) -> Router:
        return self.router

    @property
    def compute_dtype(self):
        return self.embeddings.token_embeddings.weight.dtype

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
