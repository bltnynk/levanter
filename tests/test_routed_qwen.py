import dataclasses
import tempfile

import equinox as eqx
import jax
import numpy as np
import pytest

import haliax as hax

from levanter.models.attention import AttentionMask
from levanter.models.routed_qwen_model import (
    ExpertInit,
    ExpertType,
    RLoraLinear,
    RQwenConfig,
    RQwenLMHeadModel,
    reinit_expert_weights,
    routed_experts_trainable_params_filter,
)
from test_utils import skip_if_no_torch


def test_routed_qwen_state_dict_keys():
    config = RQwenConfig(seq_len=512, num_layers=2, hidden_dim=256, intermediate_dim=512, tie_word_embeddings=True)
    Vocab = hax.Axis("vocab", 1000)
    key = jax.random.PRNGKey(0)
    model = RQwenLMHeadModel.init(Vocab, config, key=key)

    print(model.to_state_dict().keys())


def test_routed_qwen_forward():
    config = RQwenConfig(seq_len=512, num_layers=2, hidden_dim=256, intermediate_dim=512, tie_word_embeddings=True)
    Vocab = hax.Axis("vocab", 1000)
    key = jax.random.PRNGKey(0)
    model = RQwenLMHeadModel.init(Vocab, config, key=key)

    Batch = hax.Axis("batch", 32)
    x = hax.random.randint(key, (Batch, config.Pos), 0, Vocab.size)
    inds = hax.random.randint(key, (Batch, config.Pos), 0, config.Pos.size - 1)
    _ = model.routed_forward(Batch, x, inds)

    # test with num_experts=1
    config = dataclasses.replace(config, num_experts=1, top_k=1)
    model = RQwenLMHeadModel.init(Vocab, config, key=key)
    _ = model.routed_forward(Batch, x, inds)


@skip_if_no_torch
@pytest.mark.parametrize("expert_type", [t for t in ExpertType])
@pytest.mark.parametrize("expert_init", [t for t in ExpertInit])
def test_rqwen_consistent_with_base_qwen(expert_type, expert_init):
    import torch
    from transformers import Qwen2ForCausalLM

    inits = {
        ExpertType.LORA: [ExpertInit.LORA_ZERO_A, ExpertInit.LORA_ZERO_B],
        ExpertType.MLP: [ExpertInit.MLP_ZERO_DOWN, ExpertInit.MLP_ZERO_UP],
        ExpertType.MLP_GLU: [ExpertInit.MLP_ZERO_DOWN, ExpertInit.MLP_ZERO_UP, ExpertInit.MLP_ZERO_GATE],
    }

    if expert_init not in inits[expert_type]:
        pytest.skip(f"Expert type {expert_type} does not support expert init {expert_init}")

    converter = RQwenConfig().hf_checkpoint_converter()
    config = RQwenConfig(
        seq_len=128,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        hidden_dim=16,
        intermediate_dim=32,
        tie_word_embeddings=True,
        disable_expert_mask=True,  # disable expert mask to keep consistency
        expert_type=expert_type,
        expert_init=expert_init,
    )
    Vocab = hax.Axis("vocab", 1000)
    hf_config = config.to_hf_config(Vocab.size)

    Batch = hax.Axis("batch", 2)
    input = hax.random.randint(jax.random.PRNGKey(0), (Batch, config.Pos), 0, Vocab.size)
    seq_inds = hax.random.randint(jax.random.PRNGKey(0), (Batch, config.Pos), 0, config.Pos.size - 1)
    attn_mask = AttentionMask.causal()
    input_torch = torch.from_numpy(np.array(input.array)).to(torch.int32)

    torch.random.manual_seed(0)

    torch_model = Qwen2ForCausalLM(hf_config)
    torch_model.eval()

    torch_out = torch_model(input_torch)
    torch_out = torch_out.logits.detach().cpu().numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

        model = converter.load_pretrained(
            Qwen2ForCausalLM, ref=f"{tmpdir}/torch_model", config=config, resize_vocab_to_match_tokenizer=False
        )

        @hax.named_jit
        def compute(model: RQwenLMHeadModel, input):
            model_output = model.routed_forward(Batch, input, router_hs_idxs=seq_inds, attn_mask=attn_mask)
            return model_output

        token_pred, mask, extras = compute(model, input)
        jax_out = token_pred.array

        assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
        assert np.isclose(torch_out, np.array(jax_out), rtol=1e-4, atol=1e-4).all(), f"{torch_out} != {jax_out}"

        cfg_with_expert = dataclasses.replace(config, disable_expert_mask=False)
        model = dataclasses.replace(model, transformer=dataclasses.replace(model.transformer, config=cfg_with_expert))

        @hax.named_jit
        def compute(model: RQwenLMHeadModel, input):
            model_output = model.routed_forward(Batch, input, router_hs_idxs=seq_inds, attn_mask=attn_mask)
            return model_output

        token_pred, mask, extras = compute(model, input)
        jax_out = token_pred.array

        assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
        should_be_close = expert_init != ExpertInit.NONZERO
        assert (
            should_be_close == np.isclose(torch_out, np.array(jax_out), rtol=1e-4, atol=1e-4).all()
        ), f"{torch_out} == {jax_out} with lora mask"


def init_const_model(
    expert_type: ExpertType, expert_init: ExpertInit, const_val=5.0
) -> tuple[RQwenLMHeadModel, RQwenConfig]:
    config = RQwenConfig(
        seq_len=512,
        num_layers=2,
        hidden_dim=256,
        intermediate_dim=512,
        tie_word_embeddings=True,
        expert_type=expert_type,
        expert_init=expert_init,
    )
    model: RQwenLMHeadModel = RQwenLMHeadModel.init(hax.Axis("vocab", 100), config, key=jax.random.PRNGKey(0))
    const_model = jax.tree.map(
        lambda x: const_val * hax.ones_like(x), model, is_leaf=lambda x: isinstance(x, hax.NamedArray)
    )
    del model

    def check_weight(x):
        if isinstance(x, hax.NamedArray):
            assert hax.all(x == const_val).item()
        return x

    jax.tree.map(check_weight, const_model, is_leaf=lambda x: isinstance(x, hax.NamedArray))
    return const_model, config


def test_rqwen_reinit():
    def case(expert_type: ExpertType, expert_init: ExpertInit, zero_getter, nonzero_getter):
        model, config = init_const_model(expert_type, expert_init, const_val=5.0)

        model = reinit_expert_weights(config, model, key=jax.random.PRNGKey(0))

        stacked = model.transformer.layers.stacked
        for weight in zero_getter(stacked):
            assert hax.all(weight == 0.0).item(), "Weight not zero-initialized"
        for weight in nonzero_getter(stacked):
            assert not hax.all(weight == 0.0).item(), "Incorrect weight zero-initialized"

        assert not hax.all(model.router.weight == 0.0).item(), "Router weight re-initialized"

    case(
        ExpertType.MLP,
        ExpertInit.NONZERO,
        lambda x: (),
        lambda x: (x.mlp.experts.down_proj.weight, x.mlp.experts.up_proj.weight),
    )
    case(
        ExpertType.MLP_GLU,
        ExpertInit.NONZERO,
        lambda x: (),
        lambda x: (x.mlp.experts.down_proj.weight, x.mlp.experts.up_proj.weight, x.mlp.experts.gate_proj.weight),
    )
    case(
        ExpertType.LORA,
        ExpertInit.NONZERO,
        lambda x: (),
        lambda x: (x.mlp.down_proj.low_rank_linear.lora_a.weight, x.mlp.down_proj.low_rank_linear.lora_b.weight),
    )

    case(
        ExpertType.MLP,
        ExpertInit.MLP_ZERO_DOWN,
        lambda x: (x.mlp.experts.down_proj.weight,),
        lambda x: (x.mlp.experts.up_proj.weight,),
    )
    case(
        ExpertType.MLP,
        ExpertInit.MLP_ZERO_UP,
        lambda x: (x.mlp.experts.up_proj.weight,),
        lambda x: (x.mlp.experts.down_proj.weight,),
    )

    case(
        ExpertType.MLP_GLU,
        ExpertInit.MLP_ZERO_DOWN,
        lambda x: (x.mlp.experts.down_proj.weight,),
        lambda x: (x.mlp.experts.up_proj.weight, x.mlp.experts.gate_proj.weight),
    )
    case(
        ExpertType.MLP_GLU,
        ExpertInit.MLP_ZERO_UP,
        lambda x: (x.mlp.experts.up_proj.weight,),
        lambda x: (x.mlp.experts.down_proj.weight, x.mlp.experts.gate_proj.weight),
    )
    case(
        ExpertType.MLP_GLU,
        ExpertInit.MLP_ZERO_GATE,
        lambda x: (x.mlp.experts.gate_proj.weight,),
        lambda x: (x.mlp.experts.up_proj.weight, x.mlp.experts.down_proj.weight),
    )

    case(
        ExpertType.LORA,
        ExpertInit.LORA_ZERO_A,
        lambda x: (x.mlp.down_proj.low_rank_linear.lora_a.weight,),
        lambda x: (x.mlp.down_proj.low_rank_linear.lora_b.weight,),
    )
    case(
        ExpertType.LORA,
        ExpertInit.LORA_ZERO_B,
        lambda x: (x.mlp.down_proj.low_rank_linear.lora_b.weight,),
        lambda x: (x.mlp.down_proj.low_rank_linear.lora_a.weight,),
    )


@pytest.mark.parametrize("expert_type", [ExpertType.LORA, ExpertType.MLP, ExpertType.MLP_GLU])
def test_rqwen_trainable_filt(expert_type):
    config = RQwenConfig(
        seq_len=512,
        num_layers=2,
        hidden_dim=256,
        intermediate_dim=512,
        tie_word_embeddings=True,
        expert_type=expert_type,
        expert_init=ExpertInit.NONZERO,
    )

    def model_init():
        return config.build(hax.Axis("vocab", 100), key=jax.random.PRNGKey(0))

    model_shape = eqx.filter_eval_shape(model_init)
    is_trainable: RQwenLMHeadModel = routed_experts_trainable_params_filter(model_shape)
    tf = is_trainable.transformer.layers.stacked
    assert is_trainable.router is True
    if expert_type == ExpertType.LORA:
        assert tf.mlp.down_proj.low_rank_linear is True
        assert tf.mlp.up_proj.low_rank_linear is True
        assert tf.mlp.gate_proj.low_rank_linear is True
        assert tf.self_attn.k_proj.low_rank_linear is True
        assert tf.self_attn.q_proj.low_rank_linear is True
        assert tf.self_attn.v_proj.low_rank_linear is True
        assert tf.self_attn.o_proj.low_rank_linear is True
    elif expert_type == ExpertType.MLP:
        assert tf.mlp.experts is True
    elif expert_type == ExpertType.MLP_GLU:
        assert tf.mlp.experts is True


def test_rqwen_expert_type_init():
    config = RQwenConfig(
        seq_len=512,
        num_layers=2,
        hidden_dim=256,
        intermediate_dim=512,
        tie_word_embeddings=True,
        expert_type=ExpertType.LORA,
    )
    model: RQwenLMHeadModel = config.build(hax.Axis("vocab", 100), key=jax.random.PRNGKey(0))
    assert model.transformer.layers.stacked.mlp.experts is None
    assert isinstance(model.transformer.layers.stacked.mlp.down_proj, RLoraLinear)

    for expert_type in [ExpertType.MLP, ExpertType.MLP_GLU]:
        config = RQwenConfig(
            seq_len=512,
            num_layers=2,
            hidden_dim=256,
            intermediate_dim=512,
            tie_word_embeddings=True,
            expert_type=expert_type,
            expert_init=ExpertInit.NONZERO,
        )
        model: RQwenLMHeadModel = config.build(hax.Axis("vocab", 100), key=jax.random.PRNGKey(0))
        assert not isinstance(model.transformer.layers.stacked.mlp.down_proj, RLoraLinear)
        assert model.transformer.layers.stacked.mlp.experts is not None
