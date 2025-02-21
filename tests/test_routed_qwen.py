import dataclasses
import tempfile
from contextlib import ExitStack
from typing import List

import equinox as eqx
import jax
import numpy as np
import pytest
from jax.sharding import Mesh

import haliax as hax
from haliax.partitioning import ResourceAxis

from levanter.models.attention import AttentionMask
from levanter.models.lm_model import RoutableLmExample
from levanter.routed_models.qwen import RQwenConfig, RQwenLMHeadModel
from levanter.routed_models.routed import (
    ExpertBiasTracker,
    ExpertInit,
    ExpertType,
    RLoraLinear,
    base_weights_mask,
    create_expert_mask,
    create_expert_mask_from_acts,
    reinit_expert_weights,
    routed_experts_mask,
    routed_experts_trainable_params_filter,
)
from levanter.utils.stat_utils import IndexCountHistogram, IndexCountUnique
from levanter.utils.types import Extras
from test_utils import skip_if_no_torch


def test_routed_qwen_state_dict_keys():
    config = RQwenConfig(seq_len=512, num_layers=2, hidden_dim=256, intermediate_dim=512, tie_word_embeddings=True)
    Vocab = hax.Axis("vocab", 1000)
    key = jax.random.PRNGKey(0)
    model = RQwenLMHeadModel.init(Vocab, config, key=key)

    print(model.to_state_dict().keys())


def get_mesh(Batch: hax.Axis):
    stack = ExitStack()
    mesh = Mesh((jax.devices()), (ResourceAxis.DATA,))
    stack.enter_context(mesh)
    stack.enter_context(hax.axis_mapping({Batch.name: ResourceAxis.DATA}))
    return stack


def test_routed_qwen_forward():
    Batch = hax.Axis("batch", 32)
    with get_mesh(Batch):
        config = RQwenConfig(seq_len=512, num_layers=2, hidden_dim=256, intermediate_dim=512, tie_word_embeddings=True)
        Vocab = hax.Axis("vocab", 1000)
        key = jax.random.PRNGKey(0)
        model = RQwenLMHeadModel.init(Vocab, config, key=key)

        x = hax.random.randint(key, (Batch, config.Pos), 0, Vocab.size)
        inds = hax.random.randint(key, (Batch, config.Pos), 0, config.Pos.size - 1)
        first_mask = hax.random.randint(key, (Batch, config.Pos), 0, 1).astype(bool)
        example = RoutableLmExample(x, None, router_hs_idxs=inds, completion_first_token_mask=first_mask)
        _ = model.routed_forward(example)

        # test with num_experts=1
        config = dataclasses.replace(config, num_experts=1, top_k=1)
        model = RQwenLMHeadModel.init(Vocab, config, key=key)
        _ = model.routed_forward(example)


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
    first_mask = hax.random.randint(jax.random.PRNGKey(0), (Batch, config.Pos), 0, 1).astype(bool)
    example = RoutableLmExample(
        input, None, attn_mask=attn_mask, router_hs_idxs=seq_inds, completion_first_token_mask=first_mask
    )
    input_torch = torch.from_numpy(np.array(input.array)).to(torch.int32)

    torch.random.manual_seed(0)

    torch_model = Qwen2ForCausalLM(hf_config)
    torch_model.eval()

    torch_out = torch_model(input_torch)
    torch_out = torch_out.logits.detach().cpu().numpy()

    with get_mesh(Batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            torch_model.save_pretrained(f"{tmpdir}/torch_model")

            model = converter.load_pretrained(
                Qwen2ForCausalLM, ref=f"{tmpdir}/torch_model", config=config, resize_vocab_to_match_tokenizer=False
            )

            @hax.named_jit
            def compute(model: RQwenLMHeadModel, example: RoutableLmExample):
                model_output = model.routed_forward(example)
                return model_output

            token_pred, mask, _, extras = compute(model, example)
            jax_out = token_pred.array

            assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
            assert np.isclose(torch_out, np.array(jax_out), rtol=1e-2, atol=1e-4).all(), f"{torch_out} != {jax_out}"

            cfg_with_expert = dataclasses.replace(config, disable_expert_mask=False)
            model = dataclasses.replace(
                model, transformer=dataclasses.replace(model.transformer, config=cfg_with_expert)
            )

            @hax.named_jit
            def compute(model: RQwenLMHeadModel, example: RoutableLmExample):
                model_output = model.routed_forward(example)
                return model_output

            token_pred, mask, _, extras = compute(model, example)
            jax_out = token_pred.array

            assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
            should_be_close = expert_init != ExpertInit.NONZERO
            assert (
                should_be_close == np.isclose(torch_out, np.array(jax_out), rtol=1e-2, atol=1e-4).all()
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


@pytest.mark.parametrize("with_layers", [True, False])
def test_create_expert_mask(with_layers):
    Batch = hax.Axis("batch", 2)
    TopK = hax.Axis("topk", 3)
    Expert = hax.Axis("experts", 4)
    Pos = hax.Axis("pos", 5)
    Layer = hax.Axis("layers", 6)
    MaskShape = (Batch, Pos, Expert)
    if with_layers:
        MaskShape = (Batch, Pos, Layer, Expert)
    activations = hax.random.uniform(jax.random.PRNGKey(0), MaskShape)
    elems, inds = hax.top_k(activations, Expert, k=TopK.size, new_axis=TopK)

    def check_mask(mask):
        def get(x, *args):
            return x.__getitem__(args)

        def check_idx(*idxs):
            bp_inds: List[int] = get(inds, *idxs).tolist()
            for e in range(Expert.size):
                if e in bp_inds:
                    idx = bp_inds.index(e)
                    val = get(elems, *idxs, TopK, idx)
                    assert get(mask, *idxs, Expert, e) == val
                else:
                    assert get(mask, *idxs, Expert, e) == 0.0

        for b in range(Batch.size):
            for p in range(Pos.size):
                if with_layers:
                    for li in range(Layer.size):
                        check_idx(Batch, b, Pos, p, Layer, li)
                else:
                    check_idx(Batch, b, Pos, p)

    mask1 = create_expert_mask(TopK, Expert, inds, elems)
    check_mask(mask1)

    mask2 = create_expert_mask_from_acts(TopK, Expert, inds, activations)
    check_mask(mask2)


@pytest.mark.parametrize("with_expert_bias", [True, False])
def test_expert_mask_creation(with_expert_bias):
    config = RQwenConfig(
        seq_len=64,
        num_layers=2,
        hidden_dim=16,
        intermediate_dim=32,
        num_heads=2,
        num_kv_heads=2,
        tie_word_embeddings=True,
        expert_type=ExpertType.MLP_GLU,
        expert_init=ExpertInit.NONZERO,  # Experts change output
        expert_rank=1,
        num_experts=8,
        top_k=2,
        router_act_before_topk=True,
        # Test with extremely high update rate
        expert_bias_update_rate=1e6 if with_expert_bias else None,
    )

    Vocab = hax.Axis("vocab", 100)
    model: RQwenLMHeadModel = config.build(Vocab, key=jax.random.PRNGKey(0))
    Batch, Pos = hax.Axis("batch", 1), config.Pos
    tokens = hax.random.randint(jax.random.PRNGKey(0), (Batch, Pos), 0, Vocab.size)
    seq_start = 16
    hs_idxs = -hax.ones_like(tokens)
    hs_idxs = hs_idxs.at[Batch, 0, Pos, seq_start:].set(seq_start - 1)
    first_token_mask = hax.zeros_like(tokens).at[Batch, 0, Pos, seq_start].set(1)
    completion_mask = hax.zeros_like(tokens).at[Batch, 0, Pos, seq_start:].set(1)
    example = RoutableLmExample(
        tokens,
        hax.ones_like(tokens),
        router_hs_idxs=hs_idxs,
        completion_mask=completion_mask,
        completion_first_token_mask=first_token_mask,
    )

    extras = Extras()
    expert_bias = None
    if with_expert_bias:
        prev_bias = hax.zeros(config.Experts)
        # Load all but the last top_k. Update bias is high so should select only the last two
        prev_load = hax.zeros(config.Experts).at[config.Experts, : -config.top_k].set(1.0)
        expert_bias = ExpertBiasTracker(prev_bias, prev_load)
        extras.loggable["expert_bias"] = expert_bias
    router_logits = model.router_logits(Batch, tokens, hs_idxs, example.attn_mask)
    router_acts = model.router_activation(router_logits, config.Experts).astype(np.float16)
    mask = model.get_expert_mask(router_logits, hs_idxs, extras, np.float16, expert_bias, example)
    index_hist: IndexCountHistogram = extras.loggable["router/index_hist"]
    used_count: IndexCountUnique = extras.loggable["router/used_count"]
    assert index_hist.hist.bucket_counts.sum().item() == config.top_k  # Only routes one sequence
    assert used_count.item() == config.top_k  # Only routes one sequence
    assert ((mask == 0.0) | (mask == router_acts)).all()
    if with_expert_bias:
        assert (mask[config.Experts, : -config.top_k] == 0.0).all(), "Only the last top_k should be selected"
        new_expert_bias: ExpertBiasTracker = extras.aux["expert_bias"]
        assert (new_expert_bias.prev_bias == expert_bias.curr_bias(config)).all()
        load = hax.zeros(config.Experts).at[config.Experts, -config.top_k :].set(1.0)
        assert (new_expert_bias.prev_load == load).all()


def test_weight_masks():
    config = RQwenConfig(
        seq_len=512,
        num_layers=2,
        hidden_dim=256,
        intermediate_dim=512,
        tie_word_embeddings=True,
        expert_type=ExpertType.MLP_GLU,
        expert_init=ExpertInit.NONZERO,
    )

    def model_init():
        return config.build(hax.Axis("vocab", 100), key=jax.random.PRNGKey(0))

    model_shape = eqx.filter_eval_shape(model_init)

    experts_mask: RQwenLMHeadModel = routed_experts_mask(model_shape)

    def manual_expert_values(mask: RQwenLMHeadModel):
        stacked = mask.transformer.layers.stacked
        tvals = [mask.router, stacked.mlp.experts]
        fvals = [
            stacked.mlp.gate_proj.linear.weight,
            stacked.mlp.up_proj.linear.weight,
            stacked.mlp.down_proj.linear.weight,
            stacked.self_attn.k_proj.linear.weight,
            stacked.self_attn.q_proj.linear.weight,
            stacked.self_attn.v_proj.linear.weight,
            stacked.self_attn.o_proj.linear.weight,
        ]
        return tvals, fvals

    exp_vals, base_vals = manual_expert_values(experts_mask)
    assert all([t is True for t in exp_vals])
    assert all([f is False for f in base_vals])

    base_mask: RQwenLMHeadModel = base_weights_mask(model_shape)
    exp_vals, base_vals = manual_expert_values(base_mask)
    assert all([t is False for t in exp_vals])
    assert all([f is True for f in base_vals])
