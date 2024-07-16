import os
import tempfile

import jax
import jmp
import pytest

import haliax as hax
from haliax import Axis

import tiny_test_corpus
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LMDatasetConfig
from levanter.distributed import RayConfig
from levanter.main import split_lora_lm
from levanter.models import llama_splitgen as lsg
from levanter.models.attention import AttentionMask
from levanter.tracker.wandb import WandbConfig


def small_cfg(hf_test=True):
    if hf_test:
        cfg = lsg.SplitLlamaConfig(
            seq_len=256,
            hidden_dim=16,
            intermediate_dim=64,
            num_heads=4,
            num_layers=2,
            num_kv_heads=4,
            skip_indices=[1],
            attn_backend="jax_flash",
        )
        hf_url = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        return cfg, hf_url
    else:
        cfg = lsg.SplitLlamaConfig(
            seq_len=64,
            hidden_dim=16,
            intermediate_dim=64,
            num_heads=4,
            num_layers=32,
            num_kv_heads=4,
            skip_indices=[23, 25, 27, 29, 31],
            attn_backend="jax_flash",
        )
        return cfg, None


def test_loraize():
    Vocab = Axis("Vocab", 32000)
    cfg, _ = small_cfg(False)
    key = jax.random.PRNGKey(0)
    model = lsg.LlamaLMHeadModel.init(Vocab, cfg, key=key)

    model_lora = cfg.loraize(model, key=key)
    unstacked = model_lora.transformer.layers.unstacked()
    for layer_idx in range(cfg.Layers.size):
        if layer_idx not in cfg.skip_indices:
            sa = unstacked[layer_idx].self_attn
            for proj in [sa.q_proj, sa.k_proj, sa.v_proj]:
                assert isinstance(proj, lsg.SplitLoraLinear)


def _get_random_inputs(config: lsg.SplitLlamaConfig, Vocab: Axis):
    Batch = hax.Axis("batch", 2)
    x = hax.random.randint(jax.random.PRNGKey(0), (Batch, config.Pos), 0, Vocab.size)
    split_mask = hax.random.randint(jax.random.PRNGKey(1), (Batch, config.Pos), 0, 1).astype(bool)
    mask = AttentionMask.causal()
    return x, split_mask, mask


def test_splitgen_forward():
    Vocab = Axis("Vocab", 32000)
    cfg, _ = small_cfg(False)
    key = jax.random.PRNGKey(0)
    model = lsg.LlamaLMHeadModel.init(Vocab, cfg, key=key)
    inputs = _get_random_inputs(cfg, Vocab)

    model_lora = cfg.loraize(model, key=key)
    out_lora = model_lora.custom_fwd(*inputs).array
    out_refr = model.custom_fwd(*inputs).array
    assert jax.numpy.allclose(out_lora, out_refr, atol=1e-5).item()

    out = model_lora.custom_fwd(*inputs).array

    @hax.named_jit
    def jitted_fwd(x, split_mask, mask):
        return model_lora.custom_fwd(x, split_mask, mask)

    out2 = jitted_fwd(*inputs).array

    abs_diff = jax.numpy.abs(out - out2).max()
    assert jax.numpy.allclose(out, out2, atol=1e-5).item(), f"abs_diff: {abs_diff}"


@pytest.mark.entry
@pytest.mark.parametrize("hf_test", [True, False])
def test_train_splitgen_lm(hf_test):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_test_corpus.construct_small_data_cache(tmpdir)
        model_cfg, hf_url = small_cfg(hf_test)
        print(f"Have {jax.device_count()} devices")
        try:
            config = split_lora_lm.TrainLmConfig(
                initialize_from_hf=hf_url,
                skip_after_k_tokens=32,
                data=data_config,
                model=model_cfg,
                trainer=split_lora_lm.TrainerConfig(
                    mp=jmp.get_policy("p=f32,c=bfloat16"),
                    checkpointer=CheckpointerConfig(
                        base_path=tmpdir,
                    ),
                    num_train_steps=10,
                    train_batch_size=2 * len(jax.devices()),
                    max_eval_batches=2,
                    steps_per_eval=100,
                    wandb=WandbConfig(mode="disabled"),
                    require_accelerator=False,
                    ray=RayConfig(auto_start_cluster=False),
                    fsdp_axis="embed",
                    per_device_parallelism=1,
                ),
            )
            split_lora_lm.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass


@pytest.mark.entry
@pytest.mark.parametrize("hf_test", [True])  # False])
def test_train_splitgen_hfdata(hf_test):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config = LMDatasetConfig(
            id="stas/c4-en-10k",
            tokenizer="meta-llama/Llama-2-7b-hf",
            cache_dir=f"{tmpdir}/data_dir",
            stream=False,
        )
        model_cfg, hf_url = small_cfg(hf_test)
        print(f"Have {jax.device_count()} devices")
        try:
            config = split_lora_lm.TrainLmConfig(
                initialize_from_hf=hf_url,
                data=data_config,
                model=model_cfg,
                trainer=split_lora_lm.TrainerConfig(
                    mp=jmp.get_policy("p=f32,c=bfloat16"),
                    checkpointer=CheckpointerConfig(
                        base_path=tmpdir,
                    ),
                    num_train_steps=10,
                    train_batch_size=2 * len(jax.devices()),
                    max_eval_batches=2,
                    steps_per_eval=100,
                    wandb=WandbConfig(mode="disabled"),
                    require_accelerator=False,
                    ray=RayConfig(auto_start_cluster=False),
                    fsdp_axis="embed",
                    per_device_parallelism=1,
                ),
            )
            split_lora_lm.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass
