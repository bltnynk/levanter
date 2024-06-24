import os
import tempfile

import jax
import pytest

import haliax as hax
from haliax import Axis

import tiny_test_corpus
from levanter.distributed import RayConfig
from levanter.main import split_lora_lm
from levanter.models import llama_splitgen as lsg
from levanter.models.attention import AttentionMask
from levanter.tracker.wandb import WandbConfig


# {
#   "_name_or_path": "HuggingFaceM4/tiny-random-LlamaForCausalLM",
#   "architectures": [
#     "LlamaForCausalLM"
#   ],
#   "bos_token_id": 0,
#   "eos_token_id": 1,
#   "hidden_act": "silu",
#   "hidden_size": 16,
#   "initializer_range": 0.02,
#   "intermediate_size": 64,
#   "model_type": "llama",
#   "num_attention_heads": 4,
#   "num_hidden_layers": 2,
#   "pad_token_id": -1,
#   "rms_norm_eps": 1e-06,
#   "tie_word_embeddings": false,
#   "torch_dtype": "float32",
#   "transformers_version": "4.28.0.dev0",
#   "use_cache": true,
#   "vocab_size": 32000
# }


def small_cfg():
    cfg = lsg.SplitLlamaConfig(
        seq_len=256,
        hidden_dim=16,
        intermediate_dim=64,
        num_heads=4,
        num_layers=2,
        num_kv_heads=4,
        skip_indices=[1],
        skip_after_k_tokens=32,
    )
    return cfg


def test_loraize():
    Vocab = Axis("Vocab", 32000)
    cfg = small_cfg()
    key = jax.random.PRNGKey(0)
    model = lsg.LlamaLMHeadModel.init(Vocab, cfg, key=key)

    model_lora = cfg.loraize(model, key=key)
    for layer_idx in range(cfg.Layers.size):
        if layer_idx not in cfg.skip_indices:
            sa = model_lora.transformer.layers.blocks[layer_idx].self_attn
            for proj in [sa.q_proj, sa.k_proj, sa.v_proj]:
                assert isinstance(proj, lsg.SplitLoraLinear)


def _get_random_inputs(config: lsg.SplitLlamaConfig, Vocab: Axis):
    Batch = hax.Axis("batch", 2)
    x = hax.random.randint(jax.random.PRNGKey(0), (Batch, config.Pos), 0, Vocab.size)
    mask = AttentionMask.causal()
    return x, mask


def test_splitgen_forward():
    Vocab = Axis("Vocab", 32000)
    cfg = small_cfg()
    key = jax.random.PRNGKey(0)
    model = lsg.LlamaLMHeadModel.init(Vocab, cfg, key=key)

    model_lora = cfg.loraize(model, key=key)
    model_lora = cfg.splitize(model_lora)
    inputs = _get_random_inputs(cfg, Vocab)
    out = model_lora(*inputs).array

    @hax.named_jit
    def jitted_fwd(x, mask):
        return model(x, mask)

    out2 = jitted_fwd(*inputs).array

    assert jax.numpy.allclose(out, out2, atol=1e-5).item()


@pytest.mark.entry
def test_train_splitgen_lm():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_test_corpus.construct_small_data_cache(tmpdir)
        model_cfg = small_cfg()
        try:
            config = split_lora_lm.TrainLmConfig(
                initialize_from_hf="trl-internal-testing/tiny-random-LlamaForCausalLM",
                data=data_config,
                model=model_cfg,
                trainer=split_lora_lm.TrainerConfig(
                    num_train_steps=2,
                    train_batch_size=len(jax.devices()),
                    max_eval_batches=1,
                    wandb=WandbConfig(mode="disabled"),
                    require_accelerator=False,
                    ray=RayConfig(auto_start_cluster=False),
                ),
            )
            split_lora_lm.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass
