import os
import tempfile

import jax
import jax.numpy as jnp
import jmp
import pytest

import haliax as hax

import levanter.main.rlora_train as rlora_train
import tiny_test_corpus
from levanter.data.text import FIMUrlSourceConfig
from levanter.distributed import RayConfig
from levanter.models.rotary import DefaultRotaryEmbeddingsConfig
from levanter.models.routed_lora_model import RQwenLMHeadModel
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from test_utils import skip_if_no_torch


def small_model_cfg():
    return rlora_train.RQwenConfig(
        num_layers=4,
        num_heads=2,
        num_kv_heads=2,
        seq_len=64,
        hidden_dim=32,
        intermediate_dim=64,
        attn_backend="jax_flash",
        num_loras=16,
        lora_rank=1,
        top_k=4,
        tie_word_embeddings=True,
        disable_lora_mask=False,
        use_layer_norm_weight=True,
        rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0),
    )


@pytest.mark.entry
@skip_if_no_torch
def test_rlora_train():
    from transformers import Qwen2ForCausalLM

    # just testing if train_lm has a pulse
    model_cfg = small_model_cfg()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_data_jsonl = tiny_test_corpus.write_fim_data(tmpdir + "/test_data.jsonl", len=2048)
        test_validation_jsonl = tiny_test_corpus.write_fim_data(tmpdir + "/test_data_valid.jsonl", len=2048)
        data_cfg = FIMUrlSourceConfig(
            cache_dir=tmpdir + "/cache",
            train_urls=[test_data_jsonl],
            validation_urls=[test_validation_jsonl],
            add_router_token=True,
            predict_fim_token=False,
            predict_router_token=False,
            predict_prefix=False,
        )
        tokenizer = data_cfg.the_tokenizer
        hf_config = model_cfg.to_hf_config(tokenizer.vocab_size)
        torch_model = Qwen2ForCausalLM(hf_config)
        torch_model_dir = tmpdir + "/torch_model"
        torch_model.save_pretrained(torch_model_dir)

        try:
            config = rlora_train.TrainLmConfig(
                initialize_from_hf=torch_model_dir,
                data=data_cfg,
                model=model_cfg,
                trainer=rlora_train.TrainerConfig(
                    num_train_steps=16,
                    train_batch_size=8,
                    per_device_parallelism=1,  # test out grad accum
                    max_eval_batches=1,
                    steps_per_eval=2,
                    wandb=WandbConfig(mode="disabled"),
                    require_accelerator=False,
                    ray=RayConfig(auto_start_cluster=False),
                    fsdp_axis="embed",
                    batch_axis="batch",
                    tensor_parallel_axes=["mlp", "heads"],
                    mp=jmp.get_policy("p=f32,c=bf16"),
                ),
                optimizer=AdamConfig(learning_rate=0.001, weight_decay=0.1, warmup=0.00, lr_schedule="constant"),
                router_z_loss_weight=0.001,
                full_ft=False,
                embedding_router_token_ft=True,
            )
            rlora_train.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass


def test_embedding_grad_filter():
    from levanter.main.rlora_train import filter_embedding_grads

    # test that the filter works
    model_cfg = small_model_cfg()
    Vocab = hax.Axis("vocab", 100)
    model = RQwenLMHeadModel.init(Vocab, model_cfg, key=jax.random.PRNGKey(0))

    def replace_with_ones(x):
        return hax.ones_like(x)

    def is_leaf(x):
        return isinstance(x, hax.NamedArray)

    model: RQwenLMHeadModel = jax.tree.map(replace_with_ones, model, is_leaf=is_leaf)
    assert model.embeddings.token_embeddings.weight.array[0, 0].item() == 1.0

    token_mask = hax.ones(Vocab, dtype=jnp.bool).at[Vocab, 50].set(0.0)
    tform = filter_embedding_grads(model.Embed, Vocab, token_mask)
    res, _ = tform.update(model, ())
    res: RQwenLMHeadModel
    token_embs = res.embeddings.token_embeddings.weight
    assert hax.all(
        token_embs[Vocab, 50] == 0.0
    ).item(), f"Token 50 should be zeroed out, got {token_embs[Vocab, 50].tolist()}"

    print(model)
