import os
import tempfile

import jmp
import pytest

import levanter.main.rlora_train as rlora_train
import tiny_test_corpus
from levanter.data.text import FIMUrlSourceConfig
from levanter.distributed import RayConfig
from levanter.models.rotary import DefaultRotaryEmbeddingsConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from test_utils import skip_if_no_torch


@pytest.mark.entry
@skip_if_no_torch
def test_rlora_train():
    from transformers import Qwen2ForCausalLM

    # just testing if train_lm has a pulse
    model_cfg = rlora_train.RQwenConfig(
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
