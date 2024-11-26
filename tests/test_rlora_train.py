import os
import tempfile

import jax
import pytest

import levanter.main.rlora_train as rlora_train
import tiny_test_corpus
from levanter.data.text import FIMUrlSourceConfig
from levanter.distributed import RayConfig
from levanter.tracker.wandb import WandbConfig


@pytest.mark.entry
def test_rlora_train():
    # just testing if train_lm has a pulse
    with tempfile.TemporaryDirectory() as tmpdir:
        test_data_jsonl = "file://" + tiny_test_corpus.write_fim_data(tmpdir + "/test_data.jsonl")
        cfg = FIMUrlSourceConfig(cache_dir=tmpdir + "/cache", train_urls=[test_data_jsonl])
        try:
            config = rlora_train.TrainLmConfig(
                data=cfg,
                model=rlora_train.RQwenConfig(
                    num_layers=2,
                    num_heads=2,
                    num_kv_heads=2,
                    seq_len=64,
                    hidden_dim=32,
                    attn_backend=None,  # use default for platform
                    num_loras=64,
                    lora_rank=2,
                ),
                trainer=rlora_train.TrainerConfig(
                    num_train_steps=2,
                    train_batch_size=len(jax.devices()),
                    max_eval_batches=1,
                    wandb=WandbConfig(mode="disabled"),
                    require_accelerator=False,
                    ray=RayConfig(auto_start_cluster=False),
                ),
            )
            rlora_train.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass
