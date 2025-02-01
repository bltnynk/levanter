import dataclasses
import os
import tempfile

import jmp
import numpy as np
import pytest
from jax.random import PRNGKey

import haliax as hax

import levanter
import levanter.main.routed_lm as routed_lm
import tiny_test_corpus
from levanter.callbacks import StepInfo
from levanter.data.text import FIMUrlSourceConfig, mk_fim_dataset
from levanter.distributed import RayConfig
from levanter.models.rotary import DefaultRotaryEmbeddingsConfig
from levanter.models.routed_qwen_model import ExpertInit, ExpertType
from levanter.optim.config import AdamConfig
from levanter.store.cache import CacheOptions
from levanter.tracker.tracker import NoopConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import Trainer
from test_utils import skip_if_no_torch


def small_model_cfg(**kwargs):
    return routed_lm.RQwenConfig(
        num_layers=4,
        num_heads=2,
        num_kv_heads=2,
        seq_len=64,
        hidden_dim=32,
        intermediate_dim=64,
        attn_backend="jax_flash",
        num_experts=16,
        expert_rank=1,
        top_k=4,
        tie_word_embeddings=True,
        disable_expert_mask=False,
        use_layer_norm_weight=True,
        expert_init=ExpertInit.NONZERO,  # supported by every type
        rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0),
        **kwargs,
    )


@pytest.fixture
def data_cfg():
    with tempfile.TemporaryDirectory() as tmpdir:
        train_urls = [
            tiny_test_corpus.write_fim_data(tmpdir + f"/test_data_{i}.jsonl", len=16, flattened=True) for i in range(2)
        ]
        test_validation_jsonl = tiny_test_corpus.write_fim_data(
            tmpdir + "/test_data_valid.jsonl", len=16, flattened=True
        )
        yield FIMUrlSourceConfig(
            cache_dir=tmpdir + "/cache",
            train_urls=train_urls,
            validation_urls=[test_validation_jsonl],
            add_router_token=False,
            predict_fim_token=False,
            predict_router_token=False,
            cache_options=CacheOptions(target_size_per_flush=1, final_copy_cpus=1, final_copy_memory=64 * 1024 * 1024),
            predict_prefix=False,
            pack=True,
            data_format="flattened",
            shuffle=True,
        )


def get_opt_cfg():
    return AdamConfig(learning_rate=0.01, weight_decay=0.01, warmup=0.00, lr_schedule="constant")


@pytest.mark.entry
@pytest.mark.parametrize("expert_type", [t for t in ExpertType])
@pytest.mark.parametrize("prefill_expert", [True, False])
@pytest.mark.parametrize("router_activation", ["sigmoid", "softmax"])
@pytest.mark.parametrize("route_each_layer", [True, False])
@skip_if_no_torch
def test_routed_train(data_cfg: FIMUrlSourceConfig, expert_type, prefill_expert, router_activation, route_each_layer):
    from transformers import Qwen2ForCausalLM

    # just testing if train_lm has a pulse
    model_cfg = small_model_cfg(
        expert_type=expert_type,
        prefill_expert=prefill_expert,
        router_activation=router_activation,
        route_each_layer=route_each_layer,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer = data_cfg.the_tokenizer
        hf_config = model_cfg.to_hf_config(tokenizer.vocab_size)
        torch_model = Qwen2ForCausalLM(hf_config)
        torch_model_dir = tmpdir + "/torch_model"
        torch_model.save_pretrained(torch_model_dir)

        try:
            config = routed_lm.TrainLmConfig(
                initialize_from_hf=torch_model_dir,
                data=data_cfg,
                model=model_cfg,
                trainer=routed_lm.TrainerConfig(
                    seed=42,
                    num_train_steps=4,
                    train_batch_size=2,
                    per_device_parallelism=1,  # test out grad accum
                    max_eval_samples=16,
                    steps_per_eval=2,
                    tracker=WandbConfig(mode="disabled"),
                    require_accelerator=False,
                    ray=RayConfig(auto_start_cluster=False),
                    fsdp_axis="embed",
                    batch_axis="batch",
                    tensor_parallel_axes=["mlp", "heads"],
                    mp=jmp.get_policy("p=f32,c=bf16"),
                ),
                optimizer=get_opt_cfg(),
                router_z_loss_weight=0.001,
                full_ft=False,
                embedding_router_token_ft=False,
            )
            routed_lm.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass


def test_eval_loop(data_cfg):
    model_cfg = small_model_cfg()
    opt_cfg = get_opt_cfg()
    trainer_cfg = routed_lm.TrainerConfig(
        seed=42,
        max_eval_batches=32,
        per_device_eval_parallelism=1,
        train_batch_size=2,
        id="random",
        tracker=NoopConfig(),
        require_accelerator=False,
        ray=RayConfig(auto_start_cluster=False),
        mp=jmp.get_policy("p=f32,c=bf16"),
    )
    levanter.initialize(trainer_cfg)
    log_result = {}

    def log_fn(metrics, **kwargs):
        log_result.update(metrics | kwargs)

    with Trainer(trainer_cfg, opt_cfg.build(2), routed_lm.compute_next_token_loss) as trainer:
        key = PRNGKey(0)
        eval_dset = mk_fim_dataset(
            data_cfg, "validation", data_cfg.the_tokenizer, model_cfg.Pos, key=key, await_finished=False
        )
        trainer.add_eval_hook(eval_dset, name="eval_hook")
        Vocab = hax.Axis("vocab", len(data_cfg.the_tokenizer))
        state = trainer.initial_state(PRNGKey(1), model_cfg.build(Vocab, key=PRNGKey(2)))
        levanter.current_tracker().__setattr__("log", log_fn)
        state = dataclasses.replace(state, step=1)
        info = StepInfo(state, loss=0, step_duration=0)
        for hook in trainer.hooks.hooks:
            hook.fn.on_step(info)

        print(log_result)

        comp_loss = log_result["eval/eval_hook/completion_loss"]
        eval_loss = log_result["eval/eval_hook/loss"]
        all_lm_loss = log_result["eval/eval_hook/all_lm_loss"]
        assert np.allclose(comp_loss, eval_loss)
        assert np.allclose(comp_loss, all_lm_loss)
