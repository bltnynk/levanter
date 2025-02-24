import dataclasses
import os
import tempfile

import jmp
import numpy as np
import pytest
import ray
from jax.random import PRNGKey

import haliax as hax

import levanter
import levanter.main.routed_lm as routed_lm
import tiny_test_corpus
from levanter.callbacks import StepInfo
from levanter.data.text import FIMUrlSourceConfig, mk_fim_dataset
from levanter.distributed import RayConfig
from levanter.models.rotary import DefaultRotaryEmbeddingsConfig
from levanter.models.routed.common import ExpertInit, ExpertType
from levanter.models.routed.qwen import RQwenConfig
from levanter.models.routed.starcoder import RStarcoderConfig
from levanter.optim.config import AdamConfig
from levanter.store.cache import CacheOptions
from levanter.tracker.tracker import NoopConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import Trainer
from test_utils import skip_if_no_torch


@pytest.fixture(autouse=True, scope="session")
def ray_init_fixture():
    ray.init()
    yield
    ray.shutdown()


def small_model_cfg(cls, **kwargs):
    return cls(
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
        disable_expert_mask=False,
        expert_init=ExpertInit.NONZERO,  # supported by every type
        rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0),
        **kwargs,
    )


@pytest.fixture(scope="session")
def data_cfg():
    with tempfile.TemporaryDirectory() as data_dir:
        train_urls = [
            tiny_test_corpus.write_fim_data(data_dir + f"/test_data_{i}.jsonl", len=16, flattened=True)
            for i in range(2)
        ]
        test_validation_jsonl = tiny_test_corpus.write_fim_data(
            data_dir + "/test_data_valid.jsonl", len=16, flattened=True
        )
        yield FIMUrlSourceConfig(
            cache_dir=data_dir + "/cache",
            train_urls=train_urls,
            validation_urls=[test_validation_jsonl],
            add_router_token=False,
            predict_fim_token=False,
            predict_router_token=False,
            cache_options=CacheOptions(
                target_size_per_flush=1,
                final_copy_cpus=1,
                final_copy_memory=64 * 1024,
                shard_tokenize_memory=64 * 1024,
            ),
            predict_prefix=False,
            pack=True,
            data_format="flattened",
            shuffle=True,
        )


def get_opt_cfg():
    return AdamConfig(learning_rate=0.01, weight_decay=0.01, warmup=0.00, lr_schedule="constant")


@pytest.mark.entry
@pytest.mark.parametrize("cfg_cls", [RQwenConfig, RStarcoderConfig])
@pytest.mark.parametrize("expert_type", [t for t in ExpertType])
@pytest.mark.parametrize("prefill_expert", [True, False], ids=["prefill", "no_prefill"])
@skip_if_no_torch
def test_routed_train(
    data_cfg,
    cfg_cls,
    expert_type,
    prefill_expert,
    router_activation="softmax",
    route_each_layer=True,
    zloss_seq_norm=True,
    full_ft=False,
    base_params_optim=None,
    model_kwargs={},
    trainer_kwargs={},
):
    if cfg_cls == RQwenConfig:
        from transformers import Qwen2ForCausalLM as TFCausalLm
    elif cfg_cls == RStarcoderConfig:
        from transformers import Starcoder2ForCausalLM as TFCausalLm

    if cfg_cls == RStarcoderConfig and expert_type == ExpertType.MLP_GLU:
        pytest.skip("Starcoder does not support MLP_GLU")

    # just testing if train_lm has a pulse
    model_cfg = small_model_cfg(
        cfg_cls,
        expert_type=expert_type,
        prefill_expert=prefill_expert,
        router_activation=router_activation,
        route_each_layer=route_each_layer,
        **model_kwargs,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer = data_cfg.the_tokenizer
        hf_config = model_cfg.to_hf_config(tokenizer.vocab_size)
        torch_model = TFCausalLm(hf_config)
        torch_model_dir = tmpdir + "/torch_model"
        torch_model.save_pretrained(torch_model_dir)

        try:
            config = routed_lm.TrainLmConfig(
                initialize_from_hf=torch_model_dir,
                data=data_cfg,
                model=model_cfg,
                full_ft=full_ft,
                full_ft_base_weights_optimizer=base_params_optim,
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
                router_z_loss_normalize_by_seqlen=zloss_seq_norm,
                embedding_router_token_ft=False,
                **trainer_kwargs,
            )
            routed_lm.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass


@pytest.mark.parametrize("cfg_cls", [RQwenConfig, RStarcoderConfig])
@pytest.mark.parametrize("expert_type", [t for t in ExpertType])
def test_full_ft_routed_train(data_cfg, cfg_cls, expert_type):
    test_routed_train(
        data_cfg,
        cfg_cls,
        expert_type=expert_type,
        prefill_expert=True,
        router_activation="softmax",
        route_each_layer=False,
        zloss_seq_norm=True,
        full_ft=True,
        base_params_optim=get_opt_cfg(),
    )


@pytest.mark.parametrize("cfg_cls", [RQwenConfig, RStarcoderConfig])
def test_expert_bias(data_cfg, cfg_cls):
    test_routed_train(
        data_cfg,
        cfg_cls,
        expert_type=ExpertType.MLP,
        prefill_expert=True,
        router_activation="softmax",
        route_each_layer=False,
        zloss_seq_norm=True,
        model_kwargs={"expert_bias_update_rate": 0.05, "router_act_before_topk": True},
    )


@skip_if_no_torch
def test_save_state_dict(data_cfg, cfg_cls=RQwenConfig):
    import safetensors

    with tempfile.TemporaryDirectory() as tmpdir:
        test_routed_train(
            data_cfg,
            cfg_cls,
            expert_type=ExpertType.MLP,
            prefill_expert=True,
            router_activation="softmax",
            route_each_layer=False,
            zloss_seq_norm=True,
            trainer_kwargs={"save_torch_state_path": str(tmpdir)},
        )
        for f in os.listdir(tmpdir):
            with safetensors.safe_open(f"{tmpdir}/{f}", framework="flax") as f:
                assert "router.weight" in f.keys()
                assert str(f.get_tensor("router.weight").dtype) == "bfloat16"


def test_eval_loop(data_cfg):
    model_cfg = small_model_cfg(RQwenConfig)
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
