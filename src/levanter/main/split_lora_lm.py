import logging
from dataclasses import dataclass, field
from typing import Optional, Union

import jax.random as jrandom
from jaxtyping import PyTree

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter import callbacks
from levanter.data.docbound_text import dset_from_config
from levanter.data.text import LMDatasetConfig, LMMixtureDatasetConfig
from levanter.models.llama_splitgen import SplitLlamaConfig
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count


logger = logging.getLogger(__name__)


@dataclass
class TrainLmConfig:
    data: Union[LMDatasetConfig, LMMixtureDatasetConfig] = field(default_factory=LMDatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: SplitLlamaConfig = field(default_factory=SplitLlamaConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)

    # config related to continued pretraining
    initialize_from_hf: Optional[str] = "meta-llama/Llama-2-7b-hf"
    skip_after_k_tokens: int = 32
    eval_loss_mask_after_k_tokens: Optional[int] = None

    data_seed: Optional[int] = None  # if provided, will override the data seed from the trainer
    trust_remote_code: bool = False
    full_ft: bool = False

    def __post_init__(self):
        assert self.model.seq_len > self.skip_after_k_tokens, "skip_after_k_tokens must be less than seq_len"


def main(config: TrainLmConfig):
    tokenizer = config.data.the_tokenizer

    converter = config.model.hf_checkpoint_converter()
    if config.initialize_from_hf is not None:
        if hasattr(tokenizer, "vocab") and tokenizer.vocab != converter.tokenizer.vocab:
            logger.warning("The tokenizers appear to be different. You may want to check this.")
        converter = converter.replaced(reference_checkpoint=config.initialize_from_hf, tokenizer=tokenizer)

    levanter.initialize(config)
    optimizer = config.optimizer.build(config.trainer.num_train_steps)
    model_config = config.model

    # Using the trainer as a context manager does 3 things:
    # 1. Sets the device mesh
    # 2. Sets the axis mapping (for fsdp)
    # 3. Sets the global metrics tracker
    with Trainer(config.trainer, optimizer) as trainer:
        # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
        # this makes deterministic training pretty easy
        seed = config.trainer.seed
        data_key, loader_key, lora_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 5)

        if config.data_seed is not None:
            logger.info(f"Overriding data seed with {config.data_seed}")
            data_key = jrandom.PRNGKey(config.data_seed)

        # We have two axis_mappings: one for storing the model and optimizer states, and one for compute
        # This allows Zero-3-style parameter sharding, where we shard the parameters and optimizer state across the mesh
        compute_axis_mapping = trainer.compute_axis_mapping
        parameter_axis_mapping = trainer.parameter_axis_mapping

        # some axes we need
        Batch = config.trainer.TrainBatch
        EvalBatch = config.trainer.EvalBatch
        Pos = config.model.Pos
        KeyPos = config.model.KeyPos

        valid_dset = dset_from_config(
            config.data,
            "validation",
            Pos,
            KeyPos,
            skip_after_k_tokens=config.skip_after_k_tokens,
            loss_mask_after_k_tokens=config.eval_loss_mask_after_k_tokens,
        )
        train_dset = dset_from_config(
            config.data,
            "train",
            Pos,
            KeyPos,
            skip_after_k_tokens=config.skip_after_k_tokens,
            dset_key=data_key,
            shuffle=True,
        )

        # to do partitioning, our dimensions have to be divisible by the size of the physical axes they're mapped to
        # For most things, we just insist you specify the config right, but tokenizers often have strange numbers of
        # tokens: gpt-2 has 50257, for example. So we round up.
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        if config.initialize_from_hf:
            logger.info(f"Loading pretrained model from {converter.reference_checkpoint}")
            model = converter.load_pretrained(
                model_config.model_type,
                model_config,
                axis_mapping=parameter_axis_mapping,
                dtype=trainer.mp.compute_dtype,
            )
        else:
            # TODO: how does this interact with sharding?
            # I need the is_trainable tree before passing the init fn so...
            model = hax.named_jit(
                lambda: model_config.build(Vocab, key=model_key), axis_resources=parameter_axis_mapping
            )()

        @hax.named_jit(axis_resources=parameter_axis_mapping, donate_args=(True))
        def loraize_hf_model(model):
            return model_config.loraize(model, key=lora_key)

        logger.info("Loraizing and splitting model")
        model = loraize_hf_model(model)
        is_trainable: PyTree = model_config.lora_only_trainable_filter(model)
        if config.full_ft:
            is_trainable = True

        state = trainer.initial_state(training_key, model=model, is_trainable=is_trainable)

        all_param_count = parameter_count(state.model)
        # TODO: remove this once we put this in trainer itself
        just_lora_params = parameter_count(state.trainable_model)

        levanter.tracker.log_summary(
            {
                "parameter_count": all_param_count,
                "trainable_parameter_count": just_lora_params,
                "fraction_trainable": just_lora_params * 1.0 / all_param_count,
            }
        )

        logger.info(f"Total parameter count: {all_param_count}")
        logger.info(f"Trainable parameter count: {just_lora_params}")
        logger.info(f"Fraction of parameters that are trainable: {just_lora_params * 1.0 / all_param_count:.3e}")

        if valid_dset is not None:
            max_eval_examples_per_ds = config.trainer.max_eval_batches
            if max_eval_examples_per_ds is not None:
                max_eval_examples_per_ds *= config.trainer.eval_batch_size

            cb = levanter.eval.cb_tagged_lm_evaluate(
                EvalBatch,
                [(valid_dset, ["valid"])],
                trainer.device_mesh,
                compute_axis_mapping,
                max_eval_examples_per_ds,
                mp=config.trainer.mp,
            )
            trainer.add_hook(cb, every=config.trainer.steps_per_eval)

        flops_per_token = config.model.flops_per_token(vocab_size)
        # 2x the flops, because we compute backprop of the inputs and the fwd pass.
        # assume the LoRA gradients/flops are really small
        flops_per_example = 2 * flops_per_token * Pos.size if flops_per_token is not None else None
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.train_batch_size, flops_per_example), every=1
        )

        # data loader. may need to seek to the right place if we're resuming
        train_loader = iter(trainer.sharded_loader(train_dset, Batch))

        if int(state.step) > 0:
            # step is after the batch, so we need to seek to step
            # TODO: implement iter_data.seek(resume_step +1)
            import tqdm

            for _ in tqdm.tqdm(range(state.step), desc="seeking data for resume"):
                next(train_loader)

        ## OK, actually run training!
        info = trainer.train(state, train_loader)
        # Checkpoint
        ckpt = trainer.config.checkpointer.create(trainer.run_id)
        ckpt.on_step(info, force=True)
        ckpt.wait_until_finished()


if __name__ == "__main__":
    levanter.config.main(main)()
