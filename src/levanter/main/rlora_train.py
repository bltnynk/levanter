import dataclasses
import functools
import gc
import logging
from dataclasses import dataclass, field
from typing import Optional, Type

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
from haliax import Axis, NamedArray
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
from levanter import callbacks
from levanter.checkpoint import EpochCheckpointer, load_checkpoint
from levanter.data.text import FIMUrlSourceConfig, mk_fim_dataset
from levanter.models.lm_model import LmExample, RoutableLmExample
from levanter.models.loss import next_token_loss
from levanter.models.routed_lora_model import RQwenConfig, RQwenLMHeadModel, lora_trainable_params_filter
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count


logger = logging.getLogger(__name__)


@dataclass
class TrainLmConfig:
    data: FIMUrlSourceConfig = field(default_factory=FIMUrlSourceConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: RQwenConfig = field(default_factory=RQwenConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    initialize_from_hf: Optional[str] = None
    initialize_from_checkpoint_path: Optional[str] = None

    # if provided, will initialize from this checkpoint, used for llama style data mixture
    epoch: int = 0
    z_loss_weight: float = 0.0


def compute_next_token_loss(
    model: RQwenLMHeadModel,
    example: LmExample,
    *,
    batch_axis: Axis,
    key=None,
    reduction: Optional[hax.ReductionFunction] = hax.mean,
    reduction_axis: Optional[hax.AxisSelection] = None,
    logsumexp_weight: Optional[float] = None,
    loss_dtype: Optional[Type[jnp.dtype]] = jnp.float32,
) -> jnp.ndarray | NamedArray:
    """
    Computes the cross-entropy loss for a language modeling example. If reduction is not None, the loss is reduced
    across the reduction axis (with reduction_axis=None meaning all axes). If reduction is None, the loss is not
    reduced, and the result is a named array with axes (*batch axes, sequence_length).
    """
    assert isinstance(example, RoutableLmExample)
    # This is problematic, we don't get correctly batched ones so...
    idxs = jnp.squeeze(example.router_hs_idxs, axis=1)
    idxs = hax.NamedArray(idxs, (batch_axis,))
    example = dataclasses.replace(example, router_hs_idxs=idxs)
    activations = model.routed_forward(
        batch_axis, example.tokens, example.router_hs_idxs, example.attn_mask, key=key, activations=True
    )

    loss = next_token_loss(
        model.Pos,
        model.Embed,
        model.Vocab,
        activations,
        model.get_lm_head(),
        example.tokens,
        loss_mask=example.loss_mask,
        reduction=reduction,
        reduction_axis=reduction_axis,
        logsumexp_weight=logsumexp_weight,
        dtype=loss_dtype,
        block_size=model.config.cross_entropy_block_size,
    )

    return loss


def main(config: TrainLmConfig):
    tokenizer = config.data.the_tokenizer

    # this is some unpleasant code to allow us to initialize from a hf checkpoint. If this is your first read through,
    # I recommend skipping it for now
    if config.initialize_from_hf:
        if config.trainer.initialize_from is not None:
            raise ValueError("Cannot specify both initialize_from_hf and initialize_from")

        converter = config.model.hf_checkpoint_converter()
        if hasattr(tokenizer, "vocab") and tokenizer.vocab != converter.tokenizer.vocab:
            logger.warning("The tokenizers appear to be different. You may want to check this.")

        converter = converter.replaced(reference_checkpoint=config.initialize_from_hf, tokenizer=tokenizer)

    levanter.initialize(config)
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    # some axes we need
    Batch = config.trainer.TrainBatch
    Pos = config.model.Pos

    loss_function = functools.partial(
        compute_next_token_loss, logsumexp_weight=config.z_loss_weight, batch_axis=config.trainer.TrainBatch
    )

    # Using the trainer as a context manager does 3 things:
    # 1. Sets the device mesh
    # 2. Sets the axis mapping (for fsdp)
    # 3. Sets the global metrics tracker
    with Trainer(config.trainer, optimizer, loss_function) as trainer:
        # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
        # this makes deterministic training pretty easy
        seed = config.trainer.seed
        model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 2)

        # We have two axis_mappings: one for storing the model and optimizer states, and one for compute
        # This allows Zero-3-style parameter sharding, where we shard the parameters and optimizer state across the mesh
        parameter_axis_mapping = trainer.parameter_axis_mapping

        train_dataset = mk_fim_dataset(config.data, "train", tokenizer, Pos)

        # add epoch logging if epochs specified
        if config.epoch > 0:
            total_tokens_future = callbacks.get_total_dataset_tokens(train_dataset, config.model.seq_len)
            trainer.add_hook(
                callbacks.log_epoch_progress(
                    total_tokens_future, Pos.size, trainer.config.train_batch_size, max_epochs=config.epoch
                ),
                every=1,
            )

            # Add epoch checkpoint callback
            epoch_checkpointer = EpochCheckpointer(
                checkpointer=trainer.config.checkpointer.create(trainer.run_id),
                every_n_epochs=1,  # Or configure as needed
                total_dataset_size=total_tokens_future.result(),
                batch_size=trainer.config.train_batch_size,
            )
            trainer.add_hook(epoch_checkpointer, every=1)

        # to do partitioning, our dimensions have to be divisible by the size of the physical axes they're mapped to
        # For most things, we just insist you specify the config right, but tokenizers often have strange numbers of
        # tokens: gpt-2 has 50257, for example. So we round up.
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        def model_init():
            return config.model.build(Vocab, key=model_key)

        model_shape = eqx.filter_eval_shape(model_init)
        is_trainable = lora_trainable_params_filter(model_shape)

        state = trainer.initial_state(training_key, model_init=model_init, is_trainable=is_trainable)

        seek_dataloader = True
        if int(state.step) == 0 and config.initialize_from_checkpoint_path is not None:
            state = load_checkpoint(state, config.initialize_from_checkpoint_path)
            seek_dataloader = False

        if int(state.step) == 0:
            # TODO: I don't love that we init the model twice, but it's not a big deal i think?
            if config.initialize_from_hf:
                # initialize from an hf pretrained model
                logger.info(
                    "No training checkpoint found. Initializing model from HF checkpoint"
                    f" '{converter.reference_checkpoint}'"
                )
                # this is a bit gross, but we want to free up the memory from the model we just built
                state = dataclasses.replace(state, model=None)
                gc.collect()
                model = converter.load_pretrained(
                    config.model.model_type,
                    config=config.model,
                    axis_mapping=parameter_axis_mapping,
                    dtype=trainer.mp.compute_dtype,
                )
                model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(model)
                state = dataclasses.replace(state, model=model)
            else:
                logger.info("No checkpoint found. Starting from scratch.")

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        max_eval_examples_per_ds = config.trainer.max_eval_batches
        if max_eval_examples_per_ds is not None:
            max_eval_examples_per_ds *= config.trainer.eval_batch_size

        if len(config.data.validation_urls) > 0:
            eval_dataset = mk_fim_dataset(config.data, "validation", tokenizer, Pos)
            trainer.add_eval_hook(eval_dataset)

        flops_per_token = config.model.flops_per_token(vocab_size)
        # Doing 3x the flops: 1 to get the router set, 1 w/the loras, 1 for the backprop (+ the loras but that's small)
        flops_per_example = 3 * flops_per_token * Pos.size if flops_per_token is not None else None
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.train_batch_size, flops_per_example), every=1
        )

        train_loader = trainer.data_loader(train_dataset, Batch)
        if seek_dataloader:
            train_loader = train_loader.iter_from_step(state.step)
        else:
            train_loader = iter(train_loader)

        ## OK, actually run training!
        last_info = trainer.train(state, train_loader)

        # If running EpochDataset save latest checkpoint by default
        if trainer.config.checkpointer is not None and config.epoch > 0:
            trainer.run_hooks(last_info, force=True)
            checkpointer = trainer.config.checkpointer.create(trainer.run_id)
            checkpointer.wait_until_finished()

    # This isn't necessary except when Levanter is run in a subprocess (as happens w/ ray)
    trainer.tracker.finish()


if __name__ == "__main__":
    levanter.config.main(main)()
