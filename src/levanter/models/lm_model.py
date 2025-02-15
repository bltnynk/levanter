import abc
from dataclasses import dataclass
from typing import Generic, Optional, Type, TypeVar

import draccus
import equinox as eqx
import jax.numpy as jnp
from jax.random import PRNGKey

import haliax as hax
from haliax import Axis, NamedArray, NamedOrNumeric

from levanter.models.attention import AttentionMask
from levanter.models.loss import maybe_fused_next_token_loss
from levanter.utils.types import Extras


LmConfigT = TypeVar("LmConfigT", bound="LmConfig")
LmT = TypeVar("LmT", bound="LmHeadModel")


class LmExample(eqx.Module):
    tokens: hax.NamedArray
    loss_mask: hax.NamedArray
    attn_mask: AttentionMask | NamedArray = AttentionMask.causal()

    @staticmethod
    def causal(
        tokens: hax.NamedArray,
        *,
        loss_mask: Optional[hax.NamedArray] = None,
        ignore_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        segment_ids: Optional[hax.NamedArray] = None,
    ) -> "LmExample":
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1D array")

        if not jnp.issubdtype(tokens.dtype, jnp.integer):
            raise ValueError("tokens must be an integer array")

        Pos = tokens.axes[0]

        causal_loss_mask = LmExample.causal_loss_mask(Pos)

        if loss_mask is not None:
            loss_mask = loss_mask & causal_loss_mask.astype(loss_mask.dtype)
        else:
            loss_mask = causal_loss_mask

        if ignore_id is not None:
            # we don't compute loss for any tokens matching the ignore index
            ignore_mask = hax.roll(tokens, -1, Pos) != ignore_id
            loss_mask = loss_mask * ignore_mask

        loss_mask = loss_mask.astype(jnp.int32)

        attn_mask = AttentionMask.causal()

        if eos_id is not None and segment_ids is None:
            # the next token after an eos token is in a new segment
            eos_mask = hax.roll(tokens, 1, Pos) == eos_id
            # first token is always in segment 0
            eos_mask = eos_mask.at[Pos, 0].set(False).astype(jnp.int32)
            segment_ids = hax.cumsum(eos_mask, axis=Pos)
            attn_mask = attn_mask.with_segment_ids(segment_ids)
        elif segment_ids is not None:
            attn_mask = attn_mask.with_segment_ids(segment_ids)

        return LmExample(tokens=tokens, loss_mask=loss_mask, attn_mask=attn_mask)

    @staticmethod
    def from_prompt_and_completion(
        Pos,
        tokens: hax.NamedArray,
        prompt_length: NamedOrNumeric,
        *,
        ignore_id: Optional[int] = None,
        all_causal: bool = True,
    ) -> "LmExample":
        if all_causal:
            attn_mask = AttentionMask.causal()
        else:
            # causal just for the completion part. We don't have a special structured mask for this, so we just
            raise NotImplementedError("Not implemented yet")

        # mask out the prompt tokens
        loss_mask = LmExample.causal_loss_mask(Pos, prompt_length=prompt_length)

        if ignore_id is not None:
            # we don't compute loss for any tokens matching the ignore index
            ignore_mask = hax.roll(tokens, -1, Pos) != ignore_id
            loss_mask = loss_mask * ignore_mask

        return LmExample(tokens=tokens, loss_mask=loss_mask, attn_mask=attn_mask)

    def num_elements(self):
        return self.loss_mask.sum()

    @staticmethod
    def causal_loss_mask(Pos: Axis, prompt_length: Optional[int] = None) -> NamedArray:
        loss_mask = 1 - hax.nn.one_hot(-1, Pos, dtype=jnp.float32)

        if prompt_length is not None:
            # don't predict the prompt tokens
            prompt_mask = hax.arange(Pos) >= prompt_length - 1
            loss_mask = loss_mask * prompt_mask

        return loss_mask


class RoutableLmExample(LmExample):
    router_hs_idxs: Optional[hax.NamedArray] = None
    """-1 if the token is not routable, otherwise the index of the router hidden state"""
    completion_mask: Optional[hax.NamedArray] = None
    """Mask that's 1 for every element of the completion"""
    completion_first_token_mask: Optional[hax.NamedArray] = None
    """Mask that's 1 for the first token of each completion"""

    @staticmethod
    def causal_with_hs_idxs(
        tokens: hax.NamedArray,
        router_hs_idxs: Optional[hax.NamedArray] = None,
        *,
        loss_mask: Optional[hax.NamedArray] = None,
        completion_mask: Optional[hax.NamedArray] = None,
        completion_first_token_mask: Optional[hax.NamedArray] = None,
        ignore_id: Optional[int] = None,
        eos_id: Optional[int] = None,
    ) -> "RoutableLmExample":
        lm_example = LmExample.causal(tokens, loss_mask=loss_mask, ignore_id=ignore_id, eos_id=eos_id)
        return RoutableLmExample(
            tokens=lm_example.tokens,
            completion_first_token_mask=completion_first_token_mask,
            loss_mask=lm_example.loss_mask,
            attn_mask=lm_example.attn_mask,
            router_hs_idxs=router_hs_idxs,
            completion_mask=completion_mask,
        )

    def num_elements(self, completion=False):
        if completion:
            return self.completion_mask.sum()
        return super().num_elements()


# TODO: for some reason, mypy doesn't like the discover_packages_path argument?
@dataclass(frozen=True)
class LmConfig(draccus.PluginRegistry, abc.ABC, Generic[LmT], discover_packages_path="levanter.models"):  # type: ignore
    @property
    @abc.abstractmethod
    def model_type(cls) -> Type[LmT]:
        pass

    @property
    @abc.abstractmethod
    def KeyPos(self) -> Axis:
        pass

    @property
    @abc.abstractmethod
    def Pos(self) -> Axis:
        pass

    @property
    @abc.abstractmethod
    def Embed(self) -> Axis:
        pass

    cross_entropy_block_size: Optional[int] = None
    """
    The block size for computing cross-entropy loss. This is the number of tokens that are processed together
    in a single block. This can be adjusted to fit within memory constraints. It's deliberately set to a large
    value because it usually faster to compute the loss in larger blocks.
    """

    def flops_per_token(self, vocab_size: int) -> Optional[float]:
        return None

    def build(self, Vocab: Axis, *, key: PRNGKey) -> "LmT":
        return self.model_type.init(Vocab, self, key=key)  # type: ignore


class LmHeadModel(eqx.Module, Generic[LmConfigT]):
    """
    Superclass for models with a language modeling head.
    """

    @property
    @abc.abstractmethod
    def config(self) -> LmConfigT:
        pass

    @property
    @abc.abstractmethod
    def Vocab(self) -> Axis:
        pass

    @property
    def Pos(self) -> Axis:
        return self.config.Pos

    @property
    def KeyPos(self) -> Axis:
        return self.config.KeyPos

    @property
    def Embed(self) -> Axis:
        return self.config.Embed

    @classmethod
    @abc.abstractmethod
    def init(cls, Vocab: Axis, config: LmConfigT, *, key: PRNGKey) -> "LmHeadModel[LmConfigT]":
        pass

    def __call__(
        self, input_ids: NamedArray, attn_mask: Optional[AttentionMask | NamedArray] = None, *, key=None
    ) -> NamedArray:
        """
        Compute the logits for the next token in a sequence.
        Args:
            input_ids: token IDs with shape [..., Pos]
            attn_mask: attention mask with shape [..., Pos, KeyPos]
            key: PRNGKey for random number generation

        Returns:
            NamedArray: logits with shape [..., Pos, Vocab]

        """
        x = self.activations(input_ids, attn_mask, key=key)
        lm_logits = hax.dot(x, self.get_lm_head(), axis=self.Embed)

        return lm_logits

    @abc.abstractmethod
    def activations(
        self, input_ids: NamedArray, attn_mask: Optional[AttentionMask | NamedArray] = None, *, key=None
    ) -> NamedArray:
        """
        Compute the activations for the next token in a sequence.
        Args:
            input_ids: token IDs with shape {Pos}
            attn_mask: attention mask with shape {Pos, KeyPos}
            key: PRNGKey for random number generation

        Returns:
            NamedArray: activations with shape {Pos, Embed}

        """
        pass

    @abc.abstractmethod
    def get_lm_head(self) -> hax.NamedArray:
        """
        The language modeling head of the model. Should have shape {Embed, Vocab}.
        """
        raise NotImplementedError("get_lm_head not implemented")

    @abc.abstractmethod
    def resize_vocab(self, new_size: int, key: Optional[PRNGKey] = None) -> "LmHeadModel[LmConfigT]":
        """
        Resizes the vocabulary of the model. Key may be provided to use random initialization, otherwise, there
        should be some deterministic initialization of any new parameters.
        """
        pass

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size


def compute_next_token_loss(
    model: LmHeadModel,
    example: LmExample,
    *,
    key=None,
    logsumexp_weight: Optional[float] = None,
    loss_dtype: Optional[Type[jnp.dtype]] = jnp.float32,
) -> tuple[NamedArray, NamedArray, Extras]:
    """
    Computes the cross-entropy loss for a language modeling example. If reduction is not None, the loss is reduced
    across the reduction axis (with reduction_axis=None meaning all axes). If reduction is None, the loss is not
    reduced, and the result is a named array with axes (*batch axes, sequence_length).
    """
    activations = model.activations(example.tokens, example.attn_mask, key=key)
    if isinstance(activations, tuple):
        activations, extras = activations
    else:
        extras = Extras()

    losses, where = maybe_fused_next_token_loss(
        model.Pos,
        model.Embed,
        model.Vocab,
        activations,
        model.get_lm_head(),
        example.tokens,
        loss_mask=example.loss_mask,
        logsumexp_weight=logsumexp_weight,
        dtype=loss_dtype,
        block_size=model.config.cross_entropy_block_size,
    )

    return losses, where, extras
