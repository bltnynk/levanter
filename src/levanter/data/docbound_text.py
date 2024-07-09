import functools
from typing import Iterator, List, Optional, Tuple

import equinox as eqx
import jax
import numpy as np

import haliax as hax

from levanter.data.dataset import ShardableDataset
from levanter.data.text import TokenizedDocumentCache
from levanter.models.lm_model import LmExample
from levanter.utils.jax_utils import use_cpu_device


def _lens(data):
    return [len(d) for d in data]


def yield_example2(
    data: List[List[int]],
    seq_len: int,
    min_doc_length: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:

    assert sum(_lens(data)) >= seq_len, "Data must be at least seq_len long"
    example_parts: List[np.ndarray] = []
    while sum(_lens(example_parts)) < seq_len:
        if len(data) == 0:
            break
        s = data.pop(0)
        if len(s) > min_doc_length:
            example_parts.append(np.array(s))

    start_inds = np.cumsum([0] + _lens(example_parts[:-1]))
    example = np.concatenate(example_parts)
    tail = None
    if len(example) > seq_len:
        example, tail = example[:seq_len], example[seq_len:]
    return start_inds, example, tail


def make_examples(
    data: List[List[int]], seq_len: int, min_doc_length: int, keep_tail: bool = False
) -> List[Tuple[np.ndarray, np.ndarray]]:
    examples = []
    while sum(_lens(data)) >= seq_len:
        si, ex, tail = yield_example2(data, seq_len, min_doc_length)
        examples.append((si, ex))
        if keep_tail and tail is not None and len(tail) > min_doc_length:
            data.insert(0, tail.tolist())
    return examples


class TokenSeqWithDocBoundsDataset(ShardableDataset[Tuple[np.ndarray, np.ndarray]]):
    """
    A dataset that yields sequences of tokens of fixed length from a TokenizedDocumentCache.

    :param doc_cache: the TokenizedDocumentCache to draw from
    :param seq_len: The max length of sequences to emit
    """

    def __init__(
        self, doc_cache, seq_len: int, stride: Optional[int] = None, min_doc_len: int = 0, keep_tail: bool = False
    ):
        self.doc_cache = doc_cache
        self.seq_len = seq_len
        self.stride = stride
        self.min_doc_len = min_doc_len
        self.keep_tail = keep_tail

    def shard(self, shard_id: int, num_shards: int) -> "TokenSeqWithDocBoundsDataset":
        """
        Split the dataset into num_processes shards.
        """
        return TokenSeqWithDocBoundsDataset(
            self.doc_cache.shard(shard_id, num_shards), self.seq_len, self.stride, self.min_doc_len, self.keep_tail
        )

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        extra_tokens = None  # BatchEncoding of the last tokens from the previous doc
        for doc in self.doc_cache:
            # TODO: we could be cleverer here, and avoid these expensive copies etc
            # should run some benchmarks to see if it's worth it

            doc = doc.to_pydict()
            if extra_tokens is not None:
                for k in extra_tokens:
                    doc[k] = extra_tokens[k] + doc[k]
                extra_tokens = None
            # This mutably pulls examples out of doc
            examples = make_examples(doc["input_ids"], self.seq_len, self.min_doc_len, keep_tail=self.keep_tail)
            if len(doc["input_ids"]) > 0:
                extra_tokens = doc

            yield from examples

    @staticmethod
    def load(
        seq_len: int, cache_dir: str, stride: Optional[int] = None, min_doc_len: int = 0, keep_tail: bool = False
    ) -> "TokenSeqWithDocBoundsDataset":
        # Maybe force the cache to be built ahead of time?
        doc_cache = TokenizedDocumentCache.load(cache_dir, False)
        return TokenSeqWithDocBoundsDataset(doc_cache, seq_len, stride, min_doc_len=min_doc_len, keep_tail=keep_tail)


def make_skip_mask(start_inds: np.ndarray, seq_len: int, skip_after_k_tokens: int) -> np.ndarray:
    split_mask = np.ones((seq_len,), dtype=bool)
    for ind in start_inds:
        end = ind + skip_after_k_tokens
        split_mask[ind:end] = False
    return split_mask


class SplitLmExample(eqx.Module):
    lm_example: LmExample
    split_mask: hax.NamedArray

    @staticmethod
    def new(
        skip_mask: hax.NamedArray,
        tokens: hax.NamedArray,
        *,
        loss_mask: Optional[hax.NamedArray] = None,
        ignore_id: Optional[int] = None,
    ) -> "SplitLmExample":
        if tokens.ndim != 1:
            raise ValueError(f"tokens must be a 1D array: {tokens.shape}")
        if skip_mask.ndim != 1:
            raise ValueError(f"skip mask must be a 1D array: {skip_mask.shape}")
        if skip_mask.shape != tokens.shape:
            raise ValueError(f"skip mask and tokens must have the same shape: {skip_mask.shape} != {tokens.shape}")
        lm_example = LmExample.causal(tokens=tokens, loss_mask=loss_mask, ignore_id=ignore_id)
        return SplitLmExample(lm_example, skip_mask)


class SplitGenDataset(ShardableDataset[LmExample]):
    def __init__(
        self,
        dataset: ShardableDataset[Tuple[np.ndarray, np.ndarray]],
        QPos: hax.Axis,
        KPos: hax.Axis,
        skip_after_k_tokens: int = 0,
        ignore_index: Optional[int] = None,
    ):
        self.dataset = dataset
        self.QPos = QPos
        self.KPos = KPos
        self.skip_after_k_tokens = skip_after_k_tokens
        self.ignore_id = ignore_index

    def shard(self, shard_id: int, num_shards: int) -> "SplitGenDataset":
        return SplitGenDataset(
            self.dataset.shard(shard_id, num_shards),
            self.QPos,
            self.KPos,
            self.skip_after_k_tokens,
            self.ignore_id,
        )

    def __iter__(self) -> Iterator[SplitLmExample]:
        sharding = jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])

        with use_cpu_device():

            @functools.partial(eqx.filter_jit, out_shardings=sharding)
            def _create_lm_example(skip_mask, tokens):
                tokens = hax.named(tokens, self.QPos)
                skip_mask = hax.named(skip_mask, self.QPos)
                example = SplitLmExample.new(
                    skip_mask,
                    tokens,
                    ignore_id=self.ignore_id,
                )
                return example

            for start_inds, tokens in self.dataset:
                skip_mask = make_skip_mask(start_inds, self.QPos.size, self.skip_after_k_tokens)
                example = _create_lm_example(skip_mask, tokens)
                yield example
