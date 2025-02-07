import abc
import asyncio
import copy
import dataclasses
import functools
import json
import logging
import os
import random
import tempfile
import warnings
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from typing import Any, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple, TypeAlias, TypeVar, Union

import datasets
import equinox as eqx
import fsspec
import jax
import numpy as np
import regex
import tensorstore as ts
from draccus import field
from jax.random import PRNGKey
from jaxtyping import PRNGKeyArray
from tokenizers import normalizers

import haliax as hax
from haliax import Axis

import levanter
from levanter.data import AsyncDataset
from levanter.data.dataset import MappedAsyncDataset
from levanter.data.mixture import MixtureDataset, StopStrategy
from levanter.data.utils import FastReplacer
from levanter.models.attention import AttentionMask
from levanter.models.lm_model import LmExample, RoutableLmExample
from levanter.store.cache import CacheOptions, TreeCache
from levanter.store.jagged_array import JaggedArrayStore
from levanter.store.tree_store import TreeStore
from levanter.utils.fsspec_utils import expand_glob
from levanter.utils.hf_utils import HfTokenizer, num_cpus_used_by_tokenizer

# intercept the logging nonsense here
from levanter.utils.logging import silence_transformer_nag  # noqa


silence_transformer_nag()  # noqa
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast  # noqa

from levanter.compat.hf_checkpoints import load_tokenizer  # noqa
from levanter.data._preprocessor import BatchProcessor, U, dict_from_record_batch  # noqa
from levanter.data.metrics_monitor import LoggerMetricsMonitor, LoggingMetricsMonitor, MetricsMonitor  # noqa
from levanter.data.sharded_datasource import (  # noqa
    JsonlDataSource,
    ShardedDataSource,
    TextUrlDataSource,
    UrlDataSource,
    WrappedHFDataSource,
    gcs_glob,
)
from levanter.shapes import NamedShapeSpec, ShapeSpec  # noqa
from levanter.store.cache import build_or_load_cache  # noqa
from levanter.utils.jax_utils import key_iterator, use_cpu_device  # noqa


T_co = TypeVar("T_co", covariant=True)

logger = logging.getLogger("levanter.data.text")

# TASKS:
# TODO: consider adding indexing a la Map-style datasets
# TODO: support seeking/serialization/restore in the dataset

LEDGER_FILE = "ledger.json"

DEFAULT_IGNORE_INDEX = -100  # Mirrors pytorch's default ignore index


class EpochDataset(AsyncDataset[T_co]):
    """
    A dataset that wraps another dataset, providing infinite epochs by recycling indices.
    If `max_epochs` is specified, it limits the number of cycles before raising StopIteration.

    :param dataset: The dataset to wrap.
    :param max_epochs: The maximum number of epochs to cycle through. If None, cycle indefinitely.
    """

    def __init__(self, dataset: AsyncDataset[T_co], max_epochs: Optional[int] = None):
        super().__init__()
        self.dataset = dataset
        self.max_epochs = max_epochs

    async def async_len(self) -> int:
        if self.max_epochs is None:
            raise ValueError("Cannot determine length of an infinite dataset without max_epochs.")
        # Return the total number of samples: max_epochs * length of the dataset
        return self.max_epochs * await self.dataset.async_len()

    async def final_length_is_known(self) -> bool:
        return await self.dataset.final_length_is_known()

    def is_finite(self) -> bool:
        # EpochDataset can be finite if max_epochs is set.
        return self.max_epochs is not None

    async def current_len(self) -> Optional[int]:
        # If max_epochs is None, the dataset is effectively infinite.
        if self.max_epochs is None:
            return None

        # If the final length of the dataset is not known, return the current length of the underlying dataset.
        if not await self.dataset.final_length_is_known():
            return await self.dataset.current_len()

        # If the final length is known, return the max_epochs * async_len of the dataset.
        return self.max_epochs * await self.dataset.async_len()

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        # Use self.wait_until_len_at_least to ensure we have enough data for the batch.
        max_index = max(indices)
        ds_len = await self.dataset.wait_until_len_at_least(max_index + 1)

        # Determine the epoch based on the largest index
        epoch = max_index // ds_len

        # If max_epochs is specified, raise an error if the epoch exceeds the allowed number of epochs
        if self.max_epochs is not None and epoch >= self.max_epochs:
            raise StopIteration(
                f"Reached maximum number of epochs: epoch {epoch} exceeds the maximum allowed {self.max_epochs}"
            )

        # Wrap the indices within the bounds of the dataset length
        wrapped_indices = [idx % ds_len for idx in indices]

        # Delegate to the underlying dataset's get_batch
        return await self.dataset.get_batch(wrapped_indices)

    async def wait_until_len_at_least(self, length: int) -> int:
        """
        Returns the length of the dataset once it is at least `length` or if the dataset has a known (finished) length.
        If the dataset's actual length is less than `length`, it returns the minimum of async_len and the current length.
        """
        # Wait until the underlying dataset's length is at least `length`
        if not self.is_finite():
            return length

        if await self.dataset.final_length_is_known():
            base_length = await self.dataset.async_len()
        else:
            base_length = await self.dataset.wait_until_len_at_least(length)

        if base_length < length:
            # hit epoch boundary
            assert self.max_epochs is not None
            return self.max_epochs * base_length

        return base_length


class TokenSeqDataset(AsyncDataset[np.ndarray]):
    """
    A dataset that yields sequences of tokens of fixed length from an underlying TreeCache.

    :param doc_cache: the TreeCache to read from
    :param seq_len: The max length of sequences to emit
    """

    def __init__(self, doc_cache: TreeCache[dict], seq_len: int):
        super().__init__()
        self.doc_cache = doc_cache
        self.seq_len = seq_len
        self._store: Optional[TreeStore] = None
        self._cached_len: Optional[int] = None

    async def async_len(self) -> int:
        await self.doc_cache.finished()
        token_arrays = await self._await_token_cache()
        return token_arrays.data_size // self.seq_len

    async def _await_token_cache(self) -> JaggedArrayStore:
        if self._store is None:
            self._store = await self.doc_cache.store_async()
        return self._store.tree["input_ids"]

    async def final_length_is_known(self) -> bool:
        return await self.doc_cache.final_length_is_known()

    def is_finite(self) -> bool:
        return True

    async def current_len(self) -> Optional[int]:
        store = await self._await_token_cache()
        return store.data_size // self.seq_len

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        token_arrays = await self._await_token_cache()
        # logger.info(f"Time to get token cache: {time.time() - time_in}")
        len = await self.wait_until_len_at_least(max(indices) + 1)
        if len is not None and len < max(indices) + 1:
            raise ValueError("Requested indices beyond the end of the dataset")
        offsets = np.array(indices, dtype=np.int64) * self.seq_len
        with ts.Batch():
            out = []
            for offset in offsets:
                out.append(token_arrays.data[offset : offset + self.seq_len].read())

        out = await asyncio.gather(*out)
        return out

    async def wait_until_len_at_least(self, length: int) -> int:
        # length is brutally slow to compute, so we cache it
        if self._cached_len is not None and self._cached_len >= length:
            return self._cached_len

        # TODO: would be better to listen for cache updates
        length = await super().wait_until_len_at_least(length)
        self._cached_len = length
        return length


class CausalLmDataset(MappedAsyncDataset[np.ndarray, LmExample]):
    def __init__(
        self,
        dataset: AsyncDataset[np.ndarray],
        QPos: Axis,
        KPos: Axis,
        *,
        fcm_prob: float = 0.0,
        key: Optional[PRNGKey] = None,
        ignore_index: Optional[int] = None,
        eos_id: Optional[int] = None,
    ):
        self.dataset = dataset
        self.QPos = QPos
        self.KPos = KPos
        self.fcm_prob = fcm_prob
        self.ignore_id = ignore_index
        self.eos_id = eos_id
        self.key = key

        if self.fcm_prob > 0.0 and self.key is None:
            raise ValueError("must provide key if fcm_prob > 0.0")

        sharding = jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])

        @functools.partial(eqx.filter_jit, out_shardings=sharding)
        def _create_lm_example(tokens, key):
            tokens = hax.named(tokens, self.QPos)
            example = LmExample.causal(tokens=tokens, ignore_id=self.ignore_id, eos_id=eos_id)

            if self.fcm_prob > 0:
                # masks for attention
                # We support forgetful causal masking (FCM) which is a technique that improves training speed by
                # randomly masking out some of the context. This is a bit like dropout, but it's applied to the attention
                # mask instead of the activations. It's described in https://arxiv.org/abs/2210.13432
                assert self.key is not None
                this_key, key = jax.random.split(key)
                fcm_mask = hax.nn.attention.forgetful_causal_mask(self.KPos, self.fcm_prob, key=this_key)
                attn_mask = example.attn_mask & AttentionMask.explicit(fcm_mask)
                example = dataclasses.replace(example, attn_mask=attn_mask)

            return example

        super().__init__(self.dataset, _create_lm_example, key=key)

    async def async_len(self) -> int:
        return await self.dataset.async_len()


def _maybe_force_tokenizer_parallelism(tokenizer: PreTrainedTokenizerBase):
    if tokenizer.is_fast and os.getenv("TOKENIZERS_PARALLELISM") is None:
        # if we're using a fast tokenizer, we want to force parallelism
        # to be the number of CPUs
        os.environ["TOKENIZERS_PARALLELISM"] = "true"


LONG_STRING_WORKAROUND = 10_000

ws = regex.compile(r"\s")


class BatchTokenizer(BatchProcessor[str, dict]):
    """
    A batch processor that tokenizes a batch of strings using a tokenizer.
    By default, this will append eos to the end of the string, even if the tokenizer doesn't.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        enforce_bos=True,
        enforce_eos=True,
        *,
        override_resources=None,
        _workaround_len=LONG_STRING_WORKAROUND,
        return_attention_mask=False,
        padding=False,
        max_length=None,
    ):
        _maybe_force_tokenizer_parallelism(tokenizer)
        self.tokenizer = tokenizer
        self.override_resources = override_resources
        self.return_attention_mask = return_attention_mask
        self.padding = padding
        if max_length is not None:
            self.max_length = max_length
        else:
            self.max_length = self.tokenizer.model_max_length

        # see if the tokenizer appends bos/eos
        # if we don't have an eos/bos token in the tokenizer, skip
        if tokenizer.bos_token_id is None:
            enforce_bos = False
        if tokenizer.eos_token_id is None:
            enforce_eos = False

        # HF's BPE-based tokenizers do not, but the bert and roberta ones do
        # TODO: this doesn't necessarily ensure it, I guess, but eh
        if enforce_eos or enforce_bos:
            input_ids = tokenizer("hi there")["input_ids"]
            should_append_eos = input_ids[-1] != tokenizer.eos_token_id and enforce_eos
            should_append_bos = input_ids[0] != tokenizer.bos_token_id and enforce_bos
        else:
            should_append_eos = False
            should_append_bos = False

        self._need_to_add_eos = should_append_eos
        self._need_to_add_bos = should_append_bos
        self._workaround_len = _workaround_len

    def __call__(self, batch: Sequence[str]) -> list[dict]:
        if self._need_to_add_bos:
            batch = [self.tokenizer.bos_token + " " + d for d in batch]

        if self._need_to_add_eos:
            batch = [d + " " + self.tokenizer.eos_token for d in batch]

        if self._needs_long_sequence_workaround:
            batch, needs_merge = self._break_for_long_sequences(batch)
        else:
            needs_merge = []

        if self.padding is not False:
            encoding = self.tokenizer(
                batch,
                return_attention_mask=self.return_attention_mask,
                verbose=False,
                padding=self.padding,
                max_length=self.max_length,
                truncation=True,
            )  # type: ignore
        else:
            encoding = self.tokenizer(
                batch, return_attention_mask=self.return_attention_mask, verbose=False
            )  # type: ignore

        if needs_merge:
            new_encoding = self._merge_split_encodings(batch, encoding, needs_merge)
            encoding = BatchEncoding(new_encoding)

        # debatch the encoding
        unbatched = [dict(zip(encoding, t)) for t in zip(*[encoding[k] for k in encoding])]

        return unbatched

    def _break_for_long_sequences(self, batch):
        orig_lengths = [len(d) for d in batch]
        # break any strings that are longer than LONG_STRING_WORKAROUND characters into smaller chunks
        orig_batch = batch
        batch = []
        needs_merge = []
        for i, d in enumerate(orig_batch):
            needs_merge.append(False)
            orig_len = orig_lengths[i]
            while len(d) > self._workaround_len:
                # we'd rather break strings at whitespace, so find the first whitespace
                match = ws.search(d, self._workaround_len)
                # this is vanishingly unlikely, but if we can't find a whitespace, just break it at the limit
                if match is None:
                    split = len(d)
                else:
                    split = match.start()

                batch.append(d[:split])
                needs_merge.append(True)

                d = d[split:]
                orig_len -= split

            batch.append(d)
        return batch, needs_merge

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": len(self.tokenizer),
            "return_attention_mask": self.return_attention_mask,
            "padding": self.padding,
            "max_length": self.max_length,
            "append_bos": self._need_to_add_bos,
            "append_eos": self._need_to_add_eos,
        }

    @property
    def output_exemplar(self) -> dict:
        return dict(**self.tokenizer("hi there", return_attention_mask=self.return_attention_mask, verbose=False))

    @property
    def name_or_path(self):
        return self.tokenizer.name_or_path

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @staticmethod
    def _merge_split_encodings(batch, encoding, needs_merge):
        # merge the encodings back together
        # we might need to merge multiple encodings together
        # needs merge marks the first n-1 encodings that need to be merged for each document
        new_encoding = {}
        for k, v in encoding.items():
            if len(v) == 0:
                continue
            if isinstance(v[0], np.ndarray):
                assert len(v) == len(batch)
                v_out = []
                vs_to_merge = []
                for i in range(len(batch)):
                    if not needs_merge[i]:
                        v_out.append(np.concatenate(vs_to_merge))
                        vs_to_merge = []
                    vs_to_merge.append(v[i])

                if len(vs_to_merge) > 0:
                    v_out.append(np.concatenate(vs_to_merge))

                new_encoding[k] = v_out
            elif isinstance(v[0], list):
                v_out = []
                vs_to_merge = []
                for i in range(len(batch)):
                    if not needs_merge[i]:
                        if len(vs_to_merge) > 0:
                            v_out.append(list(chain(*vs_to_merge)))
                        vs_to_merge = []
                    vs_to_merge.append(v[i])

                if len(vs_to_merge) > 0:
                    v_out.append(list(chain(*vs_to_merge)))
                new_encoding[k] = v_out
            else:
                raise ValueError(f"Unknown type {type(v[0])}")
        return new_encoding

    # TODO remove this when it's resolved https://github.com/huggingface/tokenizers/issues/1495
    @cached_property
    def _needs_long_sequence_workaround(self):
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            normalizer = self.tokenizer.backend_tokenizer.normalizer
            if normalizer is None:
                return False
            # if there's a "Replace" normalizer, then we need to do the workaround
            # inexplicably there's no way to see inside a Sequence so we also have to assume it needs it
            return isinstance(normalizer, (normalizers.Replace, normalizers.Sequence))
        else:
            return False

    @property
    def num_cpus(self) -> int:
        if self.override_resources is not None:
            cpus = self.override_resources.get("num_cpus", None)
            if cpus is not None:
                return cpus
        return num_cpus_used_by_tokenizer(self.tokenizer)

    @property
    def num_gpus(self) -> int:
        if self.override_resources is not None:
            return self.override_resources.get("num_gpus", 0)
        return 0


def concatenate_and_group_texts(
    encoding: BatchEncoding,
    seq_len: int,
    stride: Optional[int] = None,
    drop_remainder: bool = True,
    mask_stride_overlap=True,
) -> Iterator[BatchEncoding]:
    """Groups texts in a batch together. Typically, you'll want to use this with a fairly large
    set of texts, e.g. 1000 docs.

    You should set mask_stride_overlap to True and drop_remainder to False if you want to use this for test data

    Args:
        encoding: The batch of texts to concatenate and group.
        seq_len: The max length of sequences to emit
        stride: The stride to use when grouping texts. If None, then the stride is set to seq_len.
        mask_stride_overlap: Whether to mask out overlapping tokens if we're using a stride.
        drop_remainder: Whether to drop the last batch if it's not a multiple of the seq_len.

    Returns:
        An iterator of tokenized texts, one at a time.
    """
    concatenated = BatchEncoding(data={k: np.array(list(chain(*v))) for k, v in encoding.items()})
    total_length = len(concatenated.input_ids)
    stride = stride or seq_len

    # Drop the "very last" bit of the dataset that doesn't fit into block size...
    if drop_remainder and total_length % stride != 0:
        total_length = ((total_length - seq_len + stride) // stride) * stride

    # Split by Chunks of Maximum Length
    # we want to take chunks up until we've covered all "total_length" tokens with a sliding window of size "stride"
    for begin in range(0, total_length - seq_len + stride, stride):
        data = {k: v[begin : begin + seq_len] for k, v in concatenated.items()}

        if mask_stride_overlap and stride != seq_len:
            labels = data.get("labels", data["input_ids"])
            if begin != 0:
                labels = _mask_overlap(labels, seq_len, stride)
            data["labels"] = labels

        yield BatchEncoding(data=data)


# -100 is pytorch's label mask
def _mask_overlap(labels, target_len, stride, sentinel=-100):
    """Masks out overlapping tokens in a sequence when we're using a stride."""
    labels = copy.deepcopy(labels)
    if isinstance(labels, list):
        for i in range(target_len - stride):
            if i < len(labels):
                labels[i] = sentinel
    else:
        labels[0 : target_len - stride] = sentinel

    return labels


def _stack_batch_encodings(a: BatchEncoding, b: BatchEncoding) -> BatchEncoding:
    """Stacks two batch encodings together, assuming that the keys are the same."""

    def _ensure_batched(x):
        if len(x) == 0:
            return list(x)
        elif isinstance(x[0], Sequence) or isinstance(x[0], np.ndarray):
            return list(x)
        else:
            return [x]

    return BatchEncoding({k: _ensure_batched(a[k]) + _ensure_batched(b[k]) for k in a.keys()})


@dataclass
class LMDatasetSourceConfig:
    """This class represents a dataset source with URLs or hf name/id."""

    tags: Optional[List[str]] = None
    """tags for the dataset. Typically the name of the dataset in the config will be added as a tag as well"""

    id: Optional[str] = None  # id (or path) for hf dataset
    name: Optional[str] = None  # name for hf dataset

    plaintext: bool = False
    stream: bool = True  # whether to use streaming when doing hf
    text_key: str = "text"  # key for the text field in the jsonl file or hf dataset

    train_urls: List[str] = ()  # type: ignore
    validation_urls: List[str] = ()  # type:ignore
    cache_dir: Optional[str] = None  # Optionally override the cache dir for this component

    def get_shard_source(self, split) -> Optional[ShardedDataSource[str]]:
        if self.id is not None:
            try:
                ds = WrappedHFDataSource(self.id, split=split, name=self.name, streaming=self.stream)
            except ValueError as e:
                # if the message starts with Bad split, then just return None
                if str(e).startswith("Bad split"):
                    logger.warning(f"Splits {split} not found for {self.id} {self.name}")
                    return None
                else:
                    raise

            if len(ds.shard_names) == 0:
                return None

            return ds.map(lambda x: x[self.text_key])
        else:
            split_urls = self.urls_for_split(split)
            if len(split_urls) == 0:
                return None
            return TextUrlDataSource(split_urls, self.text_key)

    def doc_iterator(self, split: str):
        if self.id is not None:
            dataset = datasets.load_dataset(self.id, name=self.name, streaming=self.stream)
            data = dataset[split]
            for doc in data:
                yield doc[self.text_key]
        else:
            urls = self.urls_for_split(split)

            yield from TextUrlDataSource(urls, self.text_key)

    def urls_for_split(self, split):
        if split == "train":
            urls = self.train_urls
        elif split == "validation":
            urls = self.validation_urls
        else:
            raise ValueError(f"Unknown split {split}")

        urls = [globbed for url in urls for globbed in expand_glob(url)]
        return urls


@dataclass
class LMTaskConfig(abc.ABC):
    tokenizer: str = "gpt2"
    vocab_size: Optional[int] = None  # if using the passthrough tokenizer, this is required

    # config related to caching
    cache_dir: Optional[str] = "cache/"
    cache_options: CacheOptions = field(default_factory=CacheOptions)
    enforce_eos: bool = True  # whether to append eos even if the tokenizer doesn't

    ignore_token_id: Optional[int] = None
    shuffle: bool | int = False
    """whether to shuffle the dataset. True means shuffle the whole dataset, False means don't shuffle.
    If you want to shuffle in eras, set this to the era length"""

    @cached_property
    def the_tokenizer(self) -> HfTokenizer:
        if self.tokenizer == "passthrough":
            return PassthroughTokenizer(self.vocab_size)
        else:
            return load_tokenizer(self.tokenizer)

    @abc.abstractmethod
    def train_set(
        self,
        seq_len: int,
        monitors: Union[bool, List[MetricsMonitor]] = True,
        *,
        key: Optional[PRNGKeyArray],
        epochs: Optional[int] = None,
    ) -> AsyncDataset[np.ndarray]:
        pass

    @abc.abstractmethod
    def validation_sets(
        self, seq_len: int, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Mapping[str, AsyncDataset[np.ndarray]]:
        pass

    @property
    @abc.abstractmethod
    def sources(self) -> Mapping[str, LMDatasetSourceConfig]:
        pass

    def tagged_eval_sets(
        self, seq_len: int, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> list[Tuple[AsyncDataset[np.ndarray], List[str]]]:
        tags = {name: (config.tags or []) + [name] for name, config in self.sources.items()}
        eval_sets = self.validation_sets(seq_len, monitors)

        return [(eval_sets[name], tags[name]) for name in eval_sets]


CANONICAL_INPUT_FIELD = "prompt"
CANONICAL_OUTPUT_FIELD = "response"


class SupervisedSourceConfigBase(Protocol):
    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[dict]]:
        raise NotImplementedError

    input_field: str
    output_field: str
    tags: Optional[List[str]]
    cache_dir: str


@dataclass
class LMSupervisedDatasetConfig(SupervisedSourceConfigBase):
    """Config for supervised fine-tuning datasets"""

    cache_dir: str = "cache/"

    # HF dataset config
    hf_dataset_name: Optional[str] = None  # e.g. "tatsu-lab/alpaca" or "OpenAssistant/oasst1"
    hf_dataset_split: str = "train"  # which split to use

    # Local files config
    validation_urls: List[str] = field(default_factory=list)  # paths to jsonl/json files

    # Field names in the data
    input_field: str = CANONICAL_INPUT_FIELD  # name of the input field
    output_field: str = CANONICAL_OUTPUT_FIELD  # name of output field

    # Optional metadata
    tags: Optional[List[str]] = None

    def __post_init__(self):
        warnings.warn(
            "LMSupervisedDatasetConfig is deprecated. Use SupervisedHfSourceConfig or "
            "SupervisedUrlSourceConfig instead.",
            DeprecationWarning,
        )

    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[dict]]:
        if self.hf_dataset_name is not None:
            return WrappedHFDataSource(self.hf_dataset_name, split=self.hf_dataset_split)
        elif split != "validation":
            raise ValueError("Only validation split is supported for local files")
        else:
            urls = [globbed for url in self.validation_urls for globbed in expand_glob(url)]
            return JsonlDataSource(urls)


@dataclass(frozen=True)
class SupervisedHfSourceConfig(SupervisedSourceConfigBase):
    cache_dir: str
    id: str
    name: str | None = None

    streaming: bool = True

    input_field: str = CANONICAL_INPUT_FIELD
    output_field: str = CANONICAL_OUTPUT_FIELD
    tags: Optional[List[str]] = None

    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[dict]]:
        return WrappedHFDataSource(self.id, split=split, name=self.name, streaming=self.streaming).map(
            lambda x: {CANONICAL_INPUT_FIELD: x[self.input_field], CANONICAL_OUTPUT_FIELD: x[self.output_field]}
        )


@dataclass(frozen=True)
class SupervisedUrlSourceConfig(SupervisedSourceConfigBase):
    cache_dir: str
    train_urls: list[str] = dataclasses.field(default_factory=list)
    validation_urls: list[str] = dataclasses.field(default_factory=list)

    input_field: str = CANONICAL_INPUT_FIELD
    output_field: str = CANONICAL_OUTPUT_FIELD
    tags: Optional[List[str]] = None

    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[dict]]:
        urls = self.train_urls if split == "train" else self.validation_urls
        if not urls:
            return None

        urls = [globbed for url in urls for globbed in expand_glob(url)]

        source = UrlDataSource(urls, columns=[self.input_field, self.output_field])
        return source.map(
            lambda x: {CANONICAL_INPUT_FIELD: x[self.input_field], CANONICAL_OUTPUT_FIELD: x[self.output_field]}
        )


SupervisedSourceConfig: TypeAlias = Union[SupervisedHfSourceConfig, SupervisedUrlSourceConfig]


def _preprocess_supervised_example(
    batch, tokenizer: PreTrainedTokenizerBase, input_field: str, output_field: str
) -> dict:
    sources = [example[input_field] for example in batch]

    targets = [example[output_field] for example in batch]
    # TODO: this seems pretty wasteful since you end up tokenizing twice, but it's how alpaca does it.
    examples = [s + t for s, t in zip(sources, targets)]
    sources_tokenized = tokenizer(sources, padding=False, truncation=True)
    examples_tokenized = tokenizer(examples, padding=False, truncation=True)

    source_lens = [len(s) for s in sources_tokenized["input_ids"]]

    return {
        "input_ids": [np.array(example, dtype=np.int32) for example in examples_tokenized["input_ids"]],
        "sources_len": np.array(source_lens, dtype=np.int32),
    }


def _prepare_supervised_examples(ex: list[dict], tokenizer: PreTrainedTokenizerBase, Pos: hax.Axis) -> list[LmExample]:
    """
    Prepare examples for training. This function converts the (cached) encodings into an LmExample.

    It goes through the following steps:

    1. Pad the batch to the maximum length.
    2. Mask out the input and prompt if requested.
    3. Create an LmExample with the input_ids as the input and the next token as the target.
    """
    lens = np.array([ex["sources_len"] for ex in ex])

    ex_pad = tokenizer.pad(
        ex,
        padding="max_length",
        max_length=Pos.size,
    )

    input_ids = ex_pad["input_ids"]
    truncated = [ids[-Pos.size :] for ids in input_ids]

    out = []
    for ids, len in zip(truncated, lens):
        causal = _mk_sup_example_jit(Pos, hax.named(ids, Pos), len, tokenizer.pad_token_id, tokenizer.eos_token_id)

        out.append(causal)

    return out


@functools.partial(jax.jit, static_argnums=(0, 3, 4))
def _mk_sup_example_jit(Pos, input_ids: hax.NamedArray, sources_len, pad_token_id, eos_id):
    # mask out padding and anything before the start of the target
    loss_mask = hax.arange(Pos) >= sources_len - 1
    # don't predict the padding
    targets = hax.roll(input_ids, -1, Pos)
    loss_mask = loss_mask & (targets != pad_token_id)
    loss_mask = loss_mask & (1 - hax.nn.one_hot(-1, Pos, dtype=jax.numpy.bool_))
    return LmExample.causal(input_ids, loss_mask=loss_mask, eos_id=eos_id)


def mk_supervised_datasets(
    sources: Mapping[str, SupervisedSourceConfigBase] | SupervisedSourceConfigBase,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    Pos: hax.Axis,
) -> dict[str, tuple[AsyncDataset[LmExample], Sequence[str]]]:
    """
    Create supervised datasets from a set of sources.

    Returns:
        A dictionary of dataset names to tuples of the dataset and the tags associated with the dataset.
    """
    out: dict[str, tuple[AsyncDataset[LmExample], Sequence[str]]] = {}

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if isinstance(sources, Mapping):
        for name, config in sources.items():
            source = config.get_shard_source(split)
            if source is None:
                continue

            ds = _cache_supervised_set(
                source, config.cache_dir, tokenizer, Pos, config.input_field, config.output_field
            )

            if config.tags is None:
                tags = [name]
            else:
                tags = config.tags + [name]

            out[name] = (ds, tags)
    else:
        source = sources.get_shard_source(split)  # type: ignore
        if source is not None:
            ds = _cache_supervised_set(
                source, sources.cache_dir, tokenizer, Pos, sources.input_field, sources.output_field
            )
            tags = sources.tags or []
            if isinstance(sources, SupervisedHfSourceConfig):
                name = sources.id
                if sources.name is not None:
                    name = f"{name}/{sources.name}"

                tags = [name] + tags
            else:
                name = "default"
            out[name] = (ds, tags)

    return out


def mk_supervised_dataset(
    config: SupervisedSourceConfigBase, split: str, tokenizer: HfTokenizer, Pos: hax.Axis
) -> AsyncDataset[LmExample]:

    source = config.get_shard_source(split)

    output_exemplar = {"input_ids": np.zeros((0,), dtype=np.int32), "sources_len": np.zeros((0,), dtype=np.int32)}

    dataset = source.map_batches(  # type: ignore
        lambda ex: _preprocess_supervised_example(ex, tokenizer, config.input_field, config.output_field),
        batch_size=128,
        num_cpus=num_cpus_used_by_tokenizer(tokenizer),
        output_exemplar=output_exemplar,
    )

    cached_dataset: AsyncDataset[dict] = dataset.build_or_load_cache(config.cache_dir, await_finished=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return cached_dataset.map_batches(lambda ex: _prepare_supervised_examples(ex, tokenizer, Pos))


def _cache_supervised_set(source, cache_dir, tokenizer, Pos, input_field, output_field):
    """
    Cache a supervised dataset into input_ids and sources_len.
    """
    output_exemplar = {"input_ids": np.zeros((0,), dtype=np.int32), "sources_len": np.zeros((0,), dtype=np.int32)}
    dataset = source.map_batches(
        lambda ex: _preprocess_supervised_example(ex, tokenizer, input_field, output_field),
        batch_size=128,
        num_cpus=num_cpus_used_by_tokenizer(tokenizer),
        output_exemplar=output_exemplar,
    )
    cached_dataset: AsyncDataset[dict] = dataset.build_or_load_cache(cache_dir, await_finished=True)
    ds = cached_dataset.map_batches(lambda ex: _prepare_supervised_examples(ex, tokenizer, Pos))
    return ds


@dataclass(frozen=True)
class ChatUrlDataSourceConfig:
    """Config for loading JSONL files in OpenAI chat format for supervised fine-tuning."""

    cache_dir: str
    train_urls: List[str] = field(default_factory=list)
    validation_urls: List[str] = field(default_factory=list)

    # Chat format specific fields
    messages_field: str = "messages"
    input_role: str = "user"
    output_role: str = "assistant"

    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[dict]]:
        """Gets ShardedDataSource for either training or validation data."""
        urls = self.validation_urls if split == "validation" else self.train_urls

        if not urls:
            return None

        # Use the datasource_from_chat_jsonl function from sharded_datasource
        return datasource_from_chat_jsonl(
            urls, messages_field=self.messages_field, input_role=self.input_role, output_role=self.output_role
        )


def preprocess_chat_example(batch, tokenizer: PreTrainedTokenizerBase, should_append_eos: bool) -> dict:
    """
    Preprocess chat examples to match the format of preprocess_supervised_example.
    Returns a dict with input_ids and sources_len like the supervised case.

    Args:
        batch: List of dicts with input/output pairs
        tokenizer: HuggingFace tokenizer
        should_append_eos: Whether we need to manually add EOS (True if tokenizer doesn't do it automatically)
    """
    # Get sources (inputs) and targets (outputs) from the batch
    sources = [example["input"] for example in batch]
    targets = [example["output"] for example in batch]

    # Add EOS only if needed (tokenizer doesn't do it automatically)
    if should_append_eos:
        targets = [t + tokenizer.eos_token for t in targets]

    # Tokenize sources alone first to get the source lengths
    sources_tokenized = tokenizer(sources, padding=False, truncation=True)

    # Combine source and target for full examples
    full_examples = [f"{s}{t}" for s, t in zip(sources, targets)]
    examples_tokenized = tokenizer(full_examples, padding=False, truncation=True)

    # Get source lengths to mask loss appropriately
    source_lens = [len(s) for s in sources_tokenized["input_ids"]]

    return {
        "input_ids": [np.array(example, dtype=np.int32) for example in examples_tokenized["input_ids"]],
        "sources_len": np.array(source_lens, dtype=np.int32),
    }


def mk_chat_sft_dataset(
    config: ChatUrlDataSourceConfig, tokenizer: PreTrainedTokenizerBase, Pos: hax.Axis
) -> AsyncDataset[LmExample]:
    """Creates a dataset from JSONL files containing chat format data for SFT."""
    source = config.get_shard_source("train")
    if source is None:
        raise ValueError("No training data source found")

    # Set up example structure matching supervised case
    output_exemplar = {"input_ids": np.zeros((0,), dtype=np.int32), "sources_len": np.zeros((0,), dtype=np.int32)}

    input_ids = tokenizer("hi there")["input_ids"]
    should_append_eos = input_ids[-1] != tokenizer.eos_token_id
    logger.info(f"Manual EOS Needed: {should_append_eos}")

    # Process the dataset
    dataset = source.map_batches(
        lambda ex: preprocess_chat_example(ex, tokenizer, should_append_eos),
        batch_size=128,
        num_cpus=num_cpus_used_by_tokenizer(tokenizer),
        output_exemplar=output_exemplar,
    )

    # Cache the processed data
    cached_dataset: AsyncDataset[dict] = dataset.build_or_load_cache(config.cache_dir, await_finished=True)

    # Ensure padding token is set (needed by _prepare_supervised_example)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Reuse the supervised prepare function directly
    return cached_dataset.map_batches(lambda ex: _prepare_supervised_examples(ex, tokenizer, Pos))


@dataclass
class LMDatasetConfig(LMDatasetSourceConfig, LMTaskConfig):
    """This class supports loading data both from HF Datasets and from a raw dataset of jsonl urls"""

    cache_dir: Optional[str] = "cache/"

    def train_set(
        self,
        seq_len: int,
        monitors: Union[bool, List[MetricsMonitor]] = True,
        *,
        key: Optional[PRNGKeyArray] = None,
        epochs: Optional[int] = None,
    ) -> AsyncDataset[np.ndarray]:

        ds: AsyncDataset[np.ndarray] | None = self.token_seq_dataset("train", seq_len, monitors)

        # add epoch flag here.
        if ds is None:
            raise ValueError("No training set!")

        if epochs:
            logger.info("Wrapping dataset in epoch dataset")
            ds = EpochDataset(ds, max_epochs=epochs)

        if self.shuffle is True:
            ds = ds.shuffle(key)
        elif isinstance(self.shuffle, int) and self.shuffle > 0:
            ds = ds.era_shuffle(self.shuffle, key=key)

        return ds  # type: ignore

    def validation_set(
        self, seq_len: int, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Optional[TokenSeqDataset]:
        return self.token_seq_dataset("validation", seq_len, monitors)

    def validation_sets(
        self, seq_len: int, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Mapping[str, AsyncDataset[np.ndarray]]:
        validation_set = self.validation_set(seq_len, monitors)
        if validation_set is not None:
            return {"": validation_set}
        else:
            return {}

    @property
    def sources(self) -> Mapping[str, LMDatasetSourceConfig]:
        return {"": self}

    @cached_property
    def _has_validation_set(self):
        if len(self.validation_urls) > 0:
            return True

        if self.id is not None:
            dataset = datasets.load_dataset(self.id, name=self.name, streaming=self.stream, split="validation")
            try:
                next(iter(dataset))
                return True
            except StopIteration:
                return False

        return False

    def token_seq_dataset(
        self, split: str, seq_len: int, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Optional[TokenSeqDataset]:
        cache = self.build_or_load_cache(split, monitors=monitors)
        if cache is None:
            return None
        return TokenSeqDataset(cache, seq_len)

    def build_or_load_cache(
        self, split: str, monitors: Union[bool, List[MetricsMonitor]] = True, logger_name: Optional[str] = None
    ) -> Optional[TreeCache[BatchEncoding]]:
        if self.cache_dir is None:
            raise ValueError("cache_dir cannot be None")

        split_cache_dir = os.path.join(self.cache_dir, split)
        name = logger_name or os.path.basename(self.cache_dir)

        try:
            # TODO: pass in options
            return TreeCache.load(split_cache_dir, exemplar={"input_ids": np.zeros(0, dtype=np.int32)})
        except FileNotFoundError:
            pass

        source = self.get_shard_source(split)
        if source is None:
            logger.info(f"No data for {split}")
            return None

        logger.info(f"Building cache for {split}...")

        if monitors is True:
            monitors = [
                LoggingMetricsMonitor(prefix=f"preprocessing/{name}/{split}", commit=False),
                LoggerMetricsMonitor(f"preprocessing.{name}.{split}"),
            ]
        elif monitors is False:
            monitors = []

        bt = BatchTokenizer(self.the_tokenizer, enforce_bos=True, enforce_eos=self.enforce_eos)

        return build_or_load_cache(
            split_cache_dir,
            source,
            bt,
            monitors=monitors,
            await_finished=False,
            options=self.cache_options,
            split=split,
        )


class PassthroughTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_size, **kwargs):
        self._vocab = {i: i for i in range(vocab_size)}
        self._vocab_size = vocab_size
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self):
        return self._vocab

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, ...]:
        return ()

    def _tokenize(self, text, **kwargs):
        tokens = np.fromstring(text, dtype=int, sep=" ")
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)


@dataclass
class LMMixtureDatasetConfig(LMTaskConfig):
    """This class represents a mixture of datasets with their associated weights."""

    cache_dir: Optional[str] = "cache/"

    # data source configs and weights
    configs: Dict[str, LMDatasetSourceConfig] = field(default_factory=dict)
    """ configuration of each dataset source (urls, hf dataset id, etc.) """
    train_weights: Dict[str, float] = field(default_factory=dict)
    """ weights for each dataset source. They will be normalized to sum to 1. """
    stop_strategy: str = field(default=StopStrategy.RESTART_STRATEGY)
    mixture_block_size: int = 2048
    """ block size for the mixture dataset."""

    def __post_init__(self):
        if len(self.configs) == 0:
            raise ValueError("At least one dataset must be provided")

        if set(self.configs.keys()) != set(self.train_weights.keys()):
            raise ValueError(
                f"The keys in configs and weights must be the same;got {self.configs.keys()} and"
                f" {self.train_weights.keys()}"
            )

    def train_set(
        self,
        seq_len: int,
        monitors: Union[bool, List[MetricsMonitor]] = True,
        *,
        key: Optional[PRNGKeyArray],
        epochs: Optional[int] = None,
    ) -> AsyncDataset[np.ndarray]:
        doc_caches = self.build_caches("train", monitors=monitors)
        token_datasets = {name: TokenSeqDataset(cache, seq_len) for name, cache in doc_caches.items()}

        if epochs:
            raise ValueError("Epochs are not supported for mixture datasets")

        if key is None:
            key = jax.random.PRNGKey(0)

        mix_key, shuffle_key = jax.random.split(key)

        # We shuffle the components and not the overall mixture because this lets us preserve
        # the "stable batch" property of the mixture dataset.
        def shuffle_ds(ds, key):
            if self.shuffle is True:
                ds = ds.shuffle(key)
            elif isinstance(self.shuffle, int):
                ds = ds.era_shuffle(self.shuffle, key=key)

            return ds

        if self.shuffle:
            out_token_datasets = {}
            key_iter = key_iterator(shuffle_key)
            for name, ds in token_datasets.items():
                out_token_datasets[name] = shuffle_ds(ds, next(key_iter))
            token_datasets = out_token_datasets

        mixture = MixtureDataset(
            datasets=token_datasets,
            weights=self.train_weights,
            stop_strategy=self.stop_strategy,
            key=mix_key,
            block_size=2048,
        )

        return mixture

    def training_sets(
        self, seq_len: int, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Mapping[str, TokenSeqDataset]:
        doc_caches = self.build_caches("train", monitors=monitors)
        token_datasets = {name: TokenSeqDataset(cache, seq_len) for name, cache in doc_caches.items()}
        return token_datasets

    def validation_sets(
        self, seq_len: int, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Mapping[str, AsyncDataset[np.ndarray]]:
        doc_caches = self.build_caches("validation", monitors=monitors)
        token_datasets = {name: TokenSeqDataset(cache, seq_len) for name, cache in doc_caches.items()}
        return token_datasets

    def build_caches(
        self, split: str, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Dict[str, TreeCache[dict]]:
        # this is a bit gross, but we want to forward all "Task" config fields to the LMDatasetConfig for building.
        # We do this by just grabbing all the fields from the LMDatasetConfig and forwarding them to the
        # LMDatasetConfig.build_or_load_cache method. We exclude the cache_dir field.
        task_config_fields = set(x.name for x in dataclasses.fields(LMTaskConfig))
        task_config_dict = {k: v for k, v in self.__dict__.items() if k in task_config_fields and k != "cache_dir"}

        caches = {}
        for name, source_config in self.configs.items():
            weight = self.train_weights.get(name, 0)

            if weight == 0 and split == "train":
                continue

            source_config_dict = dict(**source_config.__dict__)

            if source_config.cache_dir is None:
                # replace with the main cache dir/{name}
                if self.cache_dir is None:
                    raise ValueError(
                        "If the 'main' cache_dir is None, then all component cache_dirs must be non-None, but"
                        f"{name}'s cache_dir is None."
                    )
                cache_dir = os.path.join(self.cache_dir, name)
                source_config_dict["cache_dir"] = cache_dir

            dataset = LMDatasetConfig(
                **source_config_dict,
                **task_config_dict,
            )
            cache = dataset.build_or_load_cache(split, monitors)
            # drop the data source and corresponding weight if the cache is not built
            if cache is None:
                logger.warning(f"Skipping {name} for split {split} because no source was provided")
            else:
                caches[name] = cache

        # in practice it works best if we block on validation caches
        if split == "validation":
            for cache in caches.values():
                cache.await_finished()

        else:
            logger.info(f"Not waiting for {split} caches to finish building")

        return caches

    @property
    def sources(self) -> Mapping[str, LMDatasetSourceConfig]:
        return self.configs


def datasource_from_chat_jsonl(
    urls: Sequence[str], messages_field: str = "messages", input_role: str = "user", output_role: str = "assistant"
) -> "ShardedDataSource[dict]":
    """Creates a ShardedDataSource from JSONL files containing chat messages.

    Args:
        urls: Sequence of URLs or glob patterns pointing to JSONL files
        messages_field: Field name containing the messages in each JSON object
        input_role: Role identifier for input messages
        output_role: Role identifier for output messages

    Returns:
        ShardedDataSource configured for chat data
    """
    # Expand any glob patterns in the URLs
    expanded_urls = []
    for url in urls:
        if any(c in url for c in "*?[]"):
            expanded_urls.extend(gcs_glob(url))
        else:
            expanded_urls.append(url)

    return ChatJsonlDataSource(expanded_urls, messages_field, input_role, output_role)


# TODO: switch to actual multi-turn
class ChatJsonlDataSource(JsonlDataSource):
    """DataSource that reads JSONL files containing OpenAI chat format messages."""

    def __init__(self, urls: Sequence[str], messages_field: str, input_role: str, output_role: str):
        super().__init__(urls)
        self.messages_field = messages_field
        self.input_role = input_role
        self.output_role = output_role

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        url = self._shard_name_to_url_mapping[shard_name]
        i = 0
        with fsspec.open(url, "r", compression="infer") as f:
            for line in f:
                if i >= row:
                    data = json.loads(line)
                    messages = data[self.messages_field]

                    # Extract input/output from messages
                    input_msg = next(m["content"] for m in messages if m["role"] == self.input_role)
                    output_msg = next(m["content"] for m in messages if m["role"] == self.output_role)

                    yield {"input": input_msg, "output": output_msg}
                i += 1


CANONICAL_REPO_NAME_FIELD = "repo_name"
CANONICAL_FILES_FIELD = "files"
CANONICAL_FILE_PATH_FIELD = "path"
CANONICAL_FILE_CONTENT_FIELD = "content"
CANONICAL_ID_FIELD = "blob_id"


@dataclass(frozen=True)
class FIMUrlSourceConfig:
    cache_dir: str = "cache/"
    tokenizer: str = "Qwen/Qwen2.5-Coder-1.5B"
    train_urls: list[str] = dataclasses.field(default_factory=list)
    validation_urls: list[str] = dataclasses.field(default_factory=list)
    cache_options: CacheOptions = field(default_factory=CacheOptions)
    shuffle: bool | int = True
    pack: bool = False
    add_router_token: bool = True
    predict_prefix: bool = False
    predict_router_token: bool = False
    predict_fim_token: bool = False

    data_format: str = "flattened"
    repo_level_percentage = 0.0
    repo_name_field: str = CANONICAL_REPO_NAME_FIELD
    files_field: str = CANONICAL_FILES_FIELD
    file_path_field: str = CANONICAL_FILE_PATH_FIELD
    file_content_field: str = CANONICAL_FILE_CONTENT_FIELD
    id_field: str = CANONICAL_ID_FIELD

    prefix_token: str = "<|fim_prefix|>"
    middle_token: str = "<|fim_middle|>"
    suffix_token: str = "<|fim_suffix|>"
    repo_name_token: str = "<|repo_name|>"
    file_sep_token: str = "<|file_sep|>"
    eos_token: str = "<|endoftext|>"
    router_token: str = "<|router|>"
    """The overall prompt format is:
    <|repo_name|>repo name\n
    <|file_sep|>filename.py\n
    mycode = x
    <|file_sep|>target_file.py\n
    <|fim_prefix|>all of the code goes here now except for<|fim_suffix|>which is pretty neat<|fim_middle|>the middle
    """

    def __post_init__(self):
        if not self.add_router_token:
            assert not self.predict_router_token, "Can't predict router token if it's not in the data"
        assert self.data_format in ["flattened", "repo_level"]

    def get_flattened_source(self, split: str) -> Optional[ShardedDataSource[str]]:
        urls = self.train_urls if split == "train" else self.validation_urls
        if not urls:
            return None

        urls = [globbed for url in urls for globbed in expand_glob(url)]
        source = UrlDataSource(
            urls, columns=[self.repo_name_field, self.id_field, self.file_content_field, self.file_path_field]
        )

        def make_entry(x) -> str:
            hash_input = x[self.id_field]
            rand = random.Random(hash(hash_input))
            content = self.replace_illegal_chars(x[self.file_content_field])
            i0 = rand.randint(0, len(content) - 1)
            i1 = rand.randint(0, len(content) - 1)
            while i1 == i0:
                i1 = rand.randint(0, len(content) - 1)
            i0, i1 = min(i0, i1), max(i0, i1) + 1
            prefix = content[:i0]
            middle = content[i0:i1]
            suffix = content[i1:]
            to_join = []
            to_join.extend([self.prefix_token, prefix, self.suffix_token, suffix])
            if self.add_router_token:
                to_join.extend([self.router_token])
            to_join.extend([self.middle_token, middle, self.eos_token])
            return "".join(to_join)

        return source.map(make_entry)

    def get_repo_level_source(self, split: str) -> Optional[ShardedDataSource[str]]:
        urls = self.train_urls if split == "train" else self.validation_urls
        if not urls:
            return None

        urls = [globbed for url in urls for globbed in expand_glob(url)]

        source = UrlDataSource(urls, columns=[self.repo_name_field, self.files_field])

        def make_entry(x) -> str:
            files = x[self.files_field]
            assert len(files) > 0, f"No files found in {x[self.file_path_field]}"
            hash_input = files[0][self.id_field]
            rand = random.Random(hash(hash_input))
            is_repo_level = len(files) > 1 and (rand.random() < self.repo_level_percentage)
            file = files[rand.choice(range(len(files)))]
            content = self.replace_illegal_chars(file[self.file_content_field])
            file_len = len(content)
            i0 = rand.randint(0, file_len - 1)
            i1 = rand.randint(0, file_len - 1)
            while i1 == i0:
                i1 = rand.randint(0, file_len - 1)
            i0, i1 = min(i0, i1), max(i0, i1) + 1
            prefix = content[:i0]
            middle = content[i0:i1]
            suffix = content[i1:]
            to_join = []
            if is_repo_level:
                to_join = [self.repo_name_token, x[self.repo_name_field], "\n"]
                for f in files:
                    to_join.extend(
                        [
                            self.file_sep_token,
                            f[self.file_path_field],
                            "\n",
                            self.replace_illegal_chars(f[self.file_content_field]),
                        ]
                    )

            to_join.extend([self.prefix_token, prefix, self.suffix_token, suffix])
            if self.add_router_token:
                to_join.extend([self.router_token])
            to_join.extend([self.middle_token, middle, self.eos_token])
            return "".join(to_join)

        return source.map(make_entry)

    @property
    def the_tokenizer(self) -> HfTokenizer:
        tokenizer = load_tokenizer(self.tokenizer)
        if self.add_router_token:
            tokenizer.add_special_tokens({"additional_special_tokens": [self.router_token]})
        pad_token = tokenizer.pad_token
        eos_token = tokenizer.eos_token
        if pad_token is None or pad_token == eos_token:
            pad_token = "<|padding|>"
            tokenizer.add_special_tokens({"pad_token": pad_token})
        return tokenizer

    @cached_property
    def _replacer(self) -> FastReplacer:
        return FastReplacer(
            {
                k: k[:2] + "cleaned_" + k[2:]
                for k in [
                    self.prefix_token,
                    self.middle_token,
                    self.suffix_token,
                    self.repo_name_token,
                    self.file_sep_token,
                    self.eos_token,
                    self.router_token,
                    "<|padding|>",
                ]
            }
        )

    def replace_illegal_chars(self, text: str) -> str:
        return self._replacer.replace(text)


def _preprocess_fim_example(
    batch: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    middle_token_id: int,
    max_model_len: int,
    pack: bool = False,
) -> dict:
    if max_model_len is None:
        max_model_len = tokenizer.model_max_length
    input_ids = tokenizer(batch, padding=False, truncation=True, max_length=max_model_len)
    input_ids = [np.array(ids, dtype=np.int32) for ids in input_ids["input_ids"]]
    assert input_ids[0].ndim == 1  # make sure it's not already batched
    # Sometimes long files have the middle token idx after the max_model_len
    input_ids = [ids for ids in input_ids if np.any(ids == middle_token_id)]
    if pack:
        concatted: List[np.array] = []
        to_concat: List[np.array] = []
        while len(input_ids) > 0:
            # This is valid since each element is at most `max_model_len`
            while len(input_ids) > 0 and (sum(len(ids) for ids in to_concat) + len(input_ids[0])) <= max_model_len:
                to_concat.append(input_ids.pop(0))
            concatted.append(np.concatenate(to_concat))
            to_concat = []
        return {"input_ids": concatted}
    else:
        return {"input_ids": input_ids}


# @functools.partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7, 8, 9, 10))
def _mk_fim_example_jit(
    Pos: hax.Axis,
    input_ids: hax.NamedArray,
    hs_idxs: hax.NamedArray,
    seq_len: hax.NamedArray,
    middle_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    router_token_id: Optional[int],
    has_router_token: bool,
    predict_prefix: bool,
    predict_router_token: bool,
    predict_fim_token: bool,
) -> RoutableLmExample:
    # loss_mask[i]=true => predict the (i+1)-th token from the i-th token
    loss_mask = ~hax.nn.one_hot(-1, Pos, dtype=jax.numpy.bool_)
    loss_mask &= input_ids != pad_token_id
    loss_mask &= input_ids != eos_token_id  # don't predict after eos
    completion_mask = hs_idxs != -1

    if has_router_token and router_token_id is not None and not predict_router_token:
        # don't make the previous token predict the "<|router|>" token
        # loss_mask &= ~hax.nn.one_hot(middle_idxs - 2, Pos, dtype=jax.numpy.bool_)
        loss_mask &= hax.roll(input_ids != router_token_id, -1, Pos)
    if not predict_fim_token:
        # don't make the previous token predict the <|fim_middle|> token
        loss_mask &= hax.roll(input_ids != middle_token_id, -1, Pos)
    if not predict_prefix:
        before_fim = hax.roll(input_ids == middle_token_id, -1, Pos)
        loss_mask &= completion_mask | before_fim
    return RoutableLmExample.causal_with_hs_idxs(
        input_ids,
        router_hs_idxs=hs_idxs,
        seq_length=seq_len,
        loss_mask=loss_mask,
        completion_mask=completion_mask,
        eos_id=eos_token_id,
        ignore_id=pad_token_id,
    )


def _prepare_fim_examples(
    ex: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    Pos: hax.Axis,
    middle_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    router_token_id: Optional[int],
    has_router_token: bool,
    predict_prefix: bool,
    predict_fim_token: bool,
    predict_router_token: bool,
) -> list[RoutableLmExample]:
    ex_pad = tokenizer.pad(
        ex,
        padding="max_length",
        max_length=Pos.size,
    )

    input_ids = ex_pad["input_ids"]
    assert all(len(ids) == Pos.size for ids in input_ids)

    hs_idx_selector = []
    seq_lens = []
    for seq_idx in range(input_ids.shape[0]):
        mids = np.argwhere(input_ids[seq_idx] == middle_token_id).flatten()
        ends = np.argwhere(input_ids[seq_idx] == eos_token_id).flatten()
        if len(ends) == len(mids) - 1:  # we might not be able to fit a single long sequence in
            ends = np.concat([ends, [Pos.size - 1]])
        if len(mids) != len(ends):
            # We've hit some broken sequence, abort
            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = os.path.join(tmpdir, "invalid_sequence.txt")
                with open(os.path.join(tmpdir, "invalid_sequence.txt"), "w") as f:
                    f.write(tokenizer.decode(input_ids[seq_idx]))
                levanter.tracker.current_tracker().log_artifact(out_path, name="invalid sequence")
                raise ValueError(f"Mismatched num middle and ends, {(mids, ends)}")

        hs_idx = -1 * np.ones(len(input_ids[seq_idx]), dtype=np.int32)
        seq_len_i = -1 * np.ones(len(input_ids[seq_idx]), dtype=np.int32)
        for m, e in zip(mids, ends):
            hs_idx[m:e] = m - 1  # we use the hidden state which predicts the fim token
            seq_len_i[m:e] = e - m
        hs_idx_selector.append(hs_idx)
        seq_lens.append(seq_len_i)

    out = []
    for ids, hs, seq_len_i in zip(input_ids, hs_idx_selector, seq_lens):
        causal: RoutableLmExample = _mk_fim_example_jit(
            Pos,
            hax.named(ids, Pos),
            hax.named(hs, Pos),
            hax.named(seq_len_i, Pos),
            middle_token_id=middle_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            router_token_id=router_token_id,
            has_router_token=has_router_token,
            predict_prefix=predict_prefix,
            predict_router_token=predict_router_token,
            predict_fim_token=predict_fim_token,
        )
        if causal.router_hs_idxs is not None and hax.any(causal.router_hs_idxs >= Pos.size).item():
            raise ValueError(
                f"Middle token index {causal.router_hs_idxs.max()} out of bounds for {Pos}. You need to regenerate"
                " your cache after changing seq_len."
            )

        out.append(causal)

    return out


def mk_fim_dataset(
    config: FIMUrlSourceConfig,
    split: str,
    tokenizer: HfTokenizer,
    Pos: hax.Axis,
    key: Optional[PRNGKeyArray] = None,
    await_finished: bool = True,
) -> AsyncDataset[RoutableLmExample]:

    if config.data_format == "flattened":
        source = config.get_flattened_source(split)
    elif config.data_format == "repo_level":
        source = config.get_repo_level_source(split)
    else:
        raise ValueError(f"Unknown data format {config.data_format}")

    output_exemplar = {
        "input_ids": np.zeros((0,), dtype=np.int32),
    }

    pad_token = tokenizer.pad_token
    eos_token = tokenizer.eos_token
    assert pad_token is not None and eos_token is not None
    assert pad_token != eos_token
    pad_token_id, eos_token_id, middle_token_id = tokenizer.convert_tokens_to_ids(
        [pad_token, eos_token, config.middle_token]
    )

    router_token_id = None
    if config.add_router_token:
        router_token_id = tokenizer.convert_tokens_to_ids(config.router_token)

    dataset = source.map_batches(  # type: ignore
        lambda ex: _preprocess_fim_example(ex, tokenizer, middle_token_id, max_model_len=Pos.size, pack=config.pack),
        batch_size=1024,
        num_cpus=num_cpus_used_by_tokenizer(tokenizer),
        output_exemplar=output_exemplar,
    )

    if config.add_router_token:
        split += "_with_router"
    if config.pack:
        split += "_packed"
    split_cache_dir = os.path.join(config.cache_dir, split)
    cached_dataset: AsyncDataset[dict] = dataset.build_or_load_cache(
        split_cache_dir, await_finished=await_finished, options=config.cache_options
    )

    ds = cached_dataset.map_batches(
        lambda ex: _prepare_fim_examples(
            ex,
            tokenizer,
            Pos,
            middle_token_id=middle_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            router_token_id=router_token_id,
            has_router_token=config.add_router_token,
            predict_prefix=config.predict_prefix,
            predict_fim_token=config.predict_fim_token,
            predict_router_token=config.predict_router_token,
        )
    )

    if config.shuffle is True:
        assert key is not None, "Key must be provided when shuffle=True"
        ds = ds.shuffle(key)
    elif isinstance(config.shuffle, int) and config.shuffle > 0:
        ds = ds.era_shuffle(config.shuffle, key=key)

    return ds  # type: ignore
