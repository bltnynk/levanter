from typing import Iterator, List, Optional, Tuple

import numpy as np

from levanter.data.dataset import ShardableDataset
from levanter.data.text import TokenizedDocumentCache, _stack_batch_encodings


def _lens(data):
    return [len(d) for d in data]


def yield_example2(
    data: list, seq_len: int, min_doc_length: int = 0
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:

    assert sum(_lens(data)) >= seq_len, "Data must be at least seq_len long"
    example_parts: List[np.ndarray] = []
    while sum(_lens(example_parts)) < seq_len:
        if len(data) == 0:
            break
        s = data.pop(0)
        if len(s) > min_doc_length:
            example_parts.append(s)

    start_inds = np.cumsum([0] + _lens(example_parts[:-1]))
    example = np.concatenate(example_parts)
    tail = None
    if len(example) > seq_len:
        example, tail = example[:seq_len], example[seq_len:]
    return start_inds, example, tail


def make_examples(data: List[np.ndarray], seq_len: int, min_doc_length: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    examples = []
    while sum(_lens(data)) >= seq_len:
        si, ex, tail = yield_example2(data, seq_len, min_doc_length)
        examples.append((si, ex))
        if tail is not None and len(tail) > min_doc_length:
            data.insert(0, tail)
    return examples


class TokenSeqWithDocBoundsDataset(ShardableDataset[Tuple[np.ndarray, np.ndarray]]):
    """
    A dataset that yields sequences of tokens of fixed length from a TokenizedDocumentCache.

    :param doc_cache: the TokenizedDocumentCache to draw from
    :param seq_len: The max length of sequences to emit
    """

    def __init__(self, doc_cache, seq_len: int, stride: Optional[int] = None, min_doc_len: int = 0):
        self.doc_cache = doc_cache
        self.seq_len = seq_len
        self.stride = stride
        self.min_doc_len = min_doc_len

    def shard(self, shard_id: int, num_shards: int) -> "TokenSeqWithDocBoundsDataset":
        """
        Split the dataset into num_processes shards.
        """
        return TokenSeqWithDocBoundsDataset(self.doc_cache.shard(shard_id, num_shards), self.seq_len, self.stride)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        extra_tokens = None  # BatchEncoding of the last tokens from the previous doc
        for doc in self.doc_cache:
            # TODO: we could be cleverer here, and avoid these expensive copies etc
            # should run some benchmarks to see if it's worth it

            if extra_tokens is not None:
                doc = _stack_batch_encodings(extra_tokens, doc)
                extra_tokens = None

            # This mutably pulls examples out of doc
            input_ids = doc["input_ids"]
            assert isinstance(input_ids, list)
            for ids in input_ids:
                assert isinstance(ids, np.ndarray)
            examples = make_examples(input_ids, self.seq_len, self.min_doc_len)
            if len(doc) > 0:
                extra_tokens = doc

            yield from examples

    @staticmethod
    def load(
        seq_len: int, cache_dir: str, stride: Optional[int] = None, min_doc_len: int = 0
    ) -> "TokenSeqWithDocBoundsDataset":
        # Maybe force the cache to be built ahead of time?
        doc_cache = TokenizedDocumentCache.load(cache_dir, False)
        return TokenSeqWithDocBoundsDataset(doc_cache, seq_len, stride, min_doc_len=min_doc_len)
