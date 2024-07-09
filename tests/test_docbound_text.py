import tempfile

import numpy as np

import haliax as hax

import tiny_test_corpus
from levanter.data.docbound_text import SplitGenDataset, TokenSeqWithDocBoundsDataset, make_examples, make_split_mask


def make_text_batches(lens):
    zero = np.array([0], dtype=int)
    return [np.concatenate((zero, np.random.randint(1, 10, size=li - 1))) for li in lens]


def test_make_examples():
    data = make_text_batches([12, 324, 52, 230, 234, 1000, 24])
    assert all(di[0] == 0 for di in data)
    seq_len = 300
    min_doc_length = 50
    examples = make_examples(data, seq_len, min_doc_length)
    start_inds, examples = zip(*examples)
    assert all(len(e) == seq_len for e in examples)
    for si, e in zip(start_inds, examples):
        for i in range(1, len(si)):
            assert e[si[i]] == 0, e[si[i] - 5 : si[i] + 5]


def test_split_mask():
    start_inds = np.array([0, 5, 7])
    target = "|00111|00|001111111"
    target = np.array([bool(int(t)) for t in target if t != "|"])
    skip_after_k_tokens = 2
    seq_len = 16
    split_mask = make_split_mask(start_inds, seq_len, skip_after_k_tokens)

    for e, t in zip(split_mask, target):
        assert e == t, (split_mask.astype(int), target.astype(int))


def test_split_gen_dataset():
    skip_after_k = 16
    seq_len = 256
    bos_token_id = 0
    Pos = hax.Axis("Pos", seq_len)
    KeyPos = hax.Axis("KeyPos", seq_len)
    with tempfile.TemporaryDirectory() as tmpdir:
        _, caches = tiny_test_corpus.construct_realistic_data_cache(
            tmpdir,
            doc_len_min=8,
            doc_len_max=512,
            vocab_size=1024,
            bos_token_id=bos_token_id,
            chunk_size=64,
            num_shards=1,
        )
        dset = TokenSeqWithDocBoundsDataset(caches["train"], seq_len, min_doc_len=skip_after_k)
        split_dset = SplitGenDataset(dset, Pos, KeyPos, skip_after_k, ignore_index=0)
        for elem in split_dset:
            i = 0
            while i < elem.lm_example.tokens.size:
                if elem.lm_example.tokens[Pos, i] == 0:
                    assert not elem.split_mask[Pos, i : i + skip_after_k].any()
                    i += skip_after_k
                else:
                    assert elem.split_mask[Pos, i]
                    i += 1
