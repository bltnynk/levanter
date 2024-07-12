import tempfile

import jax.numpy as jnp
import numpy as np

import haliax as hax

import tiny_test_corpus
from levanter.data.docbound_text import dset_from_config, make_examples, make_split_mask


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
        config, _ = tiny_test_corpus.construct_realistic_data_cache(
            tmpdir,
            doc_len_min=8,
            doc_len_max=2048,
            vocab_size=1024,
            bos_token_id=bos_token_id,
            chunk_size=64,
            num_shards=1,
        )
        dset = dset_from_config(config, "train", Pos, KeyPos, skip_after_k_tokens=skip_after_k)
        for elem in dset:
            assert (elem.loss_mask.array[:-1] == elem.split_mask.array[:-1]).all()
            start_inds = jnp.argwhere(elem.tokens.array == 0).squeeze(1)
            assert elem.tokens.array.shape[0] == seq_len
            for i in range(0, len(start_inds)):
                split_end = start_inds[i] + skip_after_k
                assert not elem.split_mask[Pos, start_inds[i] : split_end].any()
                next_start = start_inds[i + 1] if i + 1 < len(start_inds) else seq_len
                assert elem.split_mask[Pos, split_end:next_start].all()

        loss_mask_after_k = 24
        dset = dset_from_config(
            config, "train", Pos, KeyPos, skip_after_k_tokens=skip_after_k, loss_mask_after_k_tokens=loss_mask_after_k
        )
        for elem in dset:
            start_inds = jnp.argwhere(elem.tokens.array == 0).squeeze(1)
            assert elem.tokens.array.shape[0] == seq_len
            for i in range(0, len(start_inds)):
                split_end = start_inds[i] + skip_after_k
                assert not elem.split_mask[Pos, start_inds[i] : split_end].any()
                next_start = start_inds[i + 1] if i + 1 < len(start_inds) else seq_len
                assert elem.split_mask[Pos, split_end:next_start].all()

                # loss mask checking
                mask_until = min(next_start, start_inds[i] + loss_mask_after_k, seq_len - 1)
                assert not elem.loss_mask[Pos, start_inds[i] : mask_until].any()
                unmask_until = min(next_start, seq_len - 1)
                print(mask_until, unmask_until)
                assert elem.loss_mask[Pos, mask_until:unmask_until].all()
