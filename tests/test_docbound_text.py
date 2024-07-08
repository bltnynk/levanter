import numpy as np

from levanter.data.docbound_text import make_examples


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
