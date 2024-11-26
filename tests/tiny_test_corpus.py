import json
import os

import numpy
import numpy as np

from levanter.data.audio import AudioIODatasetConfig
from levanter.data.text import (
    CANONICAL_FILE_CONTENT_FIELD,
    CANONICAL_FILE_PATH_FIELD,
    CANONICAL_FILES_FIELD,
    CANONICAL_ID_FIELD,
    CANONICAL_REPO_NAME_FIELD,
    LMDatasetConfig,
)
from levanter.store.cache import TreeCache


def _write_tiny_corpus(path):
    os.makedirs(f"{path}/train", exist_ok=True)
    with open(f"{path}/train/docs.jsonl", "w") as f:
        for i in range(10):
            f.write(json.dumps({"text": f"hello world {i} " * 100}))
            f.write("\n")

    os.makedirs(f"{path}/validation", exist_ok=True)
    with open(f"{path}/validation/docs.jsonl", "w") as f:
        for i in range(10):
            f.write(json.dumps({"text": f"bye world {i} " * 100}))
            f.write("\n")


def tiny_corpus_config(path):
    _write_tiny_corpus(path)
    return LMDatasetConfig(
        train_urls=[f"file://{path}/train/docs.jsonl"],
        validation_urls=[f"file://{path}/validation/docs.jsonl"],
        cache_dir=f"{path}/cache",
    )


def tiny_asr_corpus_config(path):
    return AudioIODatasetConfig(
        id="WillHeld/test_librispeech_parquet",
        text_key="text",
        train_split="validation",
        validation_split="validation",
        cache_dir=f"{path}/cache_asr",
    )


def construct_small_data_cache(
    path, num_shards=8, chunk_size=512, doc_len=128, vocab_size=1024
) -> tuple[LMDatasetConfig, dict[str, TreeCache]]:
    from levanter.store.cache import SerialCacheWriter

    rng = numpy.random.default_rng(0)

    caches: dict[str, TreeCache] = {}

    exemplar = {"input_ids": numpy.zeros((doc_len,), dtype=numpy.int32)}

    for split in ["train", "validation"]:
        with SerialCacheWriter(f"{path}/cache/{split}", exemplar) as writer:
            for shard in range(num_shards):
                writer.write_batch(
                    [
                        {"input_ids": rng.integers(0, vocab_size, size=(doc_len,), dtype=np.int32)}
                        for _ in range(chunk_size)
                    ]
                )
        caches[split] = writer.result()

    config = LMDatasetConfig(
        train_urls=[f"file://{path}/train/docs.jsonl"],
        validation_urls=[f"file://{path}/validation/docs.jsonl"],
        cache_dir=f"{path}/cache",
        vocab_size=vocab_size,
        tokenizer="gpt2",
    )

    return config, caches


def write_fim_data(path, len=128) -> str:
    with open(path, "w") as f:
        for i in range(len):
            output = {
                CANONICAL_REPO_NAME_FIELD: f"repo{i}",
                CANONICAL_FILES_FIELD: [
                    {
                        CANONICAL_ID_FIELD: f"file{i}",
                        CANONICAL_FILE_PATH_FIELD: f"file{i}.txt",
                        CANONICAL_FILE_CONTENT_FIELD: (
                            f"file{i}_content_a_b_c_d_e_f_g_h_i_j_k_l_m_n_o_p_q_r_s_t_u_v_w_x_y_z"
                        ),
                    }
                ],
            }
            f.write(json.dumps(output) + "\n")
        f.flush()
    return path
