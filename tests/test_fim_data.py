import json
import tempfile

import jax
import pytest
from transformers import AutoTokenizer

import haliax as hax

from levanter.data.text import (
    CANONICAL_FILE_CONTENT_FIELD,
    CANONICAL_FILE_PATH_FIELD,
    CANONICAL_FILES_FIELD,
    CANONICAL_ID_FIELD,
    CANONICAL_REPO_NAME_FIELD,
    FIMUrlSourceConfig,
    mk_fim_dataset,
)


def write_test_data(path, len=128) -> str:
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


@pytest.mark.asyncio
async def test_fim_url_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        max_len = 128
        test_data_jsonl = "file://" + write_test_data(tmpdir + "/test_data.jsonl")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B")
        cfg = FIMUrlSourceConfig(cache_dir=tmpdir + "/cache", train_urls=[test_data_jsonl])
        Pos = hax.Axis("Pos", max_len)
        dataset = mk_fim_dataset(cfg, "train", tokenizer, Pos)
        await dataset.wait_until_len_at_least(1)
        elem = await dataset.get_batch([0, 1, 2, 3])

        print("Check that for all the uppercase tokens, we want to predict the token after them")
        for e in elem:
            # trim the padding
            tokens = e.tokens.array
            last_non_pad = jax.numpy.where(tokens != tokenizer.pad_token_id)[0][-1]
            text = tokenizer.decode(e.tokens.array[: last_non_pad + 5])
            chars = list(text)
            remap = tokenizer(text)
            for token_idx, masked in enumerate(e.loss_mask.array[: last_non_pad + 5]):
                cs = remap.token_to_chars(token_idx)
                if masked:
                    chars[cs.start : cs.end] = [c.upper() for c in chars[cs.start : cs.end]]
            text = "".join(chars)

            print(text)
