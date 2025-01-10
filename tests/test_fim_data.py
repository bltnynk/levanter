import json
import tempfile

import jax.numpy as jnp
import pytest

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
@pytest.mark.parametrize("predict_prefix", [True, False])
@pytest.mark.parametrize("predict_fim_token", [True, False])
async def test_fim_url_data(predict_prefix, predict_fim_token):
    with tempfile.TemporaryDirectory() as tmpdir:
        max_len = 128
        test_data_jsonl = "file://" + write_test_data(tmpdir + "/test_data.jsonl")
        cfg = FIMUrlSourceConfig(
            cache_dir=tmpdir + "/cache",
            train_urls=[test_data_jsonl],
            predict_prefix=predict_prefix,
            predict_fim_token=predict_fim_token,
            add_router_token=False,
            predict_router_token=False,
            shuffle=False,
            pack=True,
        )
        tokenizer = cfg.the_tokenizer
        Pos = hax.Axis("Pos", max_len)
        dataset = mk_fim_dataset(cfg, "train", tokenizer, Pos)
        await dataset.wait_until_len_at_least(1)
        elem = await dataset.get_batch([0, 1, 2, 3])
        elem0 = elem[0]

        fim_token_id, eos_token_id, prefix_token, pad_token_id = tokenizer.convert_tokens_to_ids(
            [cfg.middle_token, cfg.eos_token, cfg.prefix_token, tokenizer.pad_token]
        )

        starts = jnp.argwhere(elem0.tokens.array == prefix_token).flatten().tolist()
        middles = jnp.argwhere(elem0.tokens.array == fim_token_id).flatten().tolist()
        ends = jnp.argwhere(elem0.tokens.array == eos_token_id).flatten().tolist()
        pads = jnp.argwhere(elem0.tokens.array == pad_token_id)
        lm = elem0.loss_mask.array
        assert not lm[pads].any().item(), "doesn' predict from any pad tokens"
        assert len(starts) == len(ends)
        assert len(starts) == len(middles)
        for s, m, e in zip(starts, middles, ends):
            # Loss mask testing
            assert (
                lm[s : m - 1].all().item() == predict_prefix
            ), f"predict_prefix={predict_prefix}, prefix_mask={lm[s: m-1]}"
            assert (
                lm[m - 1].item() == predict_fim_token
            ), f"predict_fim_token={predict_fim_token}, fim_token_mask={lm[m-1]}"
            assert lm[m:e].all().item(), "should always predict completion"
            assert not lm[e].all().item(), "should never predict after eos"
            # hs idxs testing
            assert (elem0.router_hs_idxs.array[m:e] == m - 1).all().item()
