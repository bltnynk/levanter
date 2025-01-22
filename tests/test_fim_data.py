import tempfile

import jax.numpy as jnp
import pytest
from jax.random import PRNGKey

import haliax as hax

from levanter.data.text import FIMUrlSourceConfig, mk_fim_dataset
from tiny_test_corpus import write_fim_data


@pytest.mark.asyncio
@pytest.mark.parametrize("flattened", [True, False])
@pytest.mark.parametrize("predict_prefix", [True, False])
@pytest.mark.parametrize("predict_fim_token", [True, False])
async def test_fim_url_data(flattened, predict_prefix, predict_fim_token):
    with tempfile.TemporaryDirectory() as tmpdir:
        max_len = 128
        test_data_jsonl = "file://" + write_fim_data(tmpdir + "/test_data.jsonl", flattened=flattened, len=1024)
        cfg = FIMUrlSourceConfig(
            cache_dir=tmpdir + "/cache",
            train_urls=[test_data_jsonl],
            predict_prefix=predict_prefix,
            predict_fim_token=predict_fim_token,
            add_router_token=False,
            predict_router_token=False,
            shuffle=True,
            pack=True,
            data_format="flatted" if flattened else "repo_level",
        )
        tokenizer = cfg.the_tokenizer
        Pos = hax.Axis("Pos", max_len)
        dataset = mk_fim_dataset(cfg, "train", tokenizer, Pos, key=PRNGKey(0))
        dset_len = await dataset.async_len()
        last_elem = await dataset.get_batch([dset_len - 1])
        elem0 = last_elem[0]

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
