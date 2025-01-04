import jax
import jax.numpy as jnp

import haliax as hax

from levanter.models.llama import LlamaConfig, LlamaLMHeadModel


def test_embedding_grad_filter():
    from levanter.main.rlora_train import filter_embedding_grads

    config = LlamaConfig(
        seq_len=128,
        hidden_dim=16,
        num_heads=4,
        num_kv_heads=4,
        gradient_checkpointing=False,
    )
    # test that the filter works
    Vocab = hax.Axis("vocab", 100)
    model = LlamaLMHeadModel.init(Vocab, config, key=jax.random.PRNGKey(0))

    def replace_with_ones(x):
        return hax.ones_like(x)

    def is_leaf(x):
        return isinstance(x, hax.NamedArray)

    model: LlamaLMHeadModel = jax.tree.map(replace_with_ones, model, is_leaf=is_leaf)
    assert hax.all(model.embeddings.token_embeddings.weight == 1.0).item()

    token_mask = hax.ones(Vocab, dtype=jnp.bool).at[Vocab, 50].set(0.0)
    tform = filter_embedding_grads(model.Embed, Vocab, token_mask)
    res, _ = tform.update(model, ())
    res: LlamaLMHeadModel
    token_embs = res.embeddings.token_embeddings.weight
    assert hax.all(
        token_embs[Vocab, 50] == 0.0
    ).item(), f"Token 50 should be zeroed out, got {token_embs[Vocab, 50].tolist()}"

    assert hax.all(
        token_embs[Vocab, :50] == 1.0
    ).item(), f"Token 50 should be zeroed out, got {token_embs[Vocab, 50].tolist()}"

    assert hax.all(
        token_embs[Vocab, 51:] == 1.0
    ).item(), f"Token 50 should be zeroed out, got {token_embs[Vocab, 50].tolist()}"
