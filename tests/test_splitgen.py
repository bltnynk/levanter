import jax

import haliax as hax
from haliax import Axis

from levanter.models import llama_splitgen as lsg
from levanter.models.attention import AttentionMask


def small_cfg():
    cfg = lsg.SplitLlamaConfig(
        seq_len=256,
        hidden_dim=64,
        intermediate_dim=128,
        num_heads=4,
        num_layers=4,
        num_kv_heads=4,
        skip_indices=[2, 3],
        skip_after_k_tokens=32,
    )
    return cfg


def test_loraize():
    Vocab = Axis("Vocab", 1000)
    cfg = small_cfg()
    key = jax.random.PRNGKey(0)
    model = lsg.LlamaLMHeadModel.init(Vocab, cfg, key=key)

    model_lora = lsg.loraize(model, cfg, key=key)
    for layer_idx in range(cfg.Layers.size):
        if layer_idx not in cfg.skip_indices:
            sa = model_lora.transformer.layers.blocks[layer_idx].self_attn
            for proj in [sa.q_proj, sa.k_proj, sa.v_proj]:
                assert isinstance(proj, lsg.SplitLoraLinear)


def _get_random_inputs(config: lsg.SplitLlamaConfig, Vocab: Axis):
    Batch = hax.Axis("batch", 2)
    x = hax.random.randint(jax.random.PRNGKey(0), (Batch, config.Pos), 0, Vocab.size)
    mask = AttentionMask.causal()
    return x, mask


def test_splitgen_forward():
    Vocab = Axis("Vocab", 1000)
    cfg = small_cfg()
    key = jax.random.PRNGKey(0)
    model = lsg.LlamaLMHeadModel.init(Vocab, cfg, key=key)

    model_lora = lsg.loraize(model, cfg, key=key)
    inputs = _get_random_inputs(cfg, Vocab)
    out = model_lora(*inputs).array

    @hax.named_jit
    def jitted_fwd(x, mask):
        return model(x, mask)

    out2 = jitted_fwd(*inputs).array

    assert jax.numpy.allclose(out, out2, atol=1e-5).item()
