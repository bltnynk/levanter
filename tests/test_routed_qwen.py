import jax

import haliax as hax

from levanter.models.routed_lora_model import RQwenConfig, RQwenLMHeadModel


def test_routed_qwen_state_dict_keys():
    config = RQwenConfig(seq_len=512, num_layers=2, hidden_dim=256, intermediate_dim=512, tie_word_embeddings=True)
    Vocab = hax.Axis("vocab", 1000)
    key = jax.random.PRNGKey(0)
    model = RQwenLMHeadModel.init(Vocab, config, key=key)

    print(model.to_state_dict().keys())


def test_routed_qwen_forward():
    config = RQwenConfig(seq_len=512, num_layers=2, hidden_dim=256, intermediate_dim=512, tie_word_embeddings=True)
    Vocab = hax.Axis("vocab", 1000)
    key = jax.random.PRNGKey(0)
    model = RQwenLMHeadModel.init(Vocab, config, key=key)

    Batch = hax.Axis("batch", 32)
    x = hax.random.randint(key, (Batch, config.Pos), 0, Vocab.size)
    inds = hax.random.randint(key, Batch, 0, config.Pos.size - 1)
    y = model.routed_forward(Batch, x, inds)
    print(y)
