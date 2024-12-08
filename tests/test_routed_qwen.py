import dataclasses
import tempfile

import jax
import numpy as np

import haliax as hax

from levanter.models.attention import AttentionMask
from levanter.models.routed_lora_model import RQwenConfig, RQwenLMHeadModel
from test_utils import skip_if_no_torch


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
    _ = model.routed_forward(Batch, x, inds)

    # test with num_loras=1
    config = dataclasses.replace(config, num_loras=1, top_k=1)
    model = RQwenLMHeadModel.init(Vocab, config, key=key)
    _ = model.routed_forward(Batch, x, inds)


@skip_if_no_torch
def test_rqwen_consistent_with_base_qwen():
    import torch
    from transformers import Qwen2ForCausalLM

    converter = RQwenConfig().hf_checkpoint_converter()
    config = RQwenConfig(
        seq_len=128,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        hidden_dim=16,
        intermediate_dim=32,
        tie_word_embeddings=True,
        disable_lora_mask=True,  # disable lora mask to keep consistency
    )
    Vocab = hax.Axis("vocab", 1000)
    hf_config = config.to_hf_config(Vocab.size)

    Batch = hax.Axis("batch", 2)
    input = hax.random.randint(jax.random.PRNGKey(0), (Batch, config.Pos), 0, Vocab.size)
    seq_inds = hax.random.randint(jax.random.PRNGKey(0), Batch, 0, config.Pos.size - 1)
    attn_mask = AttentionMask.causal()
    input_torch = torch.from_numpy(np.array(input.array)).to(torch.int32)

    torch.random.manual_seed(0)

    torch_model = Qwen2ForCausalLM(hf_config)
    torch_model.eval()

    torch_out = torch_model(input_torch)
    torch_out = torch_out.logits.detach().cpu().numpy()
    # torch_out = jax.nn.softmax(torch_out, axis=-1)

    with tempfile.TemporaryDirectory() as tmpdir:
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

        model = converter.load_pretrained(
            Qwen2ForCausalLM, ref=f"{tmpdir}/torch_model", resize_vocab_to_match_tokenizer=False
        )

        @hax.named_jit
        def compute(model: RQwenLMHeadModel, input):
            model_output = model.routed_forward(Batch, input, router_hs_idxs=seq_inds, attn_mask=attn_mask)
            return model_output

        jax_out = compute(model, input).array

        assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
        assert np.isclose(torch_out, np.array(jax_out), rtol=1e-4, atol=1e-4).all(), f"{torch_out} != {jax_out}"

        cfg_with_lora = dataclasses.replace(config, disable_lora_mask=False)
        model = dataclasses.replace(model, transformer=dataclasses.replace(model.transformer, config=cfg_with_lora))

        @hax.named_jit
        def compute(model: RQwenLMHeadModel, input):
            model_output = model.routed_forward(Batch, input, router_hs_idxs=seq_inds, attn_mask=attn_mask)
            return model_output

        jax_out = compute(model, input).array

        assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
        # Since we 0 init the A layer this might not hold actually
        # assert not np.isclose(torch_out, np.array(jax_out), rtol=1e-4, atol=1e-4).all(), f"{torch_out} == {jax_out} with lora mask"