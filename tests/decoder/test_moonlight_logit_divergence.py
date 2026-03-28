# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Teacher-forced logit divergence test for Moonlight 16B-A3B.

Compares next-token prediction logits at 32 consecutive positions between:
- HF Transformers model (BF16 on CPU)
- Neuron model (BF16 on device, on_device_sampling=False for logit access)

Approach (autoregressive teacher forcing):
1. HF generates 32 reference tokens greedily from each prompt.
2. HF runs a teacher-forced forward pass over the full sequence to get
   logits at all 32 positions.
3. Neuron runs autoregressive teacher forcing: context-encode the prompt,
   then feed each reference token one at a time via token generation,
   collecting logits at each step. This is necessary because the Neuron
   decoder only returns last-position logits per forward call.

The Neuron KV cache accumulates BF16 hardware differences over positions,
so cosine similarity degrades slightly at later positions (expected).

Validated baseline (trn2.3xlarge, tp=2, seq=512):
  cos_sim min=0.935, mean>=0.97; top-1 match >= 81% per prompt.

Requires trn2 with at least 2 NeuronCores and MOONLIGHT_MODEL_PATH set.
Total runtime: ~8 min (HF model load + generation ~2.5 min, Neuron compilation ~5.5 min).

Usage:
    MOONLIGHT_MODEL_PATH=/path/to/Moonlight-16B-A3B \
        pytest tests/decoder/test_moonlight_logit_divergence.py -v -s
"""

import gc
import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


MODEL_PATH = os.environ.get("MOONLIGHT_MODEL_PATH")

skip_no_model = pytest.mark.skipif(MODEL_PATH is None, reason="MOONLIGHT_MODEL_PATH not set")

PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "The largest planet in the solar system is",
]

NUM_TOKENS = 32

# Thresholds for autoregressive teacher forcing over 32 positions.
# BF16 KV cache divergence causes cosine similarity to degrade at later positions.
# Validated baseline: min=0.935, mean>=0.97 on trn2.3xlarge.
COS_SIM_MIN_THRESHOLD = 0.92  # worst-case single position
COS_SIM_MEAN_THRESHOLD = 0.96  # average across all 32 positions
TOP1_MATCH_FRACTION = 0.75  # at least 75% of 32 positions must match per prompt


def _generate_reference(model, tokenizer, prompt, n_tokens):
    """Generate n_tokens greedily from HF model and return teacher-forced logits.

    Uses manual greedy decoding with use_cache=False to avoid DynamicCache
    incompatibilities between the Moonlight custom code and transformers 4.57+.

    Returns:
        hf_logits: (n_tokens, vocab_size) float32 — logits predicting each generated token
        ref_token_ids: (n_tokens,) int — the generated token IDs
        prompt_len: int — number of prompt tokens
    """
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_len = input_ids.shape[1]

    # Manual greedy generation without KV cache
    gen_ids = input_ids.clone()
    for i in range(n_tokens):
        with torch.no_grad():
            out = model(input_ids=gen_ids, use_cache=False)
        next_token = out.logits[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
        gen_ids = torch.cat([gen_ids, next_token], dim=1)
        if (i + 1) % 8 == 0:
            print(f"    generated {i + 1}/{n_tokens} tokens...", flush=True)

    ref_token_ids = gen_ids[0, prompt_len : prompt_len + n_tokens].clone()

    # Teacher-forced forward: feed full sequence, extract logits at n_tokens positions
    with torch.no_grad():
        out = model(input_ids=gen_ids, use_cache=False)

    # Logits at position i predict token at position i+1
    # Positions [prompt_len-1 .. prompt_len+n_tokens-2] predict the generated tokens
    hf_logits = out.logits[0, prompt_len - 1 : prompt_len + n_tokens - 1, :].clone().float()

    return hf_logits, ref_token_ids, prompt_len


def _neuron_teacher_forced_logits(model, input_ids, prompt_len, n_tokens):
    """Run teacher-forced forward on Neuron and return logits at the same positions.

    The Neuron decoder only returns logits for the last position per forward call.
    So we do autoregressive teacher forcing:
    1. Context encode the prompt → logits predicting first generated token
    2. Feed each reference token one at a time via token generation → logits for next position

    At each step, we feed the correct reference token (from HF), not the Neuron prediction.
    """
    seq_ids = torch.tensor([0])
    sampling_params = torch.ones((1, 3))
    all_logits = []

    # Step 1: Context encode the prompt
    prompt_ids = input_ids[:, :prompt_len]
    prompt_pos = torch.arange(prompt_len).unsqueeze(0)
    model.reset()
    out = model(
        input_ids=prompt_ids,
        position_ids=prompt_pos,
        seq_ids=seq_ids,
        sampling_params=sampling_params,
    )
    # out shape: (1, 1, vocab_size) — logits at last prompt position
    all_logits.append(out[0, -1, :].clone().float())

    # Step 2: Feed each reference token via token generation
    for i in range(n_tokens - 1):
        token_id = input_ids[:, prompt_len + i]  # reference token
        pos = torch.tensor([[prompt_len + i]])
        out = model(
            input_ids=token_id.unsqueeze(0) if token_id.dim() == 0 else token_id.unsqueeze(-1),
            position_ids=pos,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
        )
        all_logits.append(out[0, -1, :].clone().float())

    return torch.stack(all_logits, dim=0)  # (n_tokens, vocab_size)


@pytest.fixture(scope="module")
def hf_reference():
    """Load HF Moonlight in BF16 on CPU, generate references, free memory.

    Ordering: this fixture MUST complete before the Neuron fixture starts,
    because both require significant RAM. The neuron_model fixture depends
    on this one to enforce that ordering.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()

    results = {}
    for prompt in PROMPTS:
        hf_logits, ref_tokens, prompt_len = _generate_reference(model, tokenizer, prompt, NUM_TOKENS)
        results[prompt] = {
            "hf_logits": hf_logits,  # (NUM_TOKENS, vocab_size)
            "ref_tokens": ref_tokens,  # (NUM_TOKENS,)
            "prompt_len": prompt_len,
        }
        decoded = tokenizer.decode(ref_tokens.tolist())
        print(f"  HF reference for {prompt!r}: {decoded!r}")

    del model
    gc.collect()

    return results, tokenizer


@pytest.fixture(scope="module")
def neuron_model(hf_reference):
    """Compile Neuron model with on_device_sampling=False for logit extraction.

    Depends on hf_reference to ensure HF model is freed before compilation starts.
    """
    nc = NeuronModelForCausalLM.get_neuron_config(
        model_name_or_path=MODEL_PATH,
        batch_size=1,
        sequence_length=512,
        tensor_parallel_size=2,
    )
    nc.on_device_sampling = False

    model = NeuronModelForCausalLM.export(
        MODEL_PATH,
        neuron_config=nc,
        load_weights=True,
        trust_remote_code=True,
    )
    yield model


@skip_no_model
@is_inferentia_test
@requires_neuronx
def test_teacher_forced_cosine_similarity(hf_reference, neuron_model):
    """Verify logit vectors are highly similar at all 32 positions per prompt."""
    results, tokenizer = hf_reference

    for prompt in PROMPTS:
        r = results[prompt]
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        full_ids = torch.cat([input_ids, r["ref_tokens"].unsqueeze(0)], dim=1)

        neuron_logits = _neuron_teacher_forced_logits(
            neuron_model, full_ids, r["prompt_len"], NUM_TOKENS
        )
        hf_logits = r["hf_logits"]

        # Truncate to common vocab size
        v = min(hf_logits.shape[1], neuron_logits.shape[1])

        cos_sims = torch.nn.functional.cosine_similarity(
            hf_logits[:, :v], neuron_logits[:, :v], dim=1
        )

        cos_min = cos_sims.min().item()
        cos_mean = cos_sims.mean().item()
        print(f"\n  {prompt!r} ({NUM_TOKENS} positions):")
        print(f"    cos_sim: min={cos_min:.6f}, mean={cos_mean:.6f}, max={cos_sims.max():.6f}")
        below = (cos_sims < COS_SIM_MIN_THRESHOLD).sum().item()
        if below:
            bad_pos = (cos_sims < COS_SIM_MIN_THRESHOLD).nonzero(as_tuple=True)[0].tolist()
            print(f"    positions below {COS_SIM_MIN_THRESHOLD}: {bad_pos}")

        assert cos_min >= COS_SIM_MIN_THRESHOLD, (
            f"Cosine similarity min below {COS_SIM_MIN_THRESHOLD} for {prompt!r}: "
            f"min={cos_min:.6f}"
        )
        assert cos_mean >= COS_SIM_MEAN_THRESHOLD, (
            f"Cosine similarity mean below {COS_SIM_MEAN_THRESHOLD} for {prompt!r}: "
            f"mean={cos_mean:.6f}"
        )


@skip_no_model
@is_inferentia_test
@requires_neuronx
def test_teacher_forced_top1_match(hf_reference, neuron_model):
    """Verify greedy-decoded token matches at >= 75% of 32 positions per prompt."""
    results, tokenizer = hf_reference

    for prompt in PROMPTS:
        r = results[prompt]
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        full_ids = torch.cat([input_ids, r["ref_tokens"].unsqueeze(0)], dim=1)

        neuron_logits = _neuron_teacher_forced_logits(
            neuron_model, full_ids, r["prompt_len"], NUM_TOKENS
        )
        hf_logits = r["hf_logits"]

        v = min(hf_logits.shape[1], neuron_logits.shape[1])
        hf_preds = hf_logits[:, :v].argmax(dim=1)
        neuron_preds = neuron_logits[:, :v].argmax(dim=1)
        matches = (hf_preds == neuron_preds).sum().item()
        frac = matches / NUM_TOKENS

        print(f"\n  {prompt!r}: top-1 match {matches}/{NUM_TOKENS} ({frac:.0%})")
        mismatches = (hf_preds != neuron_preds).nonzero(as_tuple=True)[0].tolist()
        for pos in mismatches[:5]:  # show first 5 mismatches
            hf_tok = tokenizer.decode([hf_preds[pos].item()])
            n_tok = tokenizer.decode([neuron_preds[pos].item()])
            print(f"    pos {pos}: HF={hf_tok!r}, Neuron={n_tok!r}")

        assert frac >= TOP1_MATCH_FRACTION, (
            f"Top-1 match rate {frac:.0%} below {TOP1_MATCH_FRACTION:.0%} for {prompt!r}"
        )


@skip_no_model
@is_inferentia_test
@requires_neuronx
def test_first_token_paris(hf_reference, neuron_model):
    """Smoke check: 'The capital of France is' -> first token contains 'Paris'."""
    results, tokenizer = hf_reference
    r = results[PROMPTS[0]]

    input_ids = tokenizer(PROMPTS[0], return_tensors="pt")["input_ids"]
    full_ids = torch.cat([input_ids, r["ref_tokens"].unsqueeze(0)], dim=1)

    neuron_logits = _neuron_teacher_forced_logits(
        neuron_model, full_ids, r["prompt_len"], NUM_TOKENS
    )
    first_token_id = neuron_logits[0].argmax().item()
    decoded = tokenizer.decode([first_token_id])
    print(f"  Neuron first token: {decoded!r} (id={first_token_id})")
    assert "Paris" in decoded, f"Expected 'Paris' in first token, got {decoded!r}"
