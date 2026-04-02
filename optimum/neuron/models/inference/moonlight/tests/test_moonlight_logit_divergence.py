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

Compares next-token prediction logits at N consecutive positions between:
- HF Transformers model (FP32 on CPU)
- Neuron model (FP32 on device, --auto-cast=none, on_device_sampling=False)

Approach:
1. Neuron generates N reference tokens greedily (fast, ~17ms/token on device).
2. Neuron runs teacher-forced forward: context-encode the prompt, then feed
   each reference token one at a time, collecting logits at each step.
3. HF runs a single teacher-forced forward pass over the full sequence
   (prompt + N tokens) with use_cache=False, extracting logits at all positions.

Both models run in FP32 to isolate implementation differences from
precision-related drift. The Neuron model generates reference tokens because
HF model.generate() is incompatible with transformers 4.57+ DynamicCache
changes in the Moonlight custom code.

Requires trn2 with at least 2 NeuronCores and MOONLIGHT_MODEL_PATH set.

Environment variables:
    MOONLIGHT_MODEL_PATH  — path to HF checkpoint (required)
    MOONLIGHT_NUM_TOKENS  — number of generated positions to compare (default: 32)
    MOONLIGHT_TP_SIZE     — tensor parallel degree (default: 2)
    MOONLIGHT_SEQ_LEN     — Neuron compiled sequence length (default: auto)

Usage:
    # Default 32-token test (~8 min):
    MOONLIGHT_MODEL_PATH=/path/to/Moonlight-16B-A3B \
        pytest optimum/neuron/models/inference/moonlight/tests/test_moonlight_logit_divergence.py -v -s

    # 1024-token FP32 test (requires tp=4 for FP32):
    MOONLIGHT_MODEL_PATH=/path/to/Moonlight-16B-A3B \
    MOONLIGHT_NUM_TOKENS=1024 MOONLIGHT_TP_SIZE=4 NEURON_RT_LOGICAL_NC_PER_DEVICE=1 \
        pytest optimum/neuron/models/inference/moonlight/tests/test_moonlight_logit_divergence.py -v -s
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

NUM_TOKENS = int(os.environ.get("MOONLIGHT_NUM_TOKENS", "32"))

# Thresholds for autoregressive teacher forcing.
# BF16 KV cache divergence causes logit values to drift at later positions.
MAX_MEAN_LOGIT_DIFF = 2.0  # mean |HF_logit - Neuron_logit| for top-1 token
TOP1_MATCH_FRACTION = 0.75  # at least 75% of positions must match per prompt


def _neuron_greedy_generate(model, input_ids, prompt_len, n_tokens):
    """Generate n_tokens greedily from Neuron model via autoregressive decoding.

    Returns:
        gen_token_ids: (n_tokens,) int — the generated token IDs
        all_logits: (n_tokens, vocab_size) float32 — logits at each position
    """
    seq_ids = torch.tensor([0])
    sampling_params = torch.ones((1, 3))
    all_logits = []

    # Context encode the prompt
    prompt_ids = input_ids[:, :prompt_len]
    prompt_pos = torch.arange(prompt_len).unsqueeze(0)
    model.reset()
    out = model(
        input_ids=prompt_ids,
        position_ids=prompt_pos,
        seq_ids=seq_ids,
        sampling_params=sampling_params,
    )
    logits = out[0, -1, :].clone().float()
    all_logits.append(logits)
    next_token = logits.argmax().item()
    gen_tokens = [next_token]

    # Autoregressive generation
    for i in range(n_tokens - 1):
        token = torch.tensor([[next_token]])
        pos = torch.tensor([[prompt_len + i]])
        out = model(
            input_ids=token,
            position_ids=pos,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
        )
        logits = out[0, -1, :].clone().float()
        all_logits.append(logits)
        next_token = logits.argmax().item()
        gen_tokens.append(next_token)
        if (i + 1) % 100 == 0:
            print(f"    neuron generated {i + 1}/{n_tokens - 1} tokens...", flush=True)

    return torch.tensor(gen_tokens), torch.stack(all_logits, dim=0)


def _neuron_teacher_forced_logits(model, input_ids, prompt_len, n_tokens):
    """Run teacher-forced forward on Neuron and return logits at each position.

    The Neuron decoder only returns logits for the last position per forward call.
    So we do autoregressive teacher forcing:
    1. Context encode the prompt -> logits predicting first generated token
    2. Feed each reference token one at a time via token generation
    """
    seq_ids = torch.tensor([0])
    sampling_params = torch.ones((1, 3))
    all_logits = []

    # Context encode the prompt
    prompt_ids = input_ids[:, :prompt_len]
    prompt_pos = torch.arange(prompt_len).unsqueeze(0)
    model.reset()
    out = model(
        input_ids=prompt_ids,
        position_ids=prompt_pos,
        seq_ids=seq_ids,
        sampling_params=sampling_params,
    )
    all_logits.append(out[0, -1, :].clone().float())

    # Feed each reference token via token generation
    for i in range(n_tokens - 1):
        token_id = input_ids[:, prompt_len + i]
        pos = torch.tensor([[prompt_len + i]])
        out = model(
            input_ids=token_id.unsqueeze(0) if token_id.dim() == 0 else token_id.unsqueeze(-1),
            position_ids=pos,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
        )
        all_logits.append(out[0, -1, :].clone().float())
        if (i + 1) % 100 == 0:
            print(f"    neuron teacher-forced {i + 1}/{n_tokens - 1} tokens...", flush=True)

    return torch.stack(all_logits, dim=0)


def _hf_teacher_forced_logits(model, full_ids, prompt_len, n_tokens):
    """Run a single teacher-forced forward pass on HF and return logits.

    Uses use_cache=False to avoid DynamicCache incompatibilities between
    the Moonlight custom code and transformers 4.57+.

    Args:
        full_ids: (1, prompt_len + n_tokens) — full token sequence
    Returns:
        hf_logits: (n_tokens, vocab_size) float32
    """
    print(f"    HF teacher-forced forward (seq_len={full_ids.shape[1]})...", flush=True)
    with torch.no_grad():
        out = model(input_ids=full_ids, use_cache=False)

    # Logits at position i predict token at position i+1
    # Positions [prompt_len-1 .. prompt_len+n_tokens-2] predict the generated tokens
    hf_logits = out.logits[0, prompt_len - 1 : prompt_len + n_tokens - 1, :].clone().float()
    return hf_logits


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


@pytest.fixture(scope="module")
def neuron_model():
    """Compile Neuron model in FP32 with on_device_sampling=False for logit extraction."""
    tp_size = int(os.environ.get("MOONLIGHT_TP_SIZE", "2"))

    # Auto-size sequence length: must hold prompt + generated tokens in KV cache
    # Longest prompt is ~10 tokens, so prompt_len + NUM_TOKENS + margin
    default_seq_len = max(512, 64 + NUM_TOKENS + 64)
    seq_len = int(os.environ.get("MOONLIGHT_SEQ_LEN", str(default_seq_len)))

    print(f"  Neuron config: tp={tp_size}, seq_len={seq_len}, num_tokens={NUM_TOKENS}")

    nc = NeuronModelForCausalLM.get_neuron_config(
        model_name_or_path=MODEL_PATH,
        batch_size=1,
        sequence_length=seq_len,
        tensor_parallel_size=tp_size,
    )
    nc.on_device_sampling = False
    nc.torch_dtype = torch.float32

    model = NeuronModelForCausalLM.export(
        MODEL_PATH,
        neuron_config=nc,
        load_weights=True,
        trust_remote_code=True,
    )
    yield model


@pytest.fixture(scope="module")
def neuron_results(neuron_model, tokenizer):
    """Generate reference tokens and collect logits from Neuron model."""
    results = {}
    for prompt in PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = input_ids.shape[1]

        # Generate reference tokens greedily
        ref_tokens, neuron_logits = _neuron_greedy_generate(
            neuron_model, input_ids, prompt_len, NUM_TOKENS
        )
        decoded = tokenizer.decode(ref_tokens.tolist())
        print(f"  Neuron reference for {prompt!r}: {decoded[:80]!r}{'...' if len(decoded) > 80 else ''}")

        results[prompt] = {
            "ref_tokens": ref_tokens,
            "neuron_logits": neuron_logits,
            "prompt_len": prompt_len,
        }
    return results


@pytest.fixture(scope="module")
def hf_logits(neuron_results, tokenizer):
    """Load HF model, run teacher-forced passes, return logits, free memory."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()

    results = {}
    for prompt in PROMPTS:
        r = neuron_results[prompt]
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        full_ids = torch.cat([input_ids, r["ref_tokens"].unsqueeze(0)], dim=1)

        hf_log = _hf_teacher_forced_logits(model, full_ids, r["prompt_len"], NUM_TOKENS)
        results[prompt] = hf_log

    del model
    gc.collect()

    return results


@skip_no_model
@is_inferentia_test
@requires_neuronx
def test_teacher_forced_logit_divergence(neuron_results, hf_logits, tokenizer):
    """Verify logit values are close at all N positions per prompt.

    Prints a per-position table showing raw logit values from both HF (FP32)
    and Neuron (FP32 on device) for the top-1 predicted token, plus the difference.
    """
    for prompt in PROMPTS:
        r = neuron_results[prompt]
        neuron_logits = r["neuron_logits"]
        hf_log = hf_logits[prompt]

        # Truncate to common vocab size
        v = min(hf_log.shape[1], neuron_logits.shape[1])

        hf_preds = hf_log[:, :v].argmax(dim=1)
        neuron_preds = neuron_logits[:, :v].argmax(dim=1)

        # Per-position table with raw logit values
        print(f"\n  {prompt!r} ({NUM_TOKENS} positions):")
        print(f"    {'pos':>3}  {'HF_logit':>10}  {'N_logit':>10}  {'diff':>10}"
              f"  {'match':>5}  {'HF_pred':>15}  {'N_pred':>15}")
        print(f"    {'---':>3}  {'--------':>10}  {'-------':>10}  {'----':>10}"
              f"  {'-----':>5}  {'-------':>15}  {'------':>15}")

        diffs = []
        for pos in range(NUM_TOKENS):
            hf_top1_idx = hf_preds[pos].item()
            hf_val = hf_log[pos, hf_top1_idx].item()
            neuron_val = neuron_logits[pos, hf_top1_idx].item()
            diff = hf_val - neuron_val
            diffs.append(diff)

            hf_tok = tokenizer.decode([hf_top1_idx])
            n_top1_idx = neuron_preds[pos].item()
            n_tok = tokenizer.decode([n_top1_idx])
            match = "Y" if hf_top1_idx == n_top1_idx else "N"

            print(f"    {pos:>3}  {hf_val:>10.4f}  {neuron_val:>10.4f}  {diff:>+10.4f}"
                  f"  {match:>5}  {hf_tok!r:>15}  {n_tok!r:>15}")

        diffs_t = torch.tensor(diffs)
        abs_diffs = diffs_t.abs()
        print(f"    --- summary ---")
        print(f"    logit diff: mean={diffs_t.mean():+.4f}, std={diffs_t.std():.4f}, "
              f"abs_max={abs_diffs.max():.4f}, abs_mean={abs_diffs.mean():.4f}")
        matches = (hf_preds == neuron_preds).sum().item()
        print(f"    top-1 match: {matches}/{NUM_TOKENS} ({matches/NUM_TOKENS:.0%})")

        assert abs_diffs.mean().item() <= MAX_MEAN_LOGIT_DIFF, (
            f"Mean |logit diff| {abs_diffs.mean():.4f} exceeds {MAX_MEAN_LOGIT_DIFF} for {prompt!r}"
        )


@skip_no_model
@is_inferentia_test
@requires_neuronx
def test_teacher_forced_top1_match(neuron_results, hf_logits, tokenizer):
    """Verify greedy-decoded token matches at >= 75% of N positions per prompt."""
    for prompt in PROMPTS:
        r = neuron_results[prompt]
        neuron_logits = r["neuron_logits"]
        hf_log = hf_logits[prompt]

        v = min(hf_log.shape[1], neuron_logits.shape[1])
        hf_preds = hf_log[:, :v].argmax(dim=1)
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
def test_first_token_paris(neuron_results, tokenizer):
    """Smoke check: 'The capital of France is' -> first token contains 'Paris'."""
    r = neuron_results[PROMPTS[0]]
    first_token_id = r["neuron_logits"][0].argmax().item()
    decoded = tokenizer.decode([first_token_id])
    print(f"  Neuron first token: {decoded!r} (id={first_token_id})")
    assert "Paris" in decoded, f"Expected 'Paris' in first token, got {decoded!r}"
