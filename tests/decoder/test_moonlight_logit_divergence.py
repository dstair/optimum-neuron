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
"""Logit divergence test for Moonlight 16B-A3B (DeepSeek V3 architecture).

Compares first-token prediction logits between:
- HF Transformers model (BF16 on CPU)
- Neuron model (BF16 on device, on_device_sampling=False for logit access)

Metrics reported:
- Cosine similarity per prompt (threshold >= 0.99)
- Top-1 token match (at least 2 of 3 prompts must agree)
- Top-5 overlap per prompt

Prior validated baseline (3 fixes applied): cos_sim min=0.994, mean=0.997.
Remaining mismatches are BF16 hardware tiebreakers on near-equal logits.

Requires trn2 with at least 2 NeuronCores and MOONLIGHT_MODEL_PATH set.
Total runtime: ~25 min (HF model load ~2 min, Neuron compilation ~20 min).

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

# Thresholds derived from prior validation (cos_sim min=0.994, mean=0.997)
COS_SIM_THRESHOLD = 0.99
TOP1_MATCH_MINIMUM = 2  # out of 3 prompts (BF16 tiebreakers may cause 1 mismatch)


@pytest.fixture(scope="module")
def hf_reference():
    """Load HF Moonlight in BF16 on CPU, capture reference logits, then free memory.

    This fixture MUST complete before the Neuron fixture starts compilation,
    because both require significant RAM. Ordering is enforced by making
    the neuron_model fixture depend on this one.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()

    ref_logits = {}
    ref_tokens = {}
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, attention_mask=inputs["attention_mask"])
        logits = out.logits[0, -1, :].clone().float()
        ref_logits[prompt] = logits
        ref_tokens[prompt] = logits.argmax().item()

    del model
    gc.collect()

    return ref_logits, ref_tokens, tokenizer


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


def _get_neuron_logits(model, tokenizer, prompt):
    """Run a single prompt through the Neuron model and return logits at the last input position."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    seq_ids = torch.tensor([0])
    sampling_params = torch.ones((1, 3))

    model.reset()
    outputs = model(
        input_ids=input_ids,
        position_ids=position_ids,
        seq_ids=seq_ids,
        sampling_params=sampling_params,
    )
    return outputs[0, -1, :].clone().float()


@skip_no_model
@is_inferentia_test
@requires_neuronx
def test_cosine_similarity(hf_reference, neuron_model):
    """Verify logit vectors are highly similar (cosine similarity >= 0.99)."""
    ref_logits, _, tokenizer = hf_reference

    cos_sims = {}
    for prompt in PROMPTS:
        neuron_logits = _get_neuron_logits(neuron_model, tokenizer, prompt)
        ref = ref_logits[prompt]

        # Truncate to minimum length in case of vocab padding differences
        min_len = min(ref.shape[0], neuron_logits.shape[0])
        cos_sim = torch.nn.functional.cosine_similarity(
            ref[:min_len].unsqueeze(0),
            neuron_logits[:min_len].unsqueeze(0),
        ).item()
        cos_sims[prompt] = cos_sim

    # Report all values for debugging
    for prompt, sim in cos_sims.items():
        print(f"  cos_sim({prompt!r}) = {sim:.6f}")
    print(f"  min={min(cos_sims.values()):.6f}, mean={sum(cos_sims.values())/len(cos_sims):.6f}")

    assert min(cos_sims.values()) >= COS_SIM_THRESHOLD, (
        f"Cosine similarity below threshold {COS_SIM_THRESHOLD}: {cos_sims}"
    )


@skip_no_model
@is_inferentia_test
@requires_neuronx
def test_top1_token_match(hf_reference, neuron_model):
    """Verify top-1 predicted token matches for most prompts."""
    ref_logits, ref_tokens, tokenizer = hf_reference

    matches = 0
    for prompt in PROMPTS:
        neuron_logits = _get_neuron_logits(neuron_model, tokenizer, prompt)
        neuron_token = neuron_logits.argmax().item()
        hf_token = ref_tokens[prompt]
        match = neuron_token == hf_token
        if match:
            matches += 1
        print(
            f"  {prompt!r}: HF={hf_token} ({tokenizer.decode([hf_token])!r}), "
            f"Neuron={neuron_token} ({tokenizer.decode([neuron_token])!r}), match={match}"
        )

    print(f"  Top-1 matches: {matches}/{len(PROMPTS)}")
    assert matches >= TOP1_MATCH_MINIMUM, (
        f"Only {matches}/{len(PROMPTS)} top-1 matches (need {TOP1_MATCH_MINIMUM})"
    )


@skip_no_model
@is_inferentia_test
@requires_neuronx
def test_top5_overlap(hf_reference, neuron_model):
    """Verify top-5 predicted tokens overlap significantly."""
    ref_logits, _, tokenizer = hf_reference

    for prompt in PROMPTS:
        neuron_logits = _get_neuron_logits(neuron_model, tokenizer, prompt)
        ref = ref_logits[prompt]

        min_len = min(ref.shape[0], neuron_logits.shape[0])
        hf_top5 = set(ref[:min_len].topk(5).indices.tolist())
        neuron_top5 = set(neuron_logits[:min_len].topk(5).indices.tolist())
        overlap = len(hf_top5 & neuron_top5)

        hf_top5_tokens = [tokenizer.decode([t]) for t in hf_top5]
        neuron_top5_tokens = [tokenizer.decode([t]) for t in neuron_top5]
        print(f"  {prompt!r}: HF top5={hf_top5_tokens}, Neuron top5={neuron_top5_tokens}, overlap={overlap}/5")

        assert overlap >= 3, f"Top-5 overlap too low for {prompt!r}: {overlap}/5"


@skip_no_model
@is_inferentia_test
@requires_neuronx
def test_first_token_paris(hf_reference, neuron_model):
    """Smoke check: 'The capital of France is' -> first token contains 'Paris'."""
    _, ref_tokens, tokenizer = hf_reference
    neuron_logits = _get_neuron_logits(neuron_model, tokenizer, PROMPTS[0])
    neuron_token = neuron_logits.argmax().item()
    decoded = tokenizer.decode([neuron_token])
    print(f"  Neuron first token: {decoded!r} (id={neuron_token})")
    assert "Paris" in decoded, f"Expected 'Paris' in first token, got {decoded!r}"
