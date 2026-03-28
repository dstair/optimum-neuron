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
"""On-device tests for Moonlight 16B-A3B (DeepSeek V3 architecture) on Neuron.

Requires trn2 with at least 2 NeuronCores and MOONLIGHT_MODEL_PATH set.
Skipped in CI (no tiny-random checkpoint available for this architecture).

Usage:
    # Export from scratch (~5 min compilation):
    MOONLIGHT_MODEL_PATH=/path/to/Moonlight-16B-A3B \
        pytest tests/decoder/test_moonlight_on_device.py -v -s

    # Reuse pre-compiled model (fast):
    MOONLIGHT_MODEL_PATH=/path/to/Moonlight-16B-A3B \
    MOONLIGHT_COMPILED_PATH=/tmp/moonlight_accuracy_compiled \
        pytest tests/decoder/test_moonlight_on_device.py -v -s
"""

import os

import pytest
import torch
from transformers import AutoTokenizer

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


MODEL_PATH = os.environ.get("MOONLIGHT_MODEL_PATH")
COMPILED_PATH = os.environ.get("MOONLIGHT_COMPILED_PATH")

skip_no_model = pytest.mark.skipif(MODEL_PATH is None, reason="MOONLIGHT_MODEL_PATH not set")


@pytest.fixture(scope="module")
def neuron_model():
    if COMPILED_PATH and os.path.isdir(COMPILED_PATH):
        model = NeuronModelForCausalLM.from_pretrained(COMPILED_PATH)
    else:
        nc = NeuronModelForCausalLM.get_neuron_config(
            model_name_or_path=MODEL_PATH,
            batch_size=1,
            sequence_length=4096,
            tensor_parallel_size=2,
        )
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
def test_neuron_config(neuron_model):
    assert neuron_model.neuron_config.tp_degree == 2
    assert neuron_model.neuron_config.batch_size == 1


@skip_no_model
@is_inferentia_test
@requires_neuronx
def test_forward(neuron_model):
    input_ids = torch.ones((1, 10), dtype=torch.int64)
    position_ids = torch.arange(10).unsqueeze(0)
    seq_ids = torch.tensor([0])
    sampling_params = torch.ones((1, 3))
    outputs = neuron_model(
        input_ids=input_ids,
        position_ids=position_ids,
        seq_ids=seq_ids,
        sampling_params=sampling_params,
    )
    assert outputs is not None


@skip_no_model
@is_inferentia_test
@requires_neuronx
def test_greedy_generation(neuron_model):
    """Logit validation: verify greedy generation matches HF CPU reference."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    inputs = tokenizer("The capital of France is", return_tensors="pt")
    outputs = neuron_model.generate(**inputs, do_sample=False, max_new_tokens=10)
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :].tolist()
    generated = tokenizer.decode(generated_ids)
    # First token must be " Paris" — validated against HF CPU reference
    assert generated.lstrip().startswith("Paris"), f"Expected 'Paris...', got '{generated.strip()}'"
