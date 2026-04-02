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
"""Export/save/reload test for Moonlight 16B-A3B on Neuron.

Separated from test_moonlight_on_device.py to ensure exclusive device access
(the 16B model uses ~39 GB, leaving no room for a second copy on trn2.3xlarge).

Usage:
    MOONLIGHT_MODEL_PATH=/path/to/Moonlight-16B-A3B \
        pytest optimum/neuron/models/inference/moonlight/tests/test_moonlight_export.py -v -s
"""

import os
from tempfile import TemporaryDirectory

import pytest
import torch

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


MODEL_PATH = os.environ.get("MOONLIGHT_MODEL_PATH")


@pytest.mark.skipif(MODEL_PATH is None, reason="MOONLIGHT_MODEL_PATH not set")
@is_inferentia_test
@requires_neuronx
def test_export_save_reload():
    """Test the full export -> save -> reload -> forward cycle."""
    nc = NeuronModelForCausalLM.get_neuron_config(
        model_name_or_path=MODEL_PATH,
        batch_size=1,
        sequence_length=4096,
        tensor_parallel_size=2,
    )
    with TemporaryDirectory() as tmpdir:
        model = NeuronModelForCausalLM.export(
            MODEL_PATH,
            neuron_config=nc,
            load_weights=False,
            trust_remote_code=True,
        )
        model.save_pretrained(tmpdir)
        del model
        reloaded = NeuronModelForCausalLM.from_pretrained(tmpdir)
        input_ids = torch.ones((1, 10), dtype=torch.int64)
        position_ids = torch.arange(10).unsqueeze(0)
        seq_ids = torch.tensor([0])
        sampling_params = torch.ones((1, 3))
        outputs = reloaded(
            input_ids=input_ids,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
        )
        assert outputs is not None
