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
"""Smoke tests for Moonlight 16B-A3B Optimum Neuron port (CPU-only)."""

import pytest
import torch
from transformers import AutoConfig

from optimum.neuron.models.inference.moonlight.modeling_moonlight import (
    MoonlightNxDModelForCausalLM,
    convert_moonlight_hf_to_neuron_state_dict,
)

MODEL_PATH = "/home/ubuntu/environment/models/Moonlight-16B-A3B"


class TestMoonlightConfigLoading:
    """Test 1: Verify HF config has the expected MLA and MoE attributes."""

    @pytest.fixture(autouse=True)
    def load_config(self):
        self.config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    def test_mla_attributes(self):
        assert self.config.kv_lora_rank == 512
        assert self.config.qk_nope_head_dim == 128
        assert self.config.qk_rope_head_dim == 64
        assert self.config.v_head_dim == 128

    def test_moe_attributes(self):
        assert self.config.n_routed_experts == 64
        assert self.config.num_experts_per_tok == 6
        assert self.config.n_shared_experts == 2

    def test_model_type(self):
        assert self.config.model_type == "deepseek_v3"


class TestStateDictConversion:
    """Test 2: Verify state dict conversion logic (mock weights, CPU-only)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.hidden_size = 16
        self.intermediate_size = 8
        self.moe_intermediate_size = 4
        self.num_experts = 2
        self.num_layers = 2  # 1 dense + 1 MoE

        class MockConfig:
            num_hidden_layers = 2
            num_local_experts = 2
            first_k_dense_replace = 1

        class MockNeuronConfig:
            tp_degree = 1
            glu_mlp = True

        self.config = MockConfig()
        self.neuron_config = MockNeuronConfig()

        # Build a mock state dict with correct key structure
        self.state_dict = {}
        for l in range(self.num_layers):
            # Attention weights (just placeholders)
            self.state_dict[f"layers.{l}.self_attn.q_proj.weight"] = torch.randn(32, self.hidden_size)

            if l >= 1:  # MoE layer
                # Router
                self.state_dict[f"layers.{l}.mlp.gate.weight"] = torch.randn(self.num_experts, self.hidden_size)
                self.state_dict[f"layers.{l}.mlp.gate.e_score_correction_bias"] = torch.randn(self.num_experts)
                # Per-expert weights
                for e in range(self.num_experts):
                    self.state_dict[f"layers.{l}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
                        self.moe_intermediate_size, self.hidden_size
                    )
                    self.state_dict[f"layers.{l}.mlp.experts.{e}.up_proj.weight"] = torch.randn(
                        self.moe_intermediate_size, self.hidden_size
                    )
                    self.state_dict[f"layers.{l}.mlp.experts.{e}.down_proj.weight"] = torch.randn(
                        self.hidden_size, self.moe_intermediate_size
                    )
                # Shared expert weights (should be left untouched)
                self.state_dict[f"layers.{l}.mlp.shared_experts.gate_proj.weight"] = torch.randn(
                    self.moe_intermediate_size, self.hidden_size
                )

    def test_rank_tensors_added(self):
        result = convert_moonlight_hf_to_neuron_state_dict(
            dict(self.state_dict), self.config, self.neuron_config
        )
        assert "rank_util.rank" in result
        for l in range(self.num_layers):
            assert f"layers.{l}.self_attn.rank_util.rank" in result

    def test_router_renamed(self):
        result = convert_moonlight_hf_to_neuron_state_dict(
            dict(self.state_dict), self.config, self.neuron_config
        )
        # Layer 0 (dense) should NOT have router
        assert "layers.0.mlp.router.linear_router.weight" not in result
        # Layer 1 (MoE) should have renamed router
        assert "layers.1.mlp.router.linear_router.weight" in result
        assert "layers.1.mlp.gate.weight" not in result

    def test_e_score_correction_bias_renamed(self):
        result = convert_moonlight_hf_to_neuron_state_dict(
            dict(self.state_dict), self.config, self.neuron_config
        )
        assert "layers.1.mlp.router.e_score_correction_bias" in result
        assert "layers.1.mlp.gate.e_score_correction_bias" not in result

    def test_expert_weights_fused(self):
        result = convert_moonlight_hf_to_neuron_state_dict(
            dict(self.state_dict), self.config, self.neuron_config
        )
        # gate_up_proj: (num_experts, hidden_size, 2 * moe_intermediate_size)
        gate_up = result["layers.1.mlp.expert_mlps.mlp_op.gate_up_proj.weight"]
        assert gate_up.shape == (self.num_experts, self.hidden_size, 2 * self.moe_intermediate_size)

        # down_proj: (num_experts, moe_intermediate_size, hidden_size)
        down = result["layers.1.mlp.expert_mlps.mlp_op.down_proj.weight"]
        assert down.shape == (self.num_experts, self.moe_intermediate_size, self.hidden_size)

        # Per-expert keys should be deleted
        for e in range(self.num_experts):
            assert f"layers.1.mlp.experts.{e}.gate_proj.weight" not in result
            assert f"layers.1.mlp.experts.{e}.up_proj.weight" not in result
            assert f"layers.1.mlp.experts.{e}.down_proj.weight" not in result

    def test_shared_experts_untouched(self):
        result = convert_moonlight_hf_to_neuron_state_dict(
            dict(self.state_dict), self.config, self.neuron_config
        )
        assert "layers.1.mlp.shared_experts.gate_proj.weight" in result

    def test_dense_layer_untouched(self):
        original_q = self.state_dict["layers.0.self_attn.q_proj.weight"].clone()
        result = convert_moonlight_hf_to_neuron_state_dict(
            dict(self.state_dict), self.config, self.neuron_config
        )
        assert torch.equal(result["layers.0.self_attn.q_proj.weight"], original_q)


class TestRegistration:
    """Test 3: Verify the model is importable and registered."""

    def test_import_model_class(self):
        from optimum.neuron.models.inference.moonlight.modeling_moonlight import MoonlightNxDModelForCausalLM

        assert MoonlightNxDModelForCausalLM._model_cls is not None

    def test_import_registration(self):
        from optimum.neuron.models.inference.auto_models import MoonlightNeuronModelForCausalLM

        assert MoonlightNeuronModelForCausalLM is not None
