# coding=utf-8
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
# Adapted from https://github.com/aws-neuron/neuronx-distributed-inference
"""Moonlight 16B-A3B model for NXD inference.

Moonlight uses the DeepSeek V3 architecture: Multi-head Latent Attention (MLA)
with sparse Mixture-of-Experts (MoE). Key differences from standard GQA models:
- MLA compresses KV into a low-rank representation (kv_lora_rank=512)
- Weight absorption eliminates the need to expand KV per-head
- 27 layers: 1 dense (layer 0) + 26 MoE layers (64 routed + 2 shared experts)
"""

import gc

import torch
from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.moe_configs import RoutedExpertsMLPOpsConfig
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.modules.moe.shared_experts import SharedExperts
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
    SPMDRank,
)
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_group,
)
from torch import nn
from transformers.activations import ACT2FN

from ..backend.config import NxDNeuronConfig
from ..backend.modules.attention.utils import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    manual_softmax,
)
from ..backend.modules.decoder import NxDDecoderModelForCausalLM, NxDModelForCausalLM
from ..backend.modules.rms_norm import NeuronRMSNorm


# Note: initialize_moe_module was replaced by direct MoE construction
# in NeuronMoonlightDecoderLayer._build_moe() to support routed_scaling_factor.


def convert_moonlight_hf_to_neuron_state_dict(state_dict, config, neuron_config):
    """Convert HuggingFace Moonlight state dict to Neuron-compatible format.

    Transformations:
    1. Adds rank utility tensors for TP sharding
    2. Renames router weights: gate.weight -> router.linear_router.weight
    3. Fuses gate_proj + up_proj into gate_up_proj for each expert
    4. Stacks down_proj weights across experts
    5. Skips dense layers (layer 0)
    """
    assert neuron_config.glu_mlp is True, "Only GLU MLP is supported"

    tp_degree = neuron_config.tp_degree
    first_k_dense = getattr(config, "first_k_dense_replace", 1)

    # Model-level rank tensor
    state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

    for l in range(config.num_hidden_layers):  # noqa: E741
        # Per-layer attention rank tensor
        state_dict[f"layers.{l}.self_attn.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        # Skip dense layers (no MoE conversion needed)
        if l < first_k_dense:
            continue

        # Rename router weights
        router_key = f"layers.{l}.mlp.gate.weight"
        if router_key in state_dict:
            state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = state_dict[router_key].detach().clone()
            del state_dict[router_key]

        # Rename e_score_correction_bias (used by noaux_tc routing)
        bias_key = f"layers.{l}.mlp.gate.e_score_correction_bias"
        if bias_key in state_dict:
            state_dict[f"layers.{l}.mlp.router.e_score_correction_bias"] = state_dict[bias_key].detach().clone()
            del state_dict[bias_key]

        # Check if expert weights exist
        expert_gate_key = f"layers.{l}.mlp.experts.0.gate_proj.weight"
        if expert_gate_key not in state_dict:
            continue

        intermediate_size, hidden_size = state_dict[expert_gate_key].shape
        device = state_dict[expert_gate_key].device
        dtype = state_dict[expert_gate_key].dtype

        # Fuse gate_proj + up_proj into gate_up_proj
        gate_up_proj = torch.empty(
            config.num_local_experts,
            hidden_size,
            2 * intermediate_size,
            dtype=dtype,
            device=device,
        )
        for e in range(config.num_local_experts):
            gate_key = f"layers.{l}.mlp.experts.{e}.gate_proj.weight"
            up_key = f"layers.{l}.mlp.experts.{e}.up_proj.weight"

            gate_proj_weights = state_dict[gate_key].T.detach().clone()
            up_proj_weights = state_dict[up_key].T.detach().clone()

            gate_up_proj_slice = torch.narrow(gate_up_proj, 0, e, 1)
            gate_proj_slice = torch.narrow(gate_up_proj_slice, 2, 0, intermediate_size)
            gate_proj_slice.copy_(gate_proj_weights)
            up_proj_slice = torch.narrow(gate_up_proj_slice, 2, intermediate_size, intermediate_size)
            up_proj_slice.copy_(up_proj_weights)

            del state_dict[gate_key]
            del state_dict[up_key]
        state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

        # Stack down_proj across experts
        down_proj = torch.empty(
            config.num_local_experts,
            intermediate_size,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        for e in range(config.num_local_experts):
            down_key = f"layers.{l}.mlp.experts.{e}.down_proj.weight"
            down_proj_weights = state_dict[down_key].T.detach().clone()
            down_proj_slice = torch.narrow(down_proj, 0, e, 1)
            down_proj_slice.copy_(down_proj_weights)
            del state_dict[down_key]
        state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        gc.collect()

    return state_dict


class MoonlightRouter(RouterTopK):
    """Router with noaux_tc selection, normalization, and scaling for Moonlight.

    Moonlight/DeepSeek V3 uses noaux_tc routing:
    1. Compute sigmoid(logits) as affinities
    2. Add e_score_correction_bias to affinities for expert SELECTION
    3. Select top-K experts from bias-adjusted scores
    4. Gather weights from ORIGINAL affinities (no bias), normalize, scale

    The base RouterTopK selects on raw logits and doesn't support any of this.
    """

    def __init__(self, routed_scaling_factor=1.0, norm_topk_prob=True, **kwargs):
        super().__init__(**kwargs)
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(kwargs["num_experts"], dtype=kwargs.get("dtype", torch.float32))
        )

    def forward(self, hidden_states):
        router_logits = self.get_router_logits(hidden_states)
        expert_affinities = self.apply_activation_fn(router_logits)

        # noaux_tc: select experts using bias-adjusted scores
        scores_for_choice = expert_affinities + self.e_score_correction_bias.unsqueeze(0)
        _, expert_index = torch.topk(scores_for_choice, self.top_k)

        # Gather weights from ORIGINAL affinities (no bias), normalize, and scale
        topk_weights = expert_affinities.gather(1, expert_index)
        if self.norm_topk_prob and self.top_k > 1:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor

        # Write normalized+scaled affinities back to full tensor
        expert_affinities = torch.zeros_like(expert_affinities)
        expert_affinities.scatter_(1, expert_index, topk_weights)

        expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)
        expert_index = expert_index.detach().to(dtype=torch.long)
        return router_logits, expert_affinities, expert_index


class NeuronMoonlightAttention(nn.Module):
    """Multi-head Latent Attention (MLA) for Moonlight.

    MLA compresses KV into a low-rank representation. This does NOT extend
    NeuronAttentionBase because GQA's QKV projections are incompatible with
    MLA's structure.

    Uses weight absorption (fusing kv_b_proj into Q and V via einsum) for
    efficient attention on compressed KV. The KV cache stores compressed
    representations; absorption avoids explicit decompression.
    """

    def __init__(self, config, neuron_config: NxDNeuronConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_total_heads = config.num_attention_heads
        self.tp_degree = neuron_config.tp_degree
        self.num_heads = self.num_total_heads // self.tp_degree

        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        self.softmax_scale = self.q_head_dim ** (-0.5)

        dtype = neuron_config.torch_dtype

        self.tensor_model_parallel_group = get_tensor_model_parallel_group()
        self.rank_util = SPMDRank(world_size=self.tp_degree)

        # Q projection (direct, no LoRA — q_lora_rank is None for Moonlight)
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_total_heads * self.q_head_dim,
            bias=False,
            gather_output=False,
            dtype=dtype,
        )

        # KV compression: projects to (kv_lora_rank + qk_rope_head_dim)
        # NOT parallel — single compressed KV shared across all heads
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            dtype=dtype,
        )
        self.kv_a_layernorm = NeuronRMSNorm(self.kv_lora_rank)

        # KV decompression weight (absorbed into Q and V via einsum, not called as layer)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_total_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            gather_output=False,
            dtype=dtype,
        )

        # Output projection
        self.o_proj = RowParallelLinear(
            self.num_total_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=dtype,
            reduce_dtype=dtype,
        )

        # Plain RoPE (no Yarn scaling for Moonlight)
        self.rotary_emb = RotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        active_mask: torch.LongTensor | None = None,
        cos_cache: torch.Tensor | None = None,
        sin_cache: torch.Tensor | None = None,
    ) -> tuple:
        bsz, q_len, _ = hidden_states.size()

        # Q projection and reshape
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        # KV compression
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

        # Split Q into nope (non-positional) and pe (positional) components
        q_nope, q_pe = torch.tensor_split(q, (self.qk_nope_head_dim,), dim=-1)

        # Split compressed KV into latent representation and positional key
        compressed_kv, k_pe = torch.tensor_split(compressed_kv, (self.kv_lora_rank,), dim=-1)
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # Convert rope dims from interleaved [r0,i0,r1,i1,...] to split [r0,r1,...,i0,i1,...]
        # layout before applying RoPE. HF Moonlight's apply_rotary_pos_emb includes this
        # transpose; the shared Neuron apply_rotary_pos_emb does not.
        b, h, s, d = q_pe.shape
        q_pe = q_pe.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        b, h, s, d = k_pe.shape
        k_pe = k_pe.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        # Apply RoPE to positional components only
        if cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.rotary_emb(q_pe, position_ids)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos_cache, sin_cache)

        # Weight absorption: decompose kv_b_proj weight into Q-absorb and V-absorb
        wkv_b = self.kv_b_proj.weight
        wkv_b = wkv_b.view(self.num_heads, -1, self.kv_lora_rank)
        q_absorb = wkv_b[:, : self.qk_nope_head_dim]  # (H, nope_dim, kv_lora_rank)
        out_absorb = wkv_b[:, self.qk_nope_head_dim :, :]  # (H, v_dim, kv_lora_rank)

        # Absorb Q-nope: project from head_dim to kv_lora_rank
        q_nope = torch.einsum("hdc,bhqd->bhqc", q_absorb, q_nope)

        # Compute attention scores using absorbed Q and compressed KV
        active_scores = torch.matmul(q_pe, k_pe.transpose(2, 3)) + torch.einsum(
            "bhqc,blc->bhql", q_nope, compressed_kv
        )
        active_scores *= self.softmax_scale

        if past_key_value is None:
            # Prefill: standard causal attention
            active_scores = torch.where(attention_mask, active_scores, -1e9)
            active_scores = nn.functional.softmax(active_scores, dim=-1, dtype=torch.float32).to(q.dtype)

            # V absorption: softmax @ compressed_kv, then project via out_absorb
            x = torch.einsum("bhql,blc->bhqc", active_scores, compressed_kv)
            attn_output = torch.einsum("bhqc,hdc->bhqd", x, out_absorb)
        else:
            # Decode: unpack cached KV
            cached = past_key_value[0].squeeze(1)  # (bsz, seq_len, rope_dim + kv_lora_rank)
            k_pe_prior, compressed_kv_prior = torch.tensor_split(cached, [self.qk_rope_head_dim], dim=-1)
            k_pe_prior = k_pe_prior.unsqueeze(1)  # (bsz, 1, seq_len, rope_dim)

            # Prior attention scores using absorption
            prior_scores = torch.matmul(q_pe, k_pe_prior.transpose(2, 3)) + torch.einsum(
                "bhqc,blc->bhql", q_nope, compressed_kv_prior
            )
            prior_scores *= self.softmax_scale

            is_speculation = position_ids.shape[-1] > 1

            # Softmax over prior + active scores with masking
            softmax_prior, softmax_active = manual_softmax(
                prior_scores,
                active_scores,
                prior_mask=attention_mask,
                active_mask=active_mask if is_speculation else None,
            )
            softmax_prior = softmax_prior.to(q.dtype)
            softmax_active = softmax_active.to(q.dtype)

            # V absorption for active and prior
            x_active = torch.einsum("bhql,blc->bhqc", softmax_active, compressed_kv)
            attn_active = torch.einsum("bhqc,hdc->bhqd", x_active, out_absorb)

            x_prior = torch.einsum("bhql,blc->bhqc", softmax_prior, compressed_kv_prior)
            attn_prior = torch.einsum("bhqc,hdc->bhqd", x_prior, out_absorb)

            attn_output = attn_prior + attn_active

        # BHSD -> BSHD -> merged heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        # Pack KV for cache: concatenate k_pe + compressed_kv into single slot
        # k_pe: (bsz, 1, q_len, rope_dim), compressed_kv: (bsz, q_len, kv_lora_rank)
        combined = torch.cat([k_pe.squeeze(1), compressed_kv], dim=-1).unsqueeze(1)
        new_kv = (combined, combined)  # K and V slots (V is duplicate for MLA)

        return attn_output, new_kv, cos_cache, sin_cache


class NeuronMoonlightDenseMLP(nn.Module):
    """Dense MLP for Moonlight layer 0 (non-MoE layer).

    Uses SiLU-gated architecture: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    """

    def __init__(self, config, neuron_config: NxDNeuronConfig):
        super().__init__()
        dtype = neuron_config.torch_dtype
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.dense_intermediate_size,
            bias=False,
            gather_output=False,
            dtype=dtype,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.dense_intermediate_size,
            bias=False,
            gather_output=False,
            dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            config.dense_intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=dtype,
            reduce_dtype=dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class NeuronMoonlightDecoderLayer(nn.Module):
    """Moonlight decoder layer with MLA attention and MoE/Dense MLP.

    Layer 0 uses a dense MLP; layers 1+ use Mixture-of-Experts.
    """

    def __init__(self, config, neuron_config: NxDNeuronConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronMoonlightAttention(config, neuron_config)

        first_k_dense = getattr(config, "first_k_dense_replace", 1)
        if layer_idx < first_k_dense:
            self.mlp = NeuronMoonlightDenseMLP(config, neuron_config)
            self._is_dense = True
        else:
            self.mlp = self._build_moe(config, neuron_config)
            self._is_dense = False

        self.input_layernorm = NeuronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = NeuronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @staticmethod
    def _build_moe(config, neuron_config):
        """Build MoE module with Moonlight-specific routing (normalize + scale)."""
        router = MoonlightRouter(
            routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
            norm_topk_prob=getattr(config, "norm_topk_prob", True),
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            dtype=torch.float32,
            act_fn="sigmoid",
        )
        expert_mlps = ExpertMLPsV2(
            routed_experts_mlp_config=RoutedExpertsMLPOpsConfig(
                num_experts=config.num_local_experts,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                top_k=config.num_experts_per_tok,
                hidden_act=config.hidden_act,
                glu_mlp=neuron_config.glu_mlp,
                normalize_top_k_affinities=False,  # Router handles normalization+scaling
            ),
            dtype=neuron_config.torch_dtype,
        )
        n_shared_experts = getattr(config, "n_shared_experts", 2)
        shared_experts = SharedExperts(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_shared_experts=n_shared_experts,
            hidden_act=config.hidden_act,
            dtype=neuron_config.torch_dtype,
            reduce_dtype=neuron_config.torch_dtype,
        )
        moe = MoE(router=router, expert_mlps=expert_mlps, shared_experts=shared_experts)
        moe.eval()
        return moe

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self._is_dense:
            hidden_states = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value, cos_cache, sin_cache


class NxDMoonlightModel(NxDDecoderModelForCausalLM):
    """Moonlight inner model (traced graph).

    Overrides config attributes before super().__init__ so the base class
    KVCacheManager allocates MLA-compatible cache slots.
    """

    def __init__(self, config, neuron_config: NxDNeuronConfig):
        # MLA KV cache: single head with concatenated rope + compressed KV
        config.num_key_value_heads = 1
        config.head_dim = config.qk_rope_head_dim + config.kv_lora_rank

        # Save dense intermediate size before MoE override.
        # Guard: if already set (re-init), don't overwrite with the mutated intermediate_size.
        if not hasattr(config, "dense_intermediate_size"):
            config.dense_intermediate_size = config.intermediate_size

        # MoE config mappings
        config.intermediate_size = config.moe_intermediate_size
        config.num_local_experts = config.n_routed_experts
        config.hidden_act = getattr(config, "hidden_act", "silu")
        config.first_k_dense_replace = getattr(config, "first_k_dense_replace", 1)

        super().__init__(config, neuron_config)

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            dtype=neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList(
            [
                NeuronMoonlightDecoderLayer(config, neuron_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = NeuronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not neuron_config.on_device_sampling,
            bias=False,
        )


_MOONLIGHT_SKIP_WEIGHTS = [
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.mlp.down_proj.weight",
]


def _patch_neuron_config_skip_weights(neuron_config):
    """Patch a NxDNeuronConfig instance to skip dense MLP layer 0 weights.

    Uses type-level property override so the instance returns the skip list
    without requiring a separately registered config subclass.
    """
    from ....configuration_utils import _KEY_FOR_NEURON_CONFIG

    cls = neuron_config.__class__
    if not hasattr(cls, "_moonlight_patched"):
        # Create a one-off subclass with the property override
        patched_cls = type(
            cls.__name__,
            (cls,),
            {
                "weights_to_skip_layout_optimization": property(lambda self: _MOONLIGHT_SKIP_WEIGHTS),
                "_moonlight_patched": True,
            },
        )
        # Register the patched class with the same key as NxDNeuronConfig
        # so serialization (to_dict) and deserialization (from_pretrained) work.
        _KEY_FOR_NEURON_CONFIG[patched_cls] = "NxDNeuronConfig"
        neuron_config.__class__ = patched_cls


class MoonlightNxDModelForCausalLM(NxDModelForCausalLM):
    """Moonlight causal LM wrapper for Optimum Neuron inference."""

    _model_cls = NxDMoonlightModel

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config, neuron_config):
        return convert_moonlight_hf_to_neuron_state_dict(state_dict, config, neuron_config)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_compiler_args(cls, neuron_config: NxDNeuronConfig):
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        compiler_args += " --auto-cast=none"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        return compiler_args

    @classmethod
    def _get_neuron_config(
        cls,
        checkpoint_id: str,
        checkpoint_revision: str,
        instance_type: str,
        batch_size: int,
        sequence_length: int,
        tensor_parallel_size: int,
        dtype: torch.dtype,
    ):
        continuous_batching = (batch_size > 1) if batch_size else False
        neuron_config = NxDNeuronConfig(
            checkpoint_id=checkpoint_id,
            checkpoint_revision=checkpoint_revision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tp_degree=tensor_parallel_size,
            torch_dtype=dtype,
            target=instance_type,
            on_device_sampling=True,
            glu_mlp=True,
            continuous_batching=continuous_batching,
        )
        _patch_neuron_config_skip_weights(neuron_config)
        return neuron_config
