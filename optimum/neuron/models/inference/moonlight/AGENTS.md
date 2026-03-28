# Moonlight Inference Model Guide (MoE + MLA)

This directory contains the Neuron-optimized Moonlight 16B-A3B inference implementation. Moonlight uses the DeepSeek V3 architecture (Multi-head Latent Attention + sparse MoE). It follows [HF Transformers](https://github.com/huggingface/transformers) architecture with Neuron-specific changes from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference).

## HF reference
- Moonlight model: MoonshotAI/Moonlight-16B-A3B (uses `model_type: "deepseek_v3"`)
- HF config class: `DeepseekV3Config` (auto_map in config.json)

## Architecture overview
- **27 layers**: 1 dense (layer 0) + 26 MoE layers
- **MLA (Multi-head Latent Attention)**: Low-rank KV compression (`kv_lora_rank=512`), no query LoRA (`q_lora_rank=None`)
- **MoE**: 64 routed experts + 2 shared experts, top-6 per token, sigmoid routing
- **Plain RoPE**: `rope_theta=50000`, no Yarn scaling, 8K context
- **Vocabulary**: 163,840 tokens

## What differs vs HF
- **Custom MLA attention**: `NeuronMoonlightAttention` does NOT extend `NeuronAttentionBase` (GQA projections are incompatible with MLA). Instead, it implements MLA with weight absorption directly using `ColumnParallelLinear`/`RowParallelLinear`.
- **KV cache override**: `num_key_value_heads=1`, `head_dim=576` (rope_dim + kv_lora_rank) so KVCacheManager allocates MLA-compatible cache.
- **Expert parallelism**: MoE layers use `initialize_moe_module` from `moe_v2` with sigmoid routing and shared experts.
- **Dense layer 0**: Uses `NeuronMoonlightDenseMLP` with `dense_intermediate_size=11264`.
- **State dict remaps**: Router rename, expert weight fusion (gate_up_proj), expert stacking (down_proj).

### Removed NxDI-specific infrastructure
- **MoonlightNeuronConfig / MoonlightInferenceConfig** - Replaced by Optimum's `NxDNeuronConfig` with config overrides in `__init__`
- **ModuleMarkerStartWrapper / ModuleMarkerEndWrapper** - Compiler hints not needed in Optimum path
- **DeepseekV3Attention inheritance** - Replaced with self-contained MLA implementation
- **get_rmsnorm_cls() CPU/NXD switching** - Uses `NeuronRMSNorm` directly
- **NeuronBaseModel / NeuronBaseForCausalLM** - Replaced by `NxDDecoderModelForCausalLM` / `NxDModelForCausalLM`
- **Custom compiler args function** - Uses class method override pattern

### What Optimum Neuron Keeps
- MLA weight absorption pattern (Q-absorb, V-absorb via einsum)
- MoE with shared experts via `initialize_moe_module` (moe_v2)
- Dense layer 0 with SiLU-gated MLP
- Expert weight fusion in state dict conversion
- KV cache format: concatenated `(k_pe, compressed_kv)` in single slot

## Key files
- [modeling_moonlight.py](modeling_moonlight.py) - All model code
- [../backend/modules/attention/utils.py](../backend/modules/attention/utils.py) - RotaryEmbedding, apply_rotary_pos_emb, manual_softmax
- [../backend/modules/moe_v2.py](../backend/modules/moe_v2.py) - MoE with shared experts
- [../backend/modules/decoder/modeling_decoder.py](../backend/modules/decoder/modeling_decoder.py) - Base decoder classes
