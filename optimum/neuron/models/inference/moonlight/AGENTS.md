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
- **Expert parallelism**: MoE layers use `_build_moe()` (not `initialize_moe_module`) with a custom `MoonlightRouter` that applies top-K normalization, `routed_scaling_factor=2.446`, and `e_score_correction_bias` for accurate `noaux_tc` routing.
- **RoPE interleave transpose**: HF Moonlight stores q_pe/k_pe in interleaved layout `[r0,i0,r1,i1,...]` but Optimum's shared `apply_rotary_pos_emb` expects split layout `[r0,r1,...,i0,i1,...]`. `NeuronMoonlightAttention.forward` transposes before calling `apply_rotary_pos_emb`.
- **Dense layer 0**: Uses `NeuronMoonlightDenseMLP` with `dense_intermediate_size=11264`.
- **State dict remaps**: Router rename, `e_score_correction_bias` rename, expert weight fusion (gate_up_proj), expert stacking (down_proj).

### Removed NxDI-specific infrastructure
- **MoonlightNeuronConfig / MoonlightInferenceConfig** - Replaced by Optimum's `NxDNeuronConfig` with config overrides in `__init__`
- **ModuleMarkerStartWrapper / ModuleMarkerEndWrapper** - Compiler hints not needed in Optimum path
- **DeepseekV3Attention inheritance** - Replaced with self-contained MLA implementation
- **get_rmsnorm_cls() CPU/NXD switching** - Uses `NeuronRMSNorm` directly
- **NeuronBaseModel / NeuronBaseForCausalLM** - Replaced by `NxDDecoderModelForCausalLM` / `NxDModelForCausalLM`
- **Custom compiler args function** - Uses class method override pattern

### What Optimum Neuron Keeps
- MLA weight absorption pattern (Q-absorb, V-absorb via einsum)
- MoE with shared experts via custom `MoonlightRouter` + `_build_moe()` (moe_v2)
- Dense layer 0 with SiLU-gated MLP
- Expert weight fusion in state dict conversion
- KV cache format: concatenated `(k_pe, compressed_kv)` in single slot

## Key files
- [modeling_moonlight.py](modeling_moonlight.py) - All model code
- [../backend/modules/attention/utils.py](../backend/modules/attention/utils.py) - RotaryEmbedding, apply_rotary_pos_emb, manual_softmax
- [../backend/modules/moe_v2.py](../backend/modules/moe_v2.py) - MoE with shared experts
- [../backend/modules/decoder/modeling_decoder.py](../backend/modules/decoder/modeling_decoder.py) - Base decoder classes

## Tests
- [tests/test_moonlight_smoke.py](tests/test_moonlight_smoke.py) - CPU smoke tests (config, state dict conversion, registration)
- [tests/test_moonlight_on_device.py](tests/test_moonlight_on_device.py) - On-device: config, forward, greedy generation (" Paris")
- [tests/test_moonlight_export.py](tests/test_moonlight_export.py) - On-device: export → save → reload → forward
- [tests/test_moonlight_logit_divergence.py](tests/test_moonlight_logit_divergence.py) - Teacher-forced logit comparison (HF FP32 vs Neuron FP32). Configurable via `MOONLIGHT_NUM_TOKENS` env var. 100% top-1 match in FP32 at 32 tokens (max logit diff 0.0008).

## Numerical accuracy
- **FP32**: Neuron matches HF CPU exactly across 96 positions (abs_mean=0.0001, abs_max=0.0008). All divergence is graph-level floating-point noise.
- **BF16**: KV cache precision causes logit drift at later positions (mean ~1.7, 81-94% top-1 match). This is expected BF16 behavior, not an implementation bug.
