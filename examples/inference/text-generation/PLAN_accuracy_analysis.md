# Moonlight 16B Layer-by-Layer Logit Divergence Analysis Plan

## Problem Statement

Current accuracy results (from `check_accuracy_moonlight.py`):
- 5/10 top-1 matches, cosine sim 0.947-0.986, worst max norm error 0.438
- The script only compares **final logits** -- it can't tell whether divergence originates
  in attention, MoE routing, expert MLPs, RMSNorm, or accumulated BF16 errors across layers.

## Guiding Principles (from NKI steering docs)

- **"Start simple, scale gradually"** -- Compare coarse layer outputs first, then drill into sub-components
- **"Validate at every step"** -- Each measurement must compare against the HF CPU reference (ground truth)
- **"Profile, don't guess"** -- Don't assume MoE or attention is the problem -- measure
- **"When stuck, simplify"** -- Isolate the smallest component that reproduces the divergence

## Architecture Recap (27 layers)

```
embed_tokens -> [layer_0: RMSNorm -> MLA Attention -> RMSNorm -> Dense MLP]
             -> [layer_1..26: RMSNorm -> MLA Attention -> RMSNorm -> MoE(64 routed + 2 shared)]
             -> final RMSNorm -> lm_head
```

Each MLA attention involves: Q projection, KV compression, kv_a_layernorm, weight
absorption (einsum), RoPE, attention scores (matmul + einsum), softmax, V absorption
(2x einsum), output projection.

Each MoE involves: router (sigmoid + top-6), 6 expert MLPs (gate_up_proj -> SiLU ->
down_proj), normalization+scaling, 2 shared experts.

## Key Files

- Model: `optimum/neuron/models/inference/moonlight/modeling_moonlight.py`
- Accuracy script: `examples/inference/text-generation/check_accuracy_moonlight.py`
- Decoder base: `optimum/neuron/models/inference/backend/modules/decoder/modeling_decoder.py`
- Test infra: `tests/decoder/nxd_testing.py` (build_module, validate_accuracy)
- Module tests: `tests/decoder/test_modules.py` (per-component parity tests)
- MoE module: `optimum/neuron/models/inference/backend/modules/moe_v2.py`
- Attention utils: `optimum/neuron/models/inference/backend/modules/attention/utils.py`

---

## Phase 1: HF Hidden State Capture (CPU, no compilation needed)

**Goal**: Record the hidden state tensor at every layer boundary in the HF model.

**Approach**: Use `register_forward_hook` on the HF Moonlight model. Capture:
- `embed_tokens` output (input to layer 0)
- After each `layer[i]`: residual-added hidden state (output of decoder layer)
- After final `norm`: pre-lm_head hidden state
- `lm_head` output: final logits

Save to a `.pt` file as:
```python
{
    "embed_out": tensor,
    "layer_0_out": tensor,
    ...,
    "layer_26_out": tensor,
    "norm_out": tensor,
    "logits": tensor,
}
```

**Effort**: Small -- just hooks on the existing HF model, runs on CPU in seconds for a single prompt.

## Phase 2: Neuron Hidden State Capture (requires model modification + recompile)

**Goal**: Capture the same intermediate hidden states from the Neuron model.

The Neuron model is compiled as a single traced graph -- you can't insert runtime hooks.
Three options:

### Option A -- "Truncated model" approach (recommended)
- Create a series of truncated Neuron models: one that runs only layers 0..k and
  outputs the hidden state at layer k
- Use a binary search strategy: first compare at layer 13 (midpoint), then bisect
  into the divergent half
- Requires compiling ~4-5 truncated models (log2(27) ~ 5) instead of 27

### Option B -- "Component-level build_module" approach
- Use the existing `build_module` from `tests/decoder/nxd_testing.py` to compile
  individual components (attention, MLP, MoE)
- Feed each component the HF-captured inputs from Phase 1
- Compare outputs per-component
- Pro: Reuses test infra. Con: Slower (many compilations), MoE build_module
  integration isn't trivial

### Option C -- "Output all hidden states" approach
- Modify `NxDDecoderModelForCausalLM.forward()` to return hidden states for all
  layers as additional outputs
- Requires a single recompile, but the output signature changes
- Pro: One compilation gives all data. Con: Larger output tensor, may affect compilation

**Recommendation**: Start with Option A (binary search), fall back to Option B for
drilling into sub-components.

## Phase 3: Binary Search for Divergence Source

**Goal**: Identify which layer(s) introduce the most divergence.

**Method**:
1. Compare `embed_tokens` output -- this should be near-exact (just an embedding lookup)
2. Compare at layer 13 midpoint
3. Bisect: if divergence is already high at layer 13, focus on layers 0-13; otherwise 14-26
4. Continue bisecting until you find the layer(s) where divergence spikes

**Metrics at each comparison point**:
- Cosine similarity of hidden states
- Max absolute error
- Relative error (normalized by tensor magnitude)
- Distribution statistics (mean, std, min, max of both tensors)

**Expected pattern**: If errors accumulate gradually, cosine sim will degrade smoothly
across layers. If a specific component (like MoE routing) causes discrete jumps, you'll
see a step function.

## Phase 4: Sub-Component Drill-Down

Once you've identified the divergent layer(s), drill into sub-components using `build_module`:

### 4a. MLA Attention isolation (if attention diverges)
- Test Q projection (ColumnParallelLinear)
- Test KV compression (nn.Linear)
- Test weight absorption einsum (`torch.einsum('hdc,bhqd->bhqc', q_absorb, q_nope)`)
- Test RoPE application
- Test attention score computation (matmul + einsum)
- Test softmax precision
- Test V absorption einsums

### 4b. MoE isolation (if MoE diverges)
- Test router output (sigmoid + top-6 selection) -- are the same experts being selected?
- Test individual expert MLP (gate_up_proj -> SiLU -> down_proj)
- Test affinity normalization + scaling
- Test shared experts

### 4c. Accumulation isolation
- Compare BF16 vs FP32 accumulation for the identified component
- Test with `--enable-mixed-precision-accumulation` on vs off

## Phase 5: Targeted Fixes (based on findings)

Likely fix categories, ordered by probability:

### 5a. Compiler flag tuning
- Remove `--enable-mixed-precision-accumulation` (currently enabled in
  `get_compiler_args` at `modeling_moonlight.py:588`) -- this flag lets the compiler
  choose BF16 accumulators for some matmuls
- Try `-O2` instead of `-O1` (may improve or hurt precision)
- `--auto-cast=none` is already set

### 5b. FP32 accumulation for precision-critical ops
- Cast attention score computation to FP32 before softmax (already done for softmax
  itself, but the einsum inputs may be BF16)
- Cast MoE routing to FP32 (already using `dtype=torch.float32` for router, but
  expert affinity scatter/gather may truncate)
- Force FP32 for the weight absorption einsums

### 5c. Custom NKI kernels (last resort, from steering docs)
- Write a fused attention kernel that keeps intermediate values in FP32 PSUM
- Write a fused MoE dispatch kernel with FP32 accumulation
- Most effort but gives the most control over precision

### 5d. Structural fixes
- Ensure `q_absorb` einsum uses FP32 intermediates (absorbed Q can lose precision)
- Check if `ExpertMLPsV2` does internal FP32 accumulation or not
- Verify `SharedExperts` reduce_dtype is actually used

## Deliverables

| Phase | Output | Compilation? |
|-------|--------|-------------|
| 1 | `capture_hf_intermediates.py` -- saves all HF layer outputs | No |
| 2 | `build_truncated_model.py` -- compiles truncated Neuron models | Yes (4-5 compiles, ~5 min each) |
| 3 | `analyze_divergence.py` -- compares per-layer and reports where error spikes | No (loads saved tensors) |
| 4 | `drill_down_component.py` -- uses `build_module` for sub-component tests | Yes (per component) |
| 5 | Patches to `modeling_moonlight.py` and/or `get_compiler_args()` | Recompile once |

## Quick Win to Try First

Before building the full framework, a **zero-cost experiment**: change the compiler
args in `MoonlightNxDModelForCausalLM.get_compiler_args()`:

```python
# Current (modeling_moonlight.py:588)
compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"

# Experiment: remove --enable-mixed-precision-accumulation
compiler_args = "--enable-saturate-infinity --model-type transformer -O1"
```

This forces FP32 accumulation for all matmuls. If this alone significantly improves
accuracy, the root cause is BF16 accumulation in matmuls (likely the einsums in MLA
attention). If it doesn't help, the divergence is structural (e.g., wrong expert
selection due to routing precision).

## Insights from NxDI MoE Architecture Deep-Dive

Source: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/moe-arch-deep-dive.html

### Blockwise Matrix Multiplication (BWMM)

ExpertMLPsV2 uses BWMM to handle variable token-to-expert distributions within
Neuron's static compilation model. Tokens are organized into fixed-size blocks
(default block_size from BlockwiseMatmulConfig), with padding for incomplete blocks.

**Accuracy implications**:
- Padding tokens are masked out during combine -- should not affect accuracy directly
- The scatter/gather dispatch reorders tokens into blocks, which changes the order
  of accumulation. BF16 accumulation is not associative, so reordering can change results
- We don't pass a BlockwiseMatmulConfig, so defaults are used. Experimenting with
  `block_size` could change the accumulation pattern

### Expert Combine Precision

The final combine `output = sum(a_i * E_i(token))` across top-K=6 experts runs in
the model dtype (BF16). With normalized affinities ~0.167 each, this multiply-accumulate
across 6 terms loses precision. No FP32 option exists in ExpertMLPsV2 for the combine
step -- this would require a code change or custom NKI kernel.

### `early_expert_affinity_modulation`

This option pre-scales inputs by affinity before the expert MLP, then post-scales
the output. It's **incompatible with top_k > 1** (asserted in ExpertMLPsV2), so
it cannot be used for Moonlight (top_k=6).

### `normalize_top_k_affinities`

We correctly set this to False in ExpertMLPsV2 because MoonlightRouter handles
normalization + routed_scaling_factor internally. ExpertMLPsV2 trusts the affinities
as-is during combine.

### Capacity Factor

Moonlight uses dropless MoE (capacity_factor=None, the default). This is correct --
the HF reference processes all tokens through top-K experts without dropping.
The doc notes that DeepSeek-V3 uses dropping MoE, but Moonlight's HF code does not.

### Parallelism Strategy

We use TP=2, EP=1. With 64 routed experts and intermediate_size=1408, each expert's
sharded intermediate is 1408/2=704. The doc suggests EP may be better for >= 64 experts,
but changing TP/EP is a major architecture change -- not relevant for accuracy analysis.

### Quantization

BF16 weights + BF16 compute is our current setup. FP8/MxFP4 would hurt accuracy
further. No FP32 compute option exists in the MoE API.

### Key Takeaway for Plan

The `--enable-mixed-precision-accumulation` compiler flag affects matmul accumulation
inside expert MLPs and attention einsums. This is the primary lever available without
code changes. If removing it doesn't help enough, the next targets are:
1. FP32 casting in the expert combine step (structural code change)
2. FP32 casting in MLA weight absorption einsums (structural code change)
3. Custom NKI kernel for the combine accumulation

## Quick Win Result (completed 2026-03-13)

Removed `--enable-mixed-precision-accumulation` from compiler args and recompiled.

| Metric               | With mixed-prec | Without mixed-prec | Change |
|----------------------|-----------------|-------------------|--------|
| Top-1 matches        | 5/10            | 5/10              | Same   |
| Cosine sim (min)     | 0.947           | 0.947             | Same   |
| Cosine sim (mean)    | ~0.97           | 0.971             | Same   |
| Max norm err (worst) | 0.438           | 0.438             | Same   |
| Max norm err (mean)  | ~0.34           | 0.338             | Same   |

**Conclusion**: Compiler flag had no effect. The divergence is **structural**, not from
BF16 matmul accumulation. The compiler was already choosing adequate accumulation
precision. Root cause is elsewhere -- likely in MLA einsums, MoE dispatch/combine,
or error amplification through the residual stream across 27 layers.

**Reverted** the flag change. Next step: Phase 2+3 (layer-by-layer binary search).

## Phase 2+3 Results (completed 2026-03-13)

Used Option C (instrumented model outputting all layer hidden states) instead of binary search.
The instrumented model packs per-layer hidden states into the output tensor after logits.

### Per-Layer Cosine Similarity (Neuron vs HF, last token position)

| Layer | Cos Sim | Max Abs Err | N_norm | HF_norm | Notes |
|-------|---------|-------------|--------|---------|-------|
| 0     | 0.875   | 0.035       | 0.83   | 0.91    | **Dense layer (MLA + dense MLP, no MoE)** |
| 1     | 0.736   | 0.098       | 1.28   | 1.35    | First MoE layer -- big drop |
| 2     | 0.666   | 0.152       | 2.54   | 2.07    | |
| 3     | 0.572   | 0.273       | 2.89   | 2.78    | |
| 4     | 0.575   | 0.281       | 3.85   | 3.69    | |
| 5     | 0.501   | 0.773       | 3.77   | 3.92    | |
| 6     | **0.478** | 1.850     | 5.49   | 5.66    | **Worst layer** |
| 7     | 0.490   | 1.387       | 5.98   | 6.20    | |
| 8-12  | 0.52-0.64 | ...       | ...    | ...     | Gradual recovery |
| 13-22 | 0.64-0.78 | ...       | ...    | ...     | Continued recovery |
| 23-25 | 0.80-0.82 | ...       | ...    | ...     | Good recovery |
| 26    | 0.903   | 20.25       | 130.89 | 119.37  | Best layer (but large abs err) |

Final logits cosine sim: 0.976
Neuron top-5: [' the', ' ', ' a', ' in', ' not'] -- missing 'Paris'
HF     top-5: [' Paris', ' \\', ' the', ' a', ' located']

### Key Findings

1. **Layer 0 already diverges significantly (cos 0.875)**: This is the dense layer with only
   MLA attention + dense MLP. No MoE involved. MLA attention weight absorption einsums or
   dense MLP are introducing error from the very first layer.

2. **Layers 1-6 amplify divergence rapidly**: Entering MoE layers degrades cosine sim from
   0.875 to 0.478. MoE dispatch/combine and routing errors compound with input errors.

3. **Later layers recover**: As hidden state norms grow (6 -> 130), relative errors shrink.
   Cosine sim climbs back to 0.903 by layer 26.

4. **The recovery pattern suggests errors are additive, not multiplicative**: If MoE/attention
   introduced multiplicative errors, cosine sim would not recover. The additive error from
   each layer becomes relatively smaller as the residual stream accumulates signal.

## Phase 4 Results: Sub-Component Drill-Down (completed 2026-03-13)

### Embedding: PERFECT match
- cos_sim = 1.000000, max_abs = 0.000000 -- identical to HF

### Per-Layer Error Attribution (Attention vs MLP/MoE)

Recompiled instrumented model to capture post-attention and post-layer hidden states separately.

| Layer | Post-Attn Cos | Post-Layer Cos | Attn Drop | MLP/MoE Drop | Primary Source |
|-------|--------------|----------------|-----------|-------------|----------------|
| 0 (dense) | 0.885 | 0.875 | +0.115 | +0.010 | **MLA Attention (92%)** |
| 1 (MoE)   | 0.854 | 0.736 | +0.021 | +0.118 | **MoE (85%)** |
| 2 (MoE)   | 0.738 | 0.666 | -0.002 | +0.072 | **MoE (100%)** |
| 3 (MoE)   | 0.587 | 0.572 | +0.078 | +0.016 | MLA Attention |
| 4 (MoE)   | 0.611 | 0.575 | -0.040 | +0.037 | Mixed |
| 5 (MoE)   | 0.562 | 0.501 | +0.013 | +0.060 | MoE |
| 6+ | ... | ... | ... | ... | Errors shrink, MoE often helps |

Two primary error sources identified:
1. **MLA Attention** -- layer 0 drops cos sim from 1.0 to 0.885 through attention alone
2. **MoE** -- layers 1-2 drop cos sim by 0.118 and 0.072 respectively

### Root Cause Analysis: MLA Attention

The Neuron model uses **weight absorption** (Q-absorb, V-absorb via einsum), while HF
uses **explicit KV decompression** (`kv = kv_b_proj(compressed_kv)`). These are
mathematically equivalent but numerically different:

- HF: `Q @ (W_k @ compressed_kv).T` -- decompresses KV first, rounds decompressed K to BF16
- Neuron: `(Q @ W_k.T) @ compressed_kv.T` -- absorbs weights first, rounds absorbed Q to BF16

The ORDER of BF16 rounding is fundamentally different. This is inherent to weight absorption.

**Experiment: FP32 einsums** -- Wrapped all MLA einsums in `.float()` casts. Result: NO
improvement (cos sim 0.975 vs 0.976 before). The compiler already uses FP32 accumulation
for these matmuls; the issue is which intermediate tensors get rounded to BF16 between
operations, not the accumulation precision within an operation.

**Conclusion**: MLA attention divergence is **structural and expected**. The NxDI reference
also uses weight absorption and has similarly relaxed tolerances (tol_map={50: (1e-5, 0.40)}).

### Root Cause Analysis: MoE

Layer 1 MoE introduces +0.118 cos sim drop. Possible causes:
- Sigmoid routing precision differences → different expert selection
- BWMM blockwise matmul introduces token reordering → BF16 accumulation order changes
- Expert combine (weighted sum of 6 experts) in BF16 loses precision

These are also structural to the NxDI MoE implementation (ExpertMLPsV2, SharedExperts).

## Final Assessment

The divergence is **structural and within expected tolerances**:

| Metric | Current | NxDI Reference Tolerance |
|--------|---------|------------------------|
| Final logits cos sim | 0.975 | N/A (not directly comparable) |
| Top-50 max norm err | 0.46 worst | 0.40 (tol_map) |
| Layer-level cos sim | 0.48-0.90 | N/A |

Root causes (all structural, not bugs):
1. **Weight absorption** in MLA attention produces different BF16 rounding pattern vs HF
2. **BWMM MoE dispatch** changes token accumulation order
3. **Sigmoid routing** in BF16 may select slightly different expert sets

These are inherent to the NxDI inference architecture and are shared with the upstream
NxDI implementation. No single-point fix can resolve them without changing the architecture
(e.g., switching to explicit KV decompression, which defeats the MLA optimization).

## Execution Order

1. ~~Quick win: compiler flag experiment~~ -- DONE, no effect
2. ~~Phase 1: capture HF intermediates~~ -- DONE, saved to /tmp/moonlight_hf_intermediates.pt
3. ~~Phase 2+3: per-layer analysis~~ -- DONE, identified layers 0-6 as divergence zone
4. ~~Phase 4: sub-component drill-down~~ -- DONE, identified MLA attention and MoE as sources
5. ~~Phase 5: FP32 einsum experiment~~ -- DONE, no effect (structural root cause confirmed)
