# Moonlight Accuracy Improvement Plan v2

## Context

The previous analysis (PLAN_accuracy_analysis.md) identified two structural error sources
but concluded they were inherent to the NXDI architecture. This plan revisits that
conclusion — the NXDI "reference architecture" was itself incomplete and had accuracy
issues. We should not treat its tolerances as ground truth.

### Current State
- 5/10 top-1 matches, cosine sim 0.947-0.986, worst error 0.438
- Top-5 predictions: `[' the', ' ', ' a', ' in', ' not']` — missing `'Paris'`
- HF Top-5: `[' Paris', ' \\', ' the', ' a', ' located']`

### Two Error Sources (from Phase 4)

| Source | Where | Magnitude | Mechanism |
|--------|-------|-----------|-----------|
| MLA Attention | Layer 0 | cos 1.0 → 0.885 | Weight absorption vs explicit KV decompression |
| MoE | Layers 1-2 | cos drops 0.118, 0.072 | BWMM dispatch + combine + routing precision |

---

## Workstream 1: Explicit KV Decompression in MLA

### Problem

Neuron MLA uses **weight absorption** (fuse `kv_b_proj` into Q and V via einsum):
```
q_nope = einsum('hdc,bhqd->bhqc', q_absorb, q_nope)   # absorbed Q
scores = matmul(q_pe, k_pe.T) + einsum('bhqc,blc->bhql', q_nope, compressed_kv)
x = einsum('bhql,blc->bhqc', softmax_out, compressed_kv)
attn_out = einsum('bhqc,hdc->bhqd', x, out_absorb)     # absorbed V
```

HF MLA uses **explicit KV decompression** (expand `kv_b_proj` into full K and V):
```
kv = kv_b_proj(kv_a_layernorm(compressed_kv))  # (B,S,H,nope_dim+v_dim)
k_nope, v = split(kv)
q_full = cat(q_nope, q_pe)
k_full = cat(k_nope, k_pe)
scores = matmul(q_full, k_full.T)
attn_out = matmul(softmax_out, v)
```

These are **mathematically equivalent** but produce different BF16 rounding:
- Absorption: rounds the *absorbed Q* (shape `num_heads × q_len × kv_lora_rank`) to BF16
- Explicit: rounds the *decompressed K/V* (shape `num_heads × seq_len × nope_dim/v_dim`) to BF16

Weight absorption is an optimization (smaller KV cache for inference). But since Moonlight
already stores `compressed_kv` in the KV cache (not expanded K/V), the absorption is
applied *within the attention computation*, not for cache savings. Switching to explicit
decompression does NOT change the KV cache format.

### Proposed Change

Replace the four einsums in `NeuronMoonlightAttention.forward()` with explicit
`kv_b_proj` decompression, matching the HF computation order exactly.

### What Changes in `modeling_moonlight.py`

**Remove** (lines 269-274, 292, 302-303, 312-313, 343-344, 347-348):
```python
# Weight absorption extraction
wkv_b = self.kv_b_proj.weight
wkv_b = wkv_b.view(self.num_heads, -1, self.kv_lora_rank)
q_absorb = wkv_b[:, :self.qk_nope_head_dim]
out_absorb = wkv_b[:, self.qk_nope_head_dim:, :]

# Q-nope weight absorption
q_nope = torch.einsum('hdc,bhqd->bhqc', q_absorb, q_nope)

# Score computation using compressed KV
active_scores = (
    torch.matmul(q_pe, k_pe.T)
    + torch.einsum('bhqc,blc->bhql', q_nope, compressed_kv)
)

# V absorption
x = torch.einsum("bhql,blc->bhqc", active_scores, compressed_kv)
attn_output = torch.einsum("bhqc,hdc->bhqd", x, out_absorb)
```

**Replace with**:
```python
# Explicit KV decompression (matches HF)
kv = self.kv_b_proj(compressed_kv)  # (B, S, num_heads * (nope_dim + v_dim))
kv = kv.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
kv = kv.transpose(1, 2)
k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

# Reconstruct full Q and K
query_states = torch.cat([q_nope, q_pe], dim=-1)
key_states = torch.cat([k_nope, k_pe.expand(-1, self.num_heads, -1, -1)], dim=-1)

# Standard attention
active_scores = torch.matmul(query_states, key_states.transpose(2, 3))
active_scores *= self.softmax_scale

# ... softmax ...

attn_output = torch.matmul(softmax_out, value_states)
```

### Impact Assessment

| Aspect | Before (absorption) | After (explicit) |
|--------|-------------------|-----------------|
| KV cache format | `(k_pe, compressed_kv)` — unchanged | Same — still stores compressed_kv |
| KV cache size | `1 head × 576 dim` | Same |
| Attention compute | 4 einsums on compressed_kv | 1 linear proj + 2 standard matmuls |
| Intermediate tensors | `q_nope`: (B,H,S,kv_lora) | `kv`: (B,H,S,nope+v_dim) |
| Memory overhead | Lower (no expanded K/V) | Higher (expanded K/V per-head) |
| Decode path | Also uses absorption | Must also switch to explicit |

### Memory Concern

Expanded K/V is `num_heads × seq_len × (nope_dim + v_dim)` = `64 × S × (128+128)` per
TP rank (32 heads per rank). For context encoding with S=4096:
- Expanded: 32 × 4096 × 256 × 2 bytes = 64 MB per layer — acceptable.
- Compressed: 1 × 4096 × 512 × 2 bytes = 4 MB — much smaller.

For decode (S=1), both are trivial.

### Effort Estimate

- Modify `forward()` prefill path: ~30 lines changed
- Modify `forward()` decode path: ~30 lines changed (prior scores + active scores)
- No state dict changes (same weights, same kv_b_proj)
- No KV cache format changes
- Requires recompile (~5 min)
- **Risk**: Compute overhead from explicit decompression may slow context encoding.
  But this is an accuracy investigation, not a production optimization.

### Validation

1. Run `check_accuracy_moonlight.py` — compare layer 0 post-attention cos sim
2. Expected: layer 0 cos sim should improve from 0.885 toward ~0.99+
3. If layer 0 improves, rerun full 10-token test for end-to-end comparison

---

## Workstream 2: MoE Layer 1 Cos Sim Drop Drill-Down

### Problem

Layer 1 MoE introduces +0.118 cos sim drop (0.854 → 0.736). Three candidate causes
were listed but never isolated. This workstream prioritizes and tests each.

### Candidate Causes (prioritized)

#### 2A. Sigmoid routing precision → different expert selection (HIGH PRIORITY)

**Why first**: If Neuron selects different experts than HF, the MoE output will diverge
regardless of expert computation precision. This is a discrete error (wrong experts) vs
continuous error (BF16 rounding).

**Mechanism**: The router runs sigmoid on logits, then `torch.topk` selects top-6.
If two experts have nearly equal sigmoid scores, BF16 rounding of the sigmoid output
could swap their ranking, selecting a completely different expert.

**Test plan**:
1. Capture HF router state at layer 1: sigmoid scores (FP32), selected expert indices
2. Capture Neuron router state at layer 1: sigmoid scores (BF16→FP32), selected expert indices
3. Compare:
   - Are the same 6 experts selected? If not, which differ?
   - What's the gap between the 6th and 7th experts? (small gap = fragile selection)
   - Is the sigmoid output in BF16 close to the FP32 reference?

**How to capture**: The `MoonlightRouter.forward()` is already a Python method we control.
Add temporary logging to save `router_logits`, `expert_affinities`, `expert_index` for
the first forward pass. For HF, hook into `DeepseekV3MoE.gate`.

**If confirmed**: Two possible fixes:
- Cast router to FP32 end-to-end (sigmoid + topk). Already using `dtype=torch.float32`
  for the router, but verify the *input* to `get_router_logits` is also FP32.
- Increase routing margin: not applicable (model weights are fixed).

#### 2B. BWMM blockwise accumulation order (MEDIUM PRIORITY)

**Why second**: Even with correct expert selection, BWMM reorders tokens into blocks.
BF16 accumulation is not associative, so the output of `gate_up_proj → SiLU → down_proj`
may differ from HF's sequential per-token computation.

**Test plan**:
1. Extract the input to `ExpertMLPsV2` (i.e., the hidden states entering MoE)
2. For a single expert, compare:
   - HF: `down_proj(silu(gate_proj(x)) * up_proj(x))` applied per-token
   - Neuron: same computation but via BWMM with block dispatch
3. If single-expert outputs match, the issue is in combine. If they don't, it's in BWMM.

**Difficulty**: Isolating a single expert's I/O from the compiled BWMM graph is hard.
Alternative: build a minimal `ExpertMLPsV2` with 1 expert on CPU and compare against
a plain `nn.Linear` chain. This tests whether the BWMM dispatch itself introduces error
independent of the expert computation.

**If confirmed**: Limited options — BWMM is a fundamental NXDI mechanism. Could try:
- Different `block_size` in `BlockwiseMatmulConfig`
- `torch_blockwise_matmul_inference` fallback (non-NKI path)

#### 2C. Expert combine precision (LOWER PRIORITY)

**Why third**: The combine step `output = sum(affinity_i * expert_i(x))` over 6 experts
in BF16 loses precision. But with normalized affinities summing to ~2.446 (after scaling),
individual terms are ~0.4 each — the accumulation of 6 BF16 terms should be okay.

**Test plan**:
1. If 2A and 2B don't fully explain the drop, capture per-expert outputs and affinities
2. Compute the combine in FP32 and compare to BF16 combine
3. Quantify the combine-only error

**If confirmed**: Would require modifying `ExpertMLPsV2` or the BWMM NKI kernel to
accumulate in FP32 during combine — substantial effort.

### Implementation Approach

For capturing intermediate MoE state, modify `NeuronMoonlightDecoderLayer.forward()`
to save router outputs when a debug flag is set:

```python
if self._debug_capture and not self._is_dense:
    # Save pre-MoE input, router outputs, and MoE output
    self._debug_state = {
        'moe_input': hidden_states.detach().cpu(),
    }
```

For HF capture, use `register_forward_hook` on `model.layers[1].mlp.gate` and
`model.layers[1].mlp`.

**Problem**: The Neuron model is a compiled graph — we can't insert runtime hooks or
save intermediate tensors mid-graph. Two approaches:

**Approach A — Router-only capture (works without recompile)**:
The MoonlightRouter runs *before* the compiled BWMM. If we modify `MoonlightRouter.forward()`
to save its outputs to a global dict before returning, XLA should still trace through it
(the save is a side-effect, not a graph output). **But**: XLA tracing typically ignores
side effects. Need to verify this works.

**Approach B — Instrumented model (requires recompile)**:
Like the Phase 2+3 approach, modify the model to output router state as additional outputs
packed into the output tensor. This is reliable but expensive (recompile).

**Approach C — CPU simulation (no compile needed)**:
Run just the router computation on CPU using the same weights and input. Feed the
Neuron model's layer-1 input (captured from Phase 2+3 instrumented model, or re-captured)
through the HF router and compare. This tests routing divergence without any Neuron
compilation.

**Recommended**: Start with Approach C. We already have captured per-layer hidden states.
Feed the Neuron layer-1 input through both the HF router (CPU) and Neuron router (CPU,
using same weights) and compare expert selections.

---

## Execution Log

### Phase 1: MoE Routing Comparison (Workstream 2A) -- DONE

**Result: CONFIRMED BUG -- missing `e_score_correction_bias`**

Used `compare_moe_routing.py` to run HF and Neuron routing on CPU using captured
`layer_X_mlp_in` hidden states from previous analysis. No compilation needed.

**Root cause**: HF `MoEGate` uses `noaux_tc` routing which selects top-K from
`sigmoid(logits) + e_score_correction_bias`. The Neuron `MoonlightRouter` was selecting
top-K from raw `logits` with NO bias. The `e_score_correction_bias` parameter (shape [64],
one per expert) was completely missing from both the router and state dict conversion.

**Impact across all 26 MoE layers** (5-token test):

| Layer | Wrong experts (no bias) | Wrong experts (with bias) |
|-------|------------------------|--------------------------|
| 1     | 3/5 (60%)              | 0/5                      |
| 2     | 2/5 (40%)              | 0/5                      |
| 3     | 4/5 (80%)              | 0/5                      |
| 4-6   | 3-4/5 (60-80%)         | 0/5                      |
| 7-11  | 3-4/5 (60-80%)         | 0/5                      |
| 12    | 5/5 (100%)             | 0/5                      |
| 13-26 | 1-4/5 (20-80%)         | 0/5                      |

**Fix applied**:
1. `MoonlightRouter.__init__()`: Added `e_score_correction_bias` as `nn.Parameter`
2. `MoonlightRouter.forward()`: Selection now uses `sigmoid(logits) + bias` for top-K,
   then gathers weights from original sigmoid scores (no bias), matching HF exactly
3. `convert_moonlight_hf_to_neuron_state_dict()`: Renames
   `gate.e_score_correction_bias` → `router.e_score_correction_bias`
4. Test: Added `test_e_score_correction_bias_renamed` to smoke tests (all 11 pass)

**On-device result**: Recompiled and ran 10-token test. Result: 5/10 top-1, cos sim
min=0.945, worst error=0.466. **No improvement** in end-to-end accuracy.

**Why**: MLA attention divergence in layer 0 (cos 0.885) feeds different hidden states
to every router. Even with correct routing logic, different inputs → different expert
selection at runtime. Verified: at layer 1, only 3/6 experts overlap between HF and
Neuron even with bias fix, because the router INPUTS differ (post-attn cos 0.854).

**Conclusion**: Both fixes needed together. The routing fix is correct and necessary
(without it, even identical inputs produce 60-100% wrong experts), but its benefit is
masked until MLA attention also matches HF. Proceed to Workstream 1.

### Phase 2: Explicit KV Decompression (Workstream 1) -- DONE (FAILED)

**Result: NOT VIABLE on Neuron hardware. Reverted to absorption.**

Implemented explicit KV decompression matching HF computation order exactly.
Two variants tested:

1. **Via ColumnParallelLinear.forward()**: `kv = self.kv_b_proj(compressed_kv)` then
   reshape/split into `k_nope`, `value_states`. Standard `torch.matmul` for scores and
   attention output.

2. **Direct weight access**: `torch.matmul(compressed_kv, self.kv_b_proj.weight.t())`
   bypassing ColumnParallelLinear entirely. Same reshape/split/matmul after.

**CPU verification**: Both variants produce output within 1e-6 of the absorption approach.
The math is identical — weight absorption is just a reordering of the same matmuls.

**On-device result (both variants identical)**:
- 0/10 top-1 matches (vs 5/10 baseline)
- Cosine similarity ~0.07 (vs 0.945-0.986 baseline)
- Model consistently outputs token 3548 (`拥`, Chinese character) for every position
- Device memory: 38.16 GB (vs 36.64 GB baseline — 1.52 GB increase from expanded K/V)

**Probable root cause** (identified via second-opinion analysis): The explicit decompression
concatenates `q_nope` (128) + `q_pe` (64) into a 192-dim inner product for `Q @ K.T`.
192 is NOT a power of 2. Neuron's tensor engine is optimized for dims 64/128/256 and must
tile 192 as 128+64 with padding to 256. The compiler likely fails to zero-mask the padded
dimensions, corrupting the dot product. Absorption avoids this entirely — it uses separate
matmuls at dim 64 (`q_pe @ k_pe.T`) and dim 512 (`q_nope_absorbed @ compressed_kv`),
both powers of 2.

Note: TP reshape was also flagged but does NOT apply — `self.num_heads` is already
the local count (64 // tp_degree = 32).

**Post-revert verification**: Reverted to absorption einsums. Recompiled and tested:
2/5 top-1 matches, cos sim min=0.945, mean=0.972, PASS within tolerance.

### Phase 2b: Padded Explicit KV Decompression (retry) -- DONE (STILL GARBAGE)

**Hypothesis**: 192-dim Q@K inner product triggers Neuron compiler tiling bug.
Zero-padding to 256 should fix it.

**Implementation**: `F.pad(query/key, (0, 64))` after `torch.cat([nope, pe])`, then
standard `torch.matmul` at 256 inner dim. CPL forward for kv_b_proj.

**Result**: 0/5 top-1, cos sim ~0.06, same `拥` garbage. Padding did NOT fix it.
The 192-dim may be part of the problem, but `torch.cat` and/or `F.pad` and/or
`ColumnParallelLinear.forward()` also contribute.

### Phase 2c: Split Decompression (no concat, no pad) -- DONE (WORKS, SAME ACCURACY)

**Approach**: Avoid ALL suspicious operations. Decompose kv_b_proj weight manually
(same as absorption weight access), but use HF's computation order:
- `k_nope = einsum("hdc,blc->bhld", w_k_nope, compressed_kv)` — explicit K per head
- `value_states = einsum("hdc,blc->bhld", w_value, compressed_kv)` — explicit V per head
- `scores = matmul(q_nope, k_nope.T) + matmul(q_pe, k_pe.T)` — split scores (dims 128, 64)
- `attn_out = matmul(softmax_out, value_states)` — standard V matmul (dim 128)

No `torch.cat`, no `F.pad`, no CPL forward, all dims powers of 2.

**Result**: 2/5 top-1, cos sim min=0.945, mean=0.972, PASS. **Identical to absorption.**
Device memory: 36.61 GB (same as baseline).

**Conclusion**: The MLA computation order (absorption vs split decompression vs explicit)
does NOT affect accuracy on Neuron. Both produce cos sim ~0.97. The layer 0 attention
divergence (cos 0.885) originates in shared components: Q proj, kv_a_proj, kv_a_layernorm,
RoPE, softmax, or o_proj — NOT in the score/V computation order.

**Neuron compiler findings**:
- `torch.cat` to non-power-of-2 dim (192) + `torch.matmul` = garbage on Neuron
- `F.pad` to 256 does NOT fix the concat-based approach
- `ColumnParallelLinear.forward()` on 3D input may also contribute to corruption
- Split einsums with power-of-2 dims (128, 64, 512) work correctly
- Direct weight access + einsum works; CPL forward + reshape does not

**Reverted to absorption** — simpler, fewer ops, same accuracy.

## Remaining Execution Order

### Phase 3: MoE Expert Computation (Workstream 2B, only if needed)

1. Build minimal single-expert comparison on CPU
2. Compare BWMM output vs plain matmul for identical inputs
3. Quantify BWMM-induced error

### Phase 4: Combined Assessment

After all phases, measure end-to-end accuracy with all improvements applied.

---

## Success Criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Top-1 matches (10 tokens) | 5/10 | 7/10 | 9/10 |
| Cosine sim (min) | 0.947 | 0.98 | 0.995 |
| Layer 0 post-attn cos | 0.885 | 0.95 | 0.99 |
| Layer 1 post-MoE cos | 0.736 | 0.85 | 0.95 |
| First token prediction | ' the' | 'Paris' | 'Paris' |

## Files to Modify

| File | Change | Workstream |
|------|--------|------------|
| `modeling_moonlight.py` | Explicit KV decompression in attention | 1 |
| `check_accuracy_moonlight.py` | Add router comparison mode | 2A |
| New: `compare_moe_routing.py` | Standalone routing comparison script | 2A |
