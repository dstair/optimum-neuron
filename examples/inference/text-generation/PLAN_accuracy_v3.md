# Moonlight Accuracy Improvement Plan v3 — Layer 0 Attention Drill-Down

**Status**: COMPLETED. RoPE interleave bug found and fixed.

## Background

### Pre-Fix Accuracy Baseline
- **2/5 top-1 matches** (5-token test), cosine sim min=0.945, mean=0.972
- First token: Neuron predicts `' the'`, HF predicts `' Paris'`
- Passes at tolerance 0.50 (matching NXDI reference's relaxed thresholds)

### Pre-Fix Per-Layer Divergence (from Phase 4 analysis)
| Layer | Component | Cosine Sim | Notes |
|-------|-----------|-----------|-------|
| Embedding | — | 1.000 | Perfect match |
| 0 | Post-attention | 0.885 | **Dominant error source** |
| 0 | Post-dense-MLP | ~0.85 | Small additional drop |
| 1 | Post-MoE | 0.736 | MoE amplifies error |
| 2 | Post-MoE | 0.664 | Continues degrading |
| 6 | Post-MoE | 0.478 | Worst layer |
| 26 | Final | 0.903 | Recovers slightly |

### Previously Applied Fixes
1. **MoE routing `e_score_correction_bias`** — Fixed in `MoonlightRouter`. See PLAN_accuracy_v2.md.
2. **MoE routing normalize + scale** — `routed_scaling_factor=2.446`. See PLAN_accuracy_v2.md.

### Previously Ruled Out
**MLA computation order does NOT affect accuracy.** Three variants tested; absorption kept.

---

## Investigation: CPU Comparison Script

### Approach
Built `compare_layer0_attention.py` — a CPU-only script that:
1. Loads HF model weights for layer 0
2. Replicates both HF and Neuron attention forward passes manually using the same weights
3. Feeds identical BF16 input (embedding output for "The capital of France is")
4. Compares intermediates after EACH operation

### Results: RoPE Interleave Bug Found

**Root cause**: HF Moonlight's `apply_rotary_pos_emb` (`modeling_deepseek.py:364-368`)
transposes q_pe and k_pe from **interleaved** layout `[r0, i0, r1, i1, ...]` to **split**
layout `[r0, r1, ..., i0, i1, ...]` before applying `rotate_half`:

```python
# HF apply_rotary_pos_emb (modeling_deepseek.py:364-368)
b, h, s, d = q.shape
q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)  # interleaved → split
b, h, s, d = k.shape
k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```

The Neuron shared `apply_rotary_pos_emb` (`utils.py:152-159`) does **not** do this transpose,
meaning `_rotate_half` pairs elements at positions `(j, j+d/2)` instead of the correct
complex real/imaginary pairs `(r_j, i_j)`.

### CPU Comparison Results

```
--- Pre-RoPE (should be identical) ---
  Q projection                   cos_sim=1.000  max_diff=0.00e+00  [MATCH]
  Q nope                         cos_sim=1.000  max_diff=0.00e+00  [MATCH]
  Q pe (pre-RoPE)                cos_sim=1.000  max_diff=0.00e+00  [MATCH]
  compressed_kv                  cos_sim=1.000  max_diff=0.00e+00  [MATCH]
  k_pe (pre-RoPE)                cos_sim=1.000  max_diff=0.00e+00  [MATCH]

--- RoPE (THE BUG) ---
  q_pe after RoPE                cos_sim=-0.047  max_diff=3.35e-01  [DIFF]
  k_pe after RoPE                cos_sim=0.004   max_diff=9.53e-01  [DIFF]

--- With fix applied ---
  q_pe after RoPE (fixed)        cos_sim=1.000  max_diff=0.00e+00  [MATCH]
  k_pe after RoPE (fixed)        cos_sim=1.000  max_diff=0.00e+00  [MATCH]
```

All operations before RoPE match perfectly. RoPE outputs were essentially **uncorrelated**
(cos_sim -0.047). With the interleave transpose added, they match perfectly.

Note: on CPU, the final attention output was still cos_sim ~0.9999 even with broken RoPE
because the content-based nope part (512 dims) dominates the rope part (64 dims) for a
5-token sequence. The effect is much larger on-device where BF16 amplifies the error
through 27 layers.

---

## Fix Applied

**File**: `modeling_moonlight.py:302-307`

Added interleave-to-split transpose before calling `apply_rotary_pos_emb`:

```python
# Convert rope dims from interleaved [r0,i0,r1,i1,...] to split [r0,r1,...,i0,i1,...]
b, h, s, d = q_pe.shape
q_pe = q_pe.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
b, h, s, d = k_pe.shape
k_pe = k_pe.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
```

The fix is in `NeuronMoonlightAttention.forward` only (model-specific), not in the shared
`apply_rotary_pos_emb` which is correct for other models (Llama, etc.) that store weights
in split layout.

---

## Post-Fix Results

### End-to-End Accuracy (check_accuracy_moonlight.py, 5-token decode)

| Metric | Before Fix | After Fix | Change |
|--------|-----------|-----------|--------|
| Top-1 matches | 2/5 | **3/5** | +1 |
| Cosine sim min | 0.945 | **0.994** | +0.049 |
| Cosine sim mean | 0.972 | **0.997** | +0.025 |
| Max norm error | ~0.50 | **0.134** | 3.7x better |
| First token | `' the'` (wrong) | `' Paris'` (correct) | Fixed |

Remaining 2/5 mismatches are BF16 hardware tiebreakers (logit gaps < 0.5).

### Per-Layer Divergence (analyze_divergence.py, context encoding)

| Layer | Post-Attn (before) | Post-Attn (after) | Post-Layer (before) | Post-Layer (after) |
|-------|-------------------|-------------------|--------------------|--------------------|
| Embed | 1.000 | 1.000 | — | — |
| 0 (dense) | **0.885** | **0.99998** | ~0.85 | 0.99997 |
| 1 (MoE) | — | 0.99995 | 0.736 | 0.99994 |
| 2 (MoE) | — | 0.99994 | 0.664 | 0.99994 |
| 6 (MoE) | — | 0.99948 | **0.478** | 0.99947 |
| 11 (MoE) | — | 0.99938 | — | **0.99762** (worst) |
| 12 (MoE) | — | **0.99769** (worst attn) | — | 0.99829 |
| 26 (MoE) | — | 0.99884 | 0.903 | 0.99903 |

**Full layer table (after fix):**
```
Layer  PostAttn   PostLayer
 0 D   0.999981   0.999965
 1     0.999951   0.999935
 2     0.999938   0.999937
 3     0.999897   0.999896
 4     0.999885   0.999889
 5     0.999837   0.999817
 6     0.999478   0.999471
 7     0.999441   0.999505
 8     0.999394   0.999533
 9     0.999432   0.999429
10     0.999340   0.999478
11     0.999377   0.997615
12     0.997685   0.998286
13     0.998129   0.998395
14     0.998483   0.998864
15     0.998823   0.998310
16     0.998482   0.998847
17     0.998872   0.998854
18     0.998897   0.999069
19     0.999104   0.999034
20     0.999091   0.999268
21     0.999287   0.999316
22     0.999323   0.998448
23     0.998517   0.998729
24     0.998812   0.998846
25     0.998872   0.998790
26     0.998842   0.999025

Summary:
  Post-attn cos sim:  min=0.9977 (layer 12), mean=0.9992
  Post-layer cos sim: min=0.9976 (layer 11), mean=0.9991
  Final logits cos sim: 0.9998
  Neuron top-5: [Paris, \, the, a, located]  (identical to HF)
```

### Key Observations
- **Layer 0 post-attention: 0.885 → 0.99998** — RoPE fix eliminated the dominant error source
- **No layer drops below 0.997** — previously degraded to 0.478
- **Error no longer compounds catastrophically** through MoE layers
- **Neuron and HF top-5 are identical** in order
- Remaining ~0.001-0.003 per-layer gap is smooth BF16 hardware accumulation — no spikes

---

## All Accuracy Fixes Summary

| # | Fix | Impact | Script |
|---|-----|--------|--------|
| 1 | MoE `e_score_correction_bias` | 60-100% wrong experts → 0% | `compare_moe_routing.py` |
| 2 | MoE normalize + `routed_scaling_factor` | Correct expert weights | — |
| 3 | **RoPE interleave transpose** | **Layer 0 cos 0.885→0.99998** | `compare_layer0_attention.py` |

Combined result: cos_sim min 0.945 → 0.994, top-1 2/5 → 3/5, first token correct.

## Ruled Out
- MLA computation order (absorption vs explicit): no effect on accuracy
- NeuronRMSNorm: matches HF exactly on CPU
- Q/KV/O projections: match HF exactly on CPU
- Softmax: matches HF exactly on CPU

## Conclusion
All code-level accuracy issues are resolved. Remaining divergence (cos_sim ~0.997-0.999
per layer) is purely from Neuron hardware BF16 matmul accumulation — not actionable
without compiler-level changes.

---

## Previous Plans
- `PLAN_accuracy_analysis.md` — Phases 1-4: per-layer divergence analysis
- `PLAN_accuracy_v2.md` — Routing fix (done), explicit KV decompression (done, ruled out)
