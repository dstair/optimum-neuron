#!/usr/bin/env python3
"""Compare layer 0 attention intermediates between HF and Neuron implementations on CPU.

Isolates the RoPE interleaving bug: HF apply_rotary_pos_emb transposes q/k from
interleaved [r0, i0, r1, i1, ...] to split [r0, r1, ..., i0, i1, ...] layout
before rotate_half. Neuron's apply_rotary_pos_emb skips this transpose, pairing
wrong elements.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python compare_layer0_attention.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_PATH = "/home/ubuntu/environment/models/Moonlight-16B-A3B"
PROMPT = "The capital of France is"

# Model dimensions (from config.json)
NUM_HEADS = 16
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
Q_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192
KV_LORA_RANK = 512
SOFTMAX_SCALE = Q_HEAD_DIM ** (-0.5)


# ─── RoPE implementations ───────────────────────────────────────────────────

def rotate_half(x):
    """Standard rotate_half (same in both HF and Neuron)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def hf_apply_rotary_pos_emb(q, k, cos, sin):
    """HF Moonlight RoPE: transpose interleaved→split THEN rotate."""
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def neuron_apply_rotary_pos_emb(q, k, cos, sin):
    """Neuron RoPE: rotate_half directly (no interleave transpose)."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ─── Attention forward implementations ──────────────────────────────────────

def compute_cos_sin(position_ids, dim=64, base=50000.0, dtype=torch.bfloat16):
    """Compute cos/sin tables (same math for both, different indexing)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    # Shape: (batch, dim/2, 1) @ (batch, 1, seq) -> (batch, dim/2, seq) -> (batch, seq, dim/2)
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype).unsqueeze(1)  # (B, 1, S, D)
    sin = emb.sin().to(dtype).unsqueeze(1)
    return cos, sin


def hf_attention_forward(hidden_states, weights, position_ids):
    """HF layer 0 attention forward (explicit KV decompression, no absorption)."""
    bsz, q_len, _ = hidden_states.size()

    # Q projection
    q = F.linear(hidden_states, weights["q_proj"])
    q = q.view(bsz, q_len, NUM_HEADS, Q_HEAD_DIM).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM], dim=-1)

    # KV compression
    compressed_kv = F.linear(hidden_states, weights["kv_a_proj_with_mqa"])
    compressed_kv, k_pe = torch.split(compressed_kv, [KV_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1)

    # KV layernorm (manual RMSNorm, matching HF)
    ln_weight = weights["kv_a_layernorm"]
    x_float = compressed_kv.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    compressed_kv = (ln_weight * (x_float * torch.rsqrt(variance + 1e-6))).to(hidden_states.dtype)

    k_pe = k_pe.view(bsz, q_len, 1, QK_ROPE_HEAD_DIM).transpose(1, 2)

    # RoPE (HF version with interleave transpose)
    cos, sin = compute_cos_sin(position_ids, dim=QK_ROPE_HEAD_DIM, dtype=hidden_states.dtype)
    q_pe_hf, k_pe_hf = hf_apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

    # Explicit KV decompression (HF style)
    kv = F.linear(compressed_kv, weights["kv_b_proj"])
    kv = kv.view(bsz, q_len, NUM_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM).transpose(1, 2)
    k_nope, value_states = torch.split(kv, [QK_NOPE_HEAD_DIM, V_HEAD_DIM], dim=-1)

    # Reassemble full Q and K (k_pe is single-head, broadcast to all heads)
    query_states = torch.cat([q_nope, q_pe_hf], dim=-1)
    key_states = k_pe_hf.new_empty(bsz, NUM_HEADS, q_len, Q_HEAD_DIM)
    key_states[:, :, :, :QK_NOPE_HEAD_DIM] = k_nope
    key_states[:, :, :, QK_NOPE_HEAD_DIM:] = k_pe_hf  # broadcasts (1 head → 16)

    # Attention scores
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * SOFTMAX_SCALE

    # Causal mask (additive, -inf for future positions)
    causal = torch.full((q_len, q_len), float("-inf"), dtype=hidden_states.dtype)
    causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0)
    attn_weights = attn_weights + causal

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    # Output projection
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, NUM_HEADS * V_HEAD_DIM)
    attn_output = F.linear(attn_output, weights["o_proj"])

    return attn_output, {
        "q": q, "q_nope": q_nope, "q_pe": q_pe,
        "compressed_kv": compressed_kv, "k_pe": k_pe,
        "q_pe_rope": q_pe_hf, "k_pe_rope": k_pe_hf,
        "query_states": query_states, "key_states": key_states,
        "attn_weights": attn_weights, "attn_output_pre_proj": attn_output,
    }


def neuron_attention_forward(hidden_states, weights, position_ids, use_fixed_rope=False):
    """Neuron layer 0 attention forward (absorption, Neuron RoPE)."""
    bsz, q_len, _ = hidden_states.size()

    # Q projection (same weights, same computation)
    q = F.linear(hidden_states, weights["q_proj"])
    q = q.view(bsz, q_len, NUM_HEADS, Q_HEAD_DIM).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM], dim=-1)

    # KV compression (same)
    compressed_kv = F.linear(hidden_states, weights["kv_a_proj_with_mqa"])
    compressed_kv, k_pe = torch.split(compressed_kv, [KV_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1)

    # KV layernorm (manual RMSNorm — NeuronRMSNorm calls RmsNorm.apply which
    # may not work on CPU, so use manual impl with same weights)
    ln_weight = weights["kv_a_layernorm"]
    x_float = compressed_kv.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    compressed_kv = (ln_weight * (x_float * torch.rsqrt(variance + 1e-6))).to(hidden_states.dtype)

    k_pe = k_pe.view(bsz, q_len, 1, QK_ROPE_HEAD_DIM).transpose(1, 2)

    # RoPE — this is where the difference is
    cos, sin = compute_cos_sin(position_ids, dim=QK_ROPE_HEAD_DIM, dtype=hidden_states.dtype)
    if use_fixed_rope:
        q_pe_n, k_pe_n = hf_apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
    else:
        q_pe_n, k_pe_n = neuron_apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

    # Weight absorption (Neuron style)
    wkv_b = weights["kv_b_proj"]
    wkv_b = wkv_b.view(NUM_HEADS, -1, KV_LORA_RANK)  # (H, nope+v, kv_lora_rank)
    q_absorb = wkv_b[:, :QK_NOPE_HEAD_DIM]             # (H, 128, 512)
    out_absorb = wkv_b[:, QK_NOPE_HEAD_DIM:, :]        # (H, 128, 512)

    q_nope_absorbed = torch.einsum("hdc,bhqd->bhqc", q_absorb, q_nope)

    # Attention scores via absorption
    active_scores = (
        torch.matmul(q_pe_n, k_pe_n.transpose(2, 3))
        + torch.einsum("bhqc,blc->bhql", q_nope_absorbed, compressed_kv)
    )
    active_scores *= SOFTMAX_SCALE

    # Causal mask (boolean, matching Neuron style)
    causal = torch.tril(torch.ones(q_len, q_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
    active_scores = torch.where(causal, active_scores, -1e9)

    active_scores = F.softmax(active_scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype)

    # V absorption
    x = torch.einsum("bhql,blc->bhqc", active_scores, compressed_kv)
    attn_output = torch.einsum("bhqc,hdc->bhqd", x, out_absorb)

    # Output projection
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, NUM_HEADS * V_HEAD_DIM)
    attn_output = F.linear(attn_output, weights["o_proj"])

    return attn_output, {
        "q": q, "q_nope": q_nope, "q_pe": q_pe,
        "compressed_kv": compressed_kv, "k_pe": k_pe,
        "q_pe_rope": q_pe_n, "k_pe_rope": k_pe_n,
        "attn_weights": active_scores, "attn_output_pre_proj": attn_output,
    }


# ─── Comparison helpers ─────────────────────────────────────────────────────

def cos_sim(a, b):
    return F.cosine_similarity(a.reshape(1, -1).float(), b.reshape(1, -1).float()).item()


def compare(name, hf_val, neuron_val, rtol=1e-5):
    cs = cos_sim(hf_val, neuron_val)
    max_diff = (hf_val.float() - neuron_val.float()).abs().max().item()
    status = "MATCH" if max_diff < rtol else ("OK" if cs > 0.999 else "DIFF")
    print(f"  {name:<30} cos_sim={cs:.6f}  max_diff={max_diff:.2e}  [{status}]")
    return cs, max_diff


# ─── Weight loading ─────────────────────────────────────────────────────────

def load_layer0_weights(model_path):
    """Load layer 0 attention weights from safetensors."""
    import json
    import os
    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    prefix = "model.layers.0.self_attn."
    keys = {
        "q_proj": f"{prefix}q_proj.weight",
        "kv_a_proj_with_mqa": f"{prefix}kv_a_proj_with_mqa.weight",
        "kv_a_layernorm": f"{prefix}kv_a_layernorm.weight",
        "kv_b_proj": f"{prefix}kv_b_proj.weight",
        "o_proj": f"{prefix}o_proj.weight",
    }

    weights = {}
    # Group by shard file to minimize opens
    file_to_keys = {}
    for short_name, full_key in keys.items():
        shard = index["weight_map"][full_key]
        file_to_keys.setdefault(shard, []).append((short_name, full_key))

    for shard, key_list in file_to_keys.items():
        filepath = os.path.join(model_path, shard)
        with safe_open(filepath, framework="pt") as f:
            for short_name, full_key in key_list:
                weights[short_name] = f.get_tensor(full_key).bfloat16()

    return weights


def load_embedding(model_path):
    """Load embedding weights."""
    import json
    import os
    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    key = "model.embed_tokens.weight"
    shard = index["weight_map"][key]
    with safe_open(os.path.join(model_path, shard), framework="pt") as f:
        return f.get_tensor(key).bfloat16()


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("Layer 0 Attention Comparison: HF vs Neuron (CPU)")
    print("=" * 80)

    # Tokenize
    sys.path.insert(0, MODEL_PATH)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    seq_len = input_ids.shape[1]
    print(f"\nPrompt: {PROMPT!r}  tokens: {input_ids[0].tolist()}  seq_len: {seq_len}")

    # Load weights
    print("\nLoading weights...")
    embed_weight = load_embedding(MODEL_PATH)
    weights = load_layer0_weights(MODEL_PATH)
    for k, v in weights.items():
        print(f"  {k}: {list(v.shape)}")

    # Embedding
    hidden_states = F.embedding(input_ids, embed_weight)
    print(f"\nhidden_states: {list(hidden_states.shape)}, dtype={hidden_states.dtype}")

    position_ids = torch.arange(seq_len).unsqueeze(0)

    # ─── Step-by-step comparison ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Step-by-step comparison (all on CPU, BF16)")
    print("=" * 80)

    # Run both forwards
    hf_out, hf_inter = hf_attention_forward(hidden_states, weights, position_ids)
    neuron_out, neuron_inter = neuron_attention_forward(hidden_states, weights, position_ids)

    print("\n--- Pre-RoPE (should be identical) ---")
    compare("Q projection", hf_inter["q"], neuron_inter["q"])
    compare("Q nope", hf_inter["q_nope"], neuron_inter["q_nope"])
    compare("Q pe (pre-RoPE)", hf_inter["q_pe"], neuron_inter["q_pe"])
    compare("compressed_kv", hf_inter["compressed_kv"], neuron_inter["compressed_kv"])
    compare("k_pe (pre-RoPE)", hf_inter["k_pe"], neuron_inter["k_pe"])

    print("\n--- RoPE (THE BUG: interleave transpose) ---")
    compare("q_pe after RoPE", hf_inter["q_pe_rope"], neuron_inter["q_pe_rope"])
    compare("k_pe after RoPE", hf_inter["k_pe_rope"], neuron_inter["k_pe_rope"])

    print("\n--- Post-RoPE (divergence propagates) ---")
    compare("attention weights", hf_inter["attn_weights"], neuron_inter["attn_weights"])
    compare("FINAL attention output", hf_out, neuron_out)

    # ─── With fix ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("With RoPE fix (add interleave transpose)")
    print("=" * 80)

    fixed_out, fixed_inter = neuron_attention_forward(
        hidden_states, weights, position_ids, use_fixed_rope=True
    )

    print("\n--- RoPE (should now match HF) ---")
    compare("q_pe after RoPE (fixed)", hf_inter["q_pe_rope"], fixed_inter["q_pe_rope"])
    compare("k_pe after RoPE (fixed)", hf_inter["k_pe_rope"], fixed_inter["k_pe_rope"])

    print("\n--- Post-fix attention ---")
    compare("attention weights (fixed)", hf_inter["attn_weights"], fixed_inter["attn_weights"])
    compare("FINAL attention output (fixed)", hf_out, fixed_out)

    # ─── Direct RoPE comparison ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"Direct RoPE element comparison (first head, first token, dim={QK_ROPE_HEAD_DIM})")
    print("=" * 80)

    q_pe = hf_inter["q_pe"][0, 0, 0, :]  # (QK_ROPE_HEAD_DIM,)
    cos_vals, sin_vals = compute_cos_sin(position_ids, dim=QK_ROPE_HEAD_DIM, dtype=torch.bfloat16)
    cos_0 = cos_vals[0, 0, 0, :]  # (QK_ROPE_HEAD_DIM,) for position 0
    sin_0 = sin_vals[0, 0, 0, :]

    # HF: transpose then rotate
    half = QK_ROPE_HEAD_DIM // 2
    q_transposed = q_pe.view(half, 2).transpose(0, 1).reshape(QK_ROPE_HEAD_DIM)
    q_hf = (q_transposed * cos_0) + (rotate_half(q_transposed.unsqueeze(0)).squeeze(0) * sin_0)

    # Neuron: rotate directly
    q_neuron = (q_pe * cos_0) + (rotate_half(q_pe.unsqueeze(0)).squeeze(0) * sin_0)

    print(f"\n  q_pe[:8] (interleaved):     {q_pe[:8].tolist()}")
    print(f"  q_transposed[:8] (split):   {q_transposed[:8].tolist()}")
    print(f"\n  HF  RoPE output[:8]:        {q_hf[:8].tolist()}")
    print(f"  Neuron RoPE output[:8]:     {q_neuron[:8].tolist()}")
    print(f"\n  cos_sim(hf, neuron):        {cos_sim(q_hf, q_neuron):.6f}")

    # For position 0, cos/sin are cos(0)=1, sin(0)=0 for all freqs
    # So for pos 0 both should be identical (no rotation). Check pos 1+
    if seq_len > 1:
        q_pe_1 = hf_inter["q_pe"][0, 0, 1, :]
        cos_1 = cos_vals[0, 0, 1, :]
        sin_1 = sin_vals[0, 0, 1, :]

        q_t_1 = q_pe_1.view(32, 2).transpose(0, 1).reshape(64)
        q_hf_1 = (q_t_1 * cos_1) + (rotate_half(q_t_1.unsqueeze(0)).squeeze(0) * sin_1)
        q_n_1 = (q_pe_1 * cos_1) + (rotate_half(q_pe_1.unsqueeze(0)).squeeze(0) * sin_1)

        print(f"\n  Position 1 (non-trivial rotation):")
        print(f"  cos_sim(hf, neuron) at pos1: {cos_sim(q_hf_1, q_n_1):.6f}")
        print(f"  max_diff at pos1:            {(q_hf_1.float() - q_n_1.float()).abs().max().item():.2e}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    hf_neuron_cos = cos_sim(hf_out, neuron_out)
    hf_fixed_cos = cos_sim(hf_out, fixed_out)
    print(f"\n  HF vs Neuron (broken RoPE):  cos_sim = {hf_neuron_cos:.6f}")
    print(f"  HF vs Neuron (fixed RoPE):   cos_sim = {hf_fixed_cos:.6f}")
    if hf_fixed_cos > hf_neuron_cos + 0.001:
        print(f"\n  FIX CONFIRMED: RoPE interleave transpose improves cos_sim by {hf_fixed_cos - hf_neuron_cos:.6f}")
    elif hf_neuron_cos > 0.9999:
        print(f"\n  Both match well — RoPE difference may be negligible for this input")
    else:
        print(f"\n  Unexpected: fix did not help. Other source of divergence.")


if __name__ == "__main__":
    main()
