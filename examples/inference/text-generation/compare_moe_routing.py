#!/usr/bin/env python3
"""Compare MoE routing between HF and Neuron implementations.

Loads router weights and the captured HF layer_1_mlp_in hidden states,
then runs both routing algorithms on CPU to compare expert selection.

Key difference under investigation:
  HF MoEGate (noaux_tc): topk(sigmoid(logits) + e_score_correction_bias)
  Neuron MoonlightRouter: topk(logits)  (no bias, selects on raw logits)

Since sigmoid is monotonic, topk(logits) == topk(sigmoid(logits)).
But adding e_score_correction_bias changes the ranking.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python compare_moe_routing.py
    python compare_moe_routing.py --layers 1 2 3 4 5 6
"""

import argparse

import torch
import torch.nn.functional as F
from safetensors import safe_open


def load_gate_weights(model_path, layer_idx):
    """Load gate weight and e_score_correction_bias from safetensors."""
    import json
    import os

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_key = f"model.layers.{layer_idx}.mlp.gate.weight"
    bias_key = f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"

    weight_file = os.path.join(model_path, index["weight_map"][weight_key])
    bias_file = os.path.join(model_path, index["weight_map"][bias_key])

    with safe_open(weight_file, framework="pt") as f:
        gate_weight = f.get_tensor(weight_key)

    with safe_open(bias_file, framework="pt") as f:
        e_score_correction_bias = f.get_tensor(bias_key)

    return gate_weight, e_score_correction_bias


def hf_routing(hidden_states_fp32, gate_weight, e_score_correction_bias, top_k=6,
               routed_scaling_factor=2.446, norm_topk_prob=True, n_group=1):
    """HF MoEGate forward (noaux_tc method), all in FP32."""
    T = hidden_states_fp32.shape[0]

    # Logits and sigmoid -- HF casts both to FP32
    logits = F.linear(hidden_states_fp32, gate_weight.float())
    scores = logits.sigmoid()

    # noaux_tc: add bias for selection
    scores_for_choice = scores + e_score_correction_bias.float().unsqueeze(0)

    # Group-based selection (n_group=1 means trivially one group)
    n_experts = scores_for_choice.shape[-1]
    experts_per_group = n_experts // n_group
    group_scores = (
        scores_for_choice.view(T, n_group, experts_per_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=1, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(T, n_group, experts_per_group)
        .reshape(T, -1)
    )
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    _, topk_idx = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False)

    # Weights from original scores (no bias)
    topk_weight = scores.gather(1, topk_idx)
    if top_k > 1 and norm_topk_prob:
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weight = topk_weight * routed_scaling_factor

    return topk_idx, topk_weight, scores, logits


def neuron_routing(hidden_states_bf16, gate_weight, top_k=6,
                   routed_scaling_factor=2.446, norm_topk_prob=True):
    """Neuron MoonlightRouter forward, matching current implementation."""
    # RouterTopK.get_router_logits: linear projection
    # The router uses dtype=torch.float32 for its linear layer, but the INPUT
    # hidden_states may be BF16 (coming from post_attention_layernorm).
    # The actual router linear_router is FP32, so logits are FP32.
    logits = F.linear(hidden_states_bf16.float(), gate_weight.float())

    # apply_activation_fn: sigmoid
    affinities = logits.sigmoid()

    # MoonlightRouter: topk on raw logits (NOT on sigmoid scores, NOT with bias)
    _, expert_index = torch.topk(logits, top_k)

    # Gather, normalize, scale
    topk_weights = affinities.gather(1, expert_index)
    if norm_topk_prob and top_k > 1:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * routed_scaling_factor

    return expert_index, topk_weights, affinities, logits


def neuron_routing_with_bias(hidden_states_bf16, gate_weight, e_score_correction_bias,
                             top_k=6, routed_scaling_factor=2.446, norm_topk_prob=True):
    """Hypothetical Neuron routing WITH e_score_correction_bias (proposed fix)."""
    logits = F.linear(hidden_states_bf16.float(), gate_weight.float())
    affinities = logits.sigmoid()

    # Add bias for selection (matching HF)
    scores_for_choice = affinities + e_score_correction_bias.float().unsqueeze(0)
    _, expert_index = torch.topk(scores_for_choice, top_k)

    # Weights from original affinities (no bias)
    topk_weights = affinities.gather(1, expert_index)
    if norm_topk_prob and top_k > 1:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * routed_scaling_factor

    return expert_index, topk_weights, affinities, logits


def analyze_layer(model_path, hf_data, layer_idx, top_k=6):
    """Analyze routing differences for a single layer."""
    print(f"\n{'='*80}")
    print(f"LAYER {layer_idx} MoE Routing Comparison")
    print(f"{'='*80}")

    # Load gate weights
    gate_weight, e_score_correction_bias = load_gate_weights(model_path, layer_idx)
    print(f"  gate_weight: {list(gate_weight.shape)}, dtype={gate_weight.dtype}")
    print(f"  e_score_correction_bias: {list(e_score_correction_bias.shape)}, "
          f"dtype={e_score_correction_bias.dtype}")
    print(f"  bias range: [{e_score_correction_bias.min():.4f}, {e_score_correction_bias.max():.4f}], "
          f"mean={e_score_correction_bias.mean():.4f}, std={e_score_correction_bias.std():.4f}")

    # Get MLP input (post_attention_layernorm output)
    mlp_in_key = f"layer_{layer_idx}_mlp_in"
    if mlp_in_key not in hf_data:
        print(f"  WARNING: {mlp_in_key} not found in HF intermediates, skipping")
        return None
    hidden_states_fp32 = hf_data[mlp_in_key].squeeze(0)  # (seq_len, hidden_size)
    hidden_states_bf16 = hidden_states_fp32.bfloat16()
    seq_len = hidden_states_fp32.shape[0]
    print(f"  hidden_states: seq_len={seq_len}, hidden_size={hidden_states_fp32.shape[1]}")

    # Run all three routing variants
    hf_idx, hf_wt, hf_scores, hf_logits = hf_routing(
        hidden_states_fp32, gate_weight, e_score_correction_bias, top_k=top_k
    )
    neuron_idx, neuron_wt, neuron_scores, neuron_logits = neuron_routing(
        hidden_states_bf16, gate_weight, top_k=top_k
    )
    fixed_idx, fixed_wt, _, _ = neuron_routing_with_bias(
        hidden_states_bf16, gate_weight, e_score_correction_bias, top_k=top_k
    )

    # Compare logits and scores
    logit_cos = F.cosine_similarity(
        hf_logits.reshape(1, -1), neuron_logits.reshape(1, -1)
    ).item()
    score_cos = F.cosine_similarity(
        hf_scores.reshape(1, -1), neuron_scores.reshape(1, -1)
    ).item()
    print(f"\n  Logits cos sim (HF FP32 vs Neuron BF16 input): {logit_cos:.6f}")
    print(f"  Sigmoid scores cos sim: {score_cos:.6f}")

    # Per-token expert selection comparison
    print(f"\n  Per-token expert selection (top-{top_k}):")
    print(f"  {'Token':<6} {'HF experts':<35} {'Neuron experts':<35} {'Match':<6} {'Fixed experts':<35} {'Fixed match'}")

    total_mismatches = 0
    total_fixed_mismatches = 0
    expert_diff_details = []

    for t in range(seq_len):
        hf_set = set(hf_idx[t].tolist())
        neuron_set = set(neuron_idx[t].tolist())
        fixed_set = set(fixed_idx[t].tolist())

        match = hf_set == neuron_set
        fixed_match = hf_set == fixed_set

        hf_sorted = sorted(hf_idx[t].tolist())
        neuron_sorted = sorted(neuron_idx[t].tolist())
        fixed_sorted = sorted(fixed_idx[t].tolist())

        if not match:
            total_mismatches += 1
        if not fixed_match:
            total_fixed_mismatches += 1

        match_str = "OK" if match else "DIFF"
        fixed_str = "OK" if fixed_match else "DIFF"

        print(f"  {t:<6} {str(hf_sorted):<35} {str(neuron_sorted):<35} {match_str:<6} "
              f"{str(fixed_sorted):<35} {fixed_str}")

        if not match:
            only_hf = hf_set - neuron_set
            only_neuron = neuron_set - hf_set
            expert_diff_details.append({
                "token": t,
                "only_hf": only_hf,
                "only_neuron": only_neuron,
            })

    print(f"\n  Expert selection mismatches: {total_mismatches}/{seq_len} tokens "
          f"({100*total_mismatches/seq_len:.0f}%)")
    print(f"  With bias fix:              {total_fixed_mismatches}/{seq_len} tokens "
          f"({100*total_fixed_mismatches/seq_len:.0f}%)")

    # Analyze WHY experts differ: look at score margins
    if expert_diff_details:
        print(f"\n  Mismatch analysis:")
        for d in expert_diff_details:
            t = d["token"]
            # Show the scores + bias for the swapped experts
            hf_s = hf_scores[t]  # sigmoid scores (no bias)
            bias = e_score_correction_bias.float()
            scores_with_bias = hf_s + bias

            print(f"\n    Token {t}: HF-only experts {d['only_hf']}, Neuron-only experts {d['only_neuron']}")
            for e in sorted(d["only_hf"] | d["only_neuron"]):
                label = "HF-only" if e in d["only_hf"] else "Neuron-only"
                print(f"      Expert {e:>2} ({label:>11}): "
                      f"logit={hf_logits[t, e]:.4f}, sigmoid={hf_s[e]:.4f}, "
                      f"bias={bias[e]:.4f}, sigmoid+bias={scores_with_bias[e]:.4f}")

            # Show the 6th and 7th ranked experts by each method
            hf_ranking = scores_with_bias[t] if t == 0 else scores_with_bias  # already per-token
            # Actually just show the sorted top-8 by each method
            neuron_top8_vals, neuron_top8_idx = torch.topk(neuron_logits[t], 8)
            hf_top8_vals, hf_top8_idx = torch.topk(scores_with_bias, 8)
            print(f"      Neuron top-8 (by logit): {list(zip(neuron_top8_idx.tolist(), [f'{v:.4f}' for v in neuron_top8_vals.tolist()]))}")
            print(f"      HF top-8 (by score+bias): {list(zip(hf_top8_idx.tolist(), [f'{v:.4f}' for v in hf_top8_vals.tolist()]))}")

    # Weight comparison for matching tokens
    print(f"\n  Affinity weight comparison (for tokens where experts match):")
    for t in range(seq_len):
        hf_set = set(hf_idx[t].tolist())
        neuron_set = set(neuron_idx[t].tolist())
        if hf_set == neuron_set:
            # Sort both by expert index for comparison
            hf_order = hf_idx[t].sort()[1]
            neuron_order = neuron_idx[t].sort()[1]
            hf_wt_sorted = hf_wt[t][hf_order]
            neuron_wt_sorted = neuron_wt[t][neuron_order]
            wt_diff = (hf_wt_sorted - neuron_wt_sorted).abs().max().item()
            print(f"    Token {t}: max weight diff = {wt_diff:.6f}")

    return {
        "layer": layer_idx,
        "mismatches": total_mismatches,
        "total_tokens": seq_len,
        "fixed_mismatches": total_fixed_mismatches,
        "details": expert_diff_details,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare MoE routing HF vs Neuron")
    parser.add_argument("--model_path", type=str,
                        default="/home/ubuntu/environment/models/Moonlight-16B-A3B")
    parser.add_argument("--hf_intermediates", type=str,
                        default="/tmp/moonlight_hf_intermediates.pt")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    args = parser.parse_args()

    print("MoE Routing Comparison: HF vs Neuron")
    print(f"Model: {args.model_path}")
    print(f"HF intermediates: {args.hf_intermediates}")

    hf_data = torch.load(args.hf_intermediates, weights_only=False)
    print(f"Prompt: {hf_data['_meta']['prompt']!r}")

    results = []
    for layer_idx in args.layers:
        r = analyze_layer(args.model_path, hf_data, layer_idx)
        if r:
            results.append(r)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Layer':<8} {'Mismatches':<15} {'With bias fix':<15}")
    for r in results:
        print(f"  {r['layer']:<8} {r['mismatches']}/{r['total_tokens']:<12} "
              f"{r['fixed_mismatches']}/{r['total_tokens']}")


if __name__ == "__main__":
    main()
