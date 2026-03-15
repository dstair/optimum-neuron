#!/usr/bin/env python3
"""Per-layer divergence analysis between Neuron and HF Moonlight models.

Compiles an instrumented Moonlight model that outputs per-layer hidden states
alongside logits, then compares against the HF intermediates captured by
capture_hf_intermediates.py.

The instrumented model packs layer hidden states into the output tensor by
concatenating them after the logits along the vocab dimension. This avoids
modifying the Optimum Neuron wrapper infrastructure.

Output tensor layout (context encoding, last position only):
  [:, :, :vocab_size]  = logits (float32)
  [:, :, vocab_size + i*hidden_size : vocab_size + (i+1)*hidden_size]  = layer i hidden state

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

    # Step 1: Capture HF intermediates (if not already done)
    python capture_hf_intermediates.py

    # Step 2: Run divergence analysis
    python analyze_divergence.py
    python analyze_divergence.py --recompile  # force recompile
"""

import argparse
import glob
import os
import shutil
import time

import torch
from transformers import AutoTokenizer, DynamicCache

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.models.inference.backend.config import NxDNeuronConfig
from optimum.neuron.models.inference.moonlight.modeling_moonlight import (
    MoonlightNxDModelForCausalLM,
    NxDMoonlightModel,
    _patch_neuron_config_skip_weights,
)


def patch_dynamic_cache():
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None


# ---------------------------------------------------------------------------
# Instrumented model: captures per-layer hidden states in the output tensor
# ---------------------------------------------------------------------------

class InstrumentedNxDMoonlightModel(NxDMoonlightModel):
    """NxDMoonlightModel that packs per-layer hidden states into the output.

    Captures three kinds of intermediate states:
    1. Embedding output (1 tensor)
    2. Post-attention hidden state per layer (after attn + residual, before MLP)
    3. Post-layer hidden state per layer (after MLP + residual)

    For context encoding, gathers the last-position hidden state.
    For token generation, hidden states are already (batch, 1, hidden).

    Output tensor layout (dim=2):
      [:vocab_size]  = logits
      [vocab_size : vocab_size + H]  = embed_out
      [vocab_size + H + i*2H : vocab_size + H + i*2H + H]  = layer i post-attn
      [vocab_size + H + i*2H + H : vocab_size + H + (i+1)*2H]  = layer i post-layer
    """

    def forward(self, input_ids, position_ids, seq_ids, sampling_params):
        from optimum.neuron.models.inference.backend.modules.generation.sampling import mask_padded_logits

        is_for_context_encoding = self._is_context_encoding(input_ids)
        cache_size = self.n_positions
        device = input_ids.device

        # -- Prepare masks (same as base class) --
        if is_for_context_encoding:
            past_key_values = None
            attention_mask = torch.full(
                (self.n_positions, self.n_positions), True, device=device
            ).tril(diagonal=0)
            attention_mask = attention_mask[None, None, :, :].expand(
                self.batch_size, 1, self.n_positions, self.n_positions
            )
            active_mask = None
        else:
            past_key_values = self.kv_mgr.get_cache(cache_size)
            max_cached_positions = position_ids.expand(self.batch_size, self.n_positions) - 1
            all_positions = (
                torch.arange(self.n_positions, device=device)
                .view(1, -1)
                .expand(self.batch_size, self.n_positions)
            )
            attention_mask = (max_cached_positions >= all_positions).view(
                self.batch_size, 1, 1, self.n_positions
            )
            active_mask = None

        batch_size, seq_length = input_ids.shape[:2]
        position_ids = position_ids.view(-1, seq_length).long()

        # -- Compute gather index for last position (context encoding) --
        hidden_size = self.config.hidden_size
        if is_for_context_encoding:
            gather_idx = torch.max(position_ids, dim=1, keepdim=True).indices
            gather_idx_expanded = gather_idx.unsqueeze(1).expand(batch_size, 1, hidden_size)

        def _gather_last(t):
            """Extract last-position hidden state."""
            if is_for_context_encoding:
                return torch.gather(t, dim=1, index=gather_idx_expanded)
            return t  # already (batch, 1, hidden)

        # -- Embed --
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # Capture embedding output
        captures = [_gather_last(hidden_states).float()]

        # -- Decoder layers with sub-component capture --
        new_key_values = []
        cos_cache = None
        sin_cache = None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # === Inline decoder layer forward to capture post-attention state ===
            # Step 1: input_layernorm + attention + residual
            residual = hidden_states
            normed = decoder_layer.input_layernorm(hidden_states)
            attn_out, present_key_value, cos_cache, sin_cache = decoder_layer.self_attn(
                hidden_states=normed,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
            )
            hidden_states = residual + attn_out

            # Capture post-attention hidden state
            captures.append(_gather_last(hidden_states).float())

            # Step 2: post_attention_layernorm + MLP/MoE + residual
            residual = hidden_states
            normed = decoder_layer.post_attention_layernorm(hidden_states)
            if decoder_layer._is_dense:
                mlp_out = decoder_layer.mlp(normed)
            else:
                mlp_out = decoder_layer.mlp(normed)[0]
            hidden_states = residual + mlp_out

            new_key_values.append(present_key_value)

            # Capture post-layer hidden state
            captures.append(_gather_last(hidden_states).float())

        hidden_states = self.norm(hidden_states)

        # -- KV cache update --
        updated_kv_cache = self.kv_mgr.update_cache(
            is_for_context_encoding=is_for_context_encoding,
            seq_ids=seq_ids,
            position_ids=position_ids,
            new_key_values=new_key_values,
            seq_len=cache_size,
        )

        # -- Logits --
        if is_for_context_encoding:
            hidden_states = torch.gather(hidden_states, dim=1, index=gather_idx_expanded)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if self.lm_head.gather_output:
            rank_id = torch.tensor(0, device=logits.device, dtype=torch.int32)
        else:
            rank_id = self.rank_util.get_rank()

        if hasattr(self.lm_head, "pad_size"):
            logits = mask_padded_logits(
                logits, rank_id,
                torch.distributed.get_world_size(group=self.lm_head.tensor_parallel_group)
                if not self.lm_head.gather_output else 1,
                pad_size=self.lm_head.pad_size,
            )

        # -- Pack all captures after logits --
        # captures: [embed_out, layer0_post_attn, layer0_post_layer, layer1_post_attn, ...]
        # Total: 1 + 27*2 = 55 tensors, each (batch, 1, hidden_size)
        stacked = torch.cat(captures, dim=2)
        combined = torch.cat([logits, stacked], dim=2)

        outputs = [combined]
        outputs += updated_kv_cache
        return outputs


class InstrumentedMoonlightCausalLM(MoonlightNxDModelForCausalLM):
    """CausalLM wrapper that uses the instrumented model."""
    _model_cls = InstrumentedNxDMoonlightModel


# ---------------------------------------------------------------------------
# Compilation and execution
# ---------------------------------------------------------------------------

def export_instrumented_model(model_path, compiled_path, tp_degree, seq_length):
    print(f"Compiling instrumented model (with per-layer hidden states)...")
    print(f"  This takes ~5 min.")
    t0 = time.time()

    neuron_config = NeuronModelForCausalLM.get_neuron_config(
        model_name_or_path=model_path,
        batch_size=1,
        sequence_length=seq_length,
        tensor_parallel_size=tp_degree,
    )
    neuron_config.on_device_sampling = False
    _patch_neuron_config_skip_weights(neuron_config)

    # Use InstrumentedMoonlightCausalLM for export
    model = InstrumentedMoonlightCausalLM.export(
        model_id=model_path,
        neuron_config=neuron_config,
        load_weights=True,
        trust_remote_code=True,
    )
    model.save_pretrained(compiled_path)
    for py_file in glob.glob(os.path.join(model_path, "*.py")):
        shutil.copy2(py_file, compiled_path)
    print(f"  Compiled in {time.time() - t0:.0f}s, saved to {compiled_path}")
    return model


def load_instrumented_model(model_path, compiled_path, tp_degree, seq_length, recompile):
    config_file = os.path.join(compiled_path, "config.json")
    if recompile or not os.path.isfile(config_file):
        return export_instrumented_model(model_path, compiled_path, tp_degree, seq_length)

    print(f"Loading compiled instrumented model from {compiled_path}...")
    t0 = time.time()
    model = InstrumentedMoonlightCausalLM.from_pretrained(compiled_path, trust_remote_code=True)
    print(f"  Loaded in {time.time() - t0:.0f}s")
    return model


def run_neuron_context(neuron_model, input_ids):
    from optimum.neuron.models.inference.backend.modules.generation.generation_utils import (
        prepare_sampling_params,
    )

    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    seq_ids = torch.arange(batch_size)
    sampling_params = prepare_sampling_params(batch_size=batch_size, top_k=1, top_p=1.0, temperature=1.0)

    neuron_model.reset()
    with torch.inference_mode():
        outputs = neuron_model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
        )
    return outputs


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def unpack_outputs(outputs, vocab_size, num_layers, hidden_size):
    """Split combined output into logits, embed, post-attn, and post-layer hidden states.

    Layout: [logits | embed | layer0_post_attn | layer0_post_layer | layer1_post_attn | ...]
    Total captures: 1 (embed) + num_layers * 2 (post_attn + post_layer)
    """
    combined = outputs.float().cpu()
    logits = combined[:, :, :vocab_size]
    captures = combined[:, :, vocab_size:]

    batch = combined.shape[0]
    H = hidden_size
    num_captures = 1 + num_layers * 2  # embed + (post_attn + post_layer) per layer

    # Split into individual hidden states
    embed_out = captures[:, :, :H]
    post_attn = []
    post_layer = []
    for i in range(num_layers):
        offset = H + i * 2 * H
        post_attn.append(captures[:, :, offset : offset + H])
        post_layer.append(captures[:, :, offset + H : offset + 2 * H])

    # Stack: (num_layers, batch, 1, hidden_size)
    post_attn = torch.stack(post_attn, dim=0)
    post_layer = torch.stack(post_layer, dim=0)
    return logits, embed_out, post_attn, post_layer


def compare_layer(neuron_hidden, hf_hidden, layer_idx):
    """Compare hidden states from one layer, return metrics dict."""
    n = neuron_hidden.squeeze().float()
    h = hf_hidden.squeeze().float()

    cos = torch.nn.functional.cosine_similarity(n.unsqueeze(0), h.unsqueeze(0)).item()
    abs_diff = (n - h).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    scale = h.abs().max().clamp(min=1e-6)
    max_rel = (abs_diff / scale).max().item()
    mean_rel = (abs_diff / scale).mean().item()

    return {
        "layer": layer_idx,
        "cosine_sim": cos,
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
        "max_rel_err": max_rel,
        "mean_rel_err": mean_rel,
        "neuron_norm": n.norm().item(),
        "hf_norm": h.norm().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Moonlight per-layer divergence analysis")
    parser.add_argument("--model_path", type=str,
                        default="/home/ubuntu/environment/models/Moonlight-16B-A3B")
    parser.add_argument("--compiled_model_path", type=str,
                        default="/tmp/moonlight_instrumented_compiled",
                        help="Path for instrumented compiled model")
    parser.add_argument("--hf_intermediates", type=str,
                        default="/tmp/moonlight_hf_intermediates.pt",
                        help="Path to HF intermediates from capture_hf_intermediates.py")
    parser.add_argument("--tp_degree", type=int, default=2)
    parser.add_argument("--sequence_length", type=int, default=4096)
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--recompile", action="store_true")
    args = parser.parse_args()

    patch_dynamic_cache()

    print("=" * 70)
    print("Moonlight Per-Layer Divergence Analysis")
    print("=" * 70)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens["input_ids"]
    print(f"  Prompt: {args.prompt!r}")
    print(f"  Tokens: {input_ids.shape[1]}")

    # Load HF intermediates
    print(f"\nLoading HF intermediates from {args.hf_intermediates}...")
    hf_data = torch.load(args.hf_intermediates, weights_only=False)
    meta = hf_data["_meta"]
    assert meta["prompt"] == args.prompt, (
        f"Prompt mismatch: HF intermediates used {meta['prompt']!r}, "
        f"but this script uses {args.prompt!r}"
    )

    # Count layers from HF data
    import re
    num_layers = sum(1 for k in hf_data if re.match(r"^layer_\d+_out$", k))
    hidden_size = hf_data["layer_0_out"].shape[-1]
    vocab_size = hf_data["logits"].shape[-1]
    print(f"  Layers: {num_layers}, hidden_size: {hidden_size}, vocab_size: {vocab_size}")

    # Load/compile instrumented Neuron model
    neuron_model = load_instrumented_model(
        args.model_path, args.compiled_model_path,
        args.tp_degree, args.sequence_length, args.recompile,
    )

    # Run instrumented model
    print(f"\nRunning instrumented Neuron model...")
    t0 = time.time()
    outputs = run_neuron_context(neuron_model, input_ids)
    print(f"  Forward pass took {time.time() - t0:.1f}s")

    # Unpack
    neuron_logits, neuron_embed, neuron_post_attn, neuron_post_layer = unpack_outputs(
        outputs, vocab_size, num_layers, hidden_size
    )
    print(f"  Logits shape: {list(neuron_logits.shape)}")
    print(f"  Embed shape: {list(neuron_embed.shape)}")
    print(f"  Post-attn shape: {list(neuron_post_attn.shape)}")
    print(f"  Post-layer shape: {list(neuron_post_layer.shape)}")

    # Compare embedding
    hf_embed = hf_data["embed_out"][:, -1:, :]
    embed_m = compare_layer(neuron_embed, hf_embed, -1)
    print(f"\n{'=' * 90}")
    print(f"EMBEDDING: cos_sim={embed_m['cosine_sim']:.6f}, max_abs={embed_m['max_abs_err']:.6f}, "
          f"N_norm={embed_m['neuron_norm']:.4f}, HF_norm={embed_m['hf_norm']:.4f}")
    print(f"{'=' * 90}")

    # Compare per-layer: post-attention and post-layer
    print(f"\n{'Layer':<6} {'--- Post-Attention ---':^36} {'--- Post-Layer (full) ---':^36}")
    print(f"{'':6} {'CosSim':>10} {'MaxAbs':>10} {'N_norm':>8} {'HF_norm':>8}"
          f" {'CosSim':>10} {'MaxAbs':>10} {'N_norm':>8} {'HF_norm':>8}")
    print(f"{'=' * 90}")

    attn_metrics = []
    layer_metrics = []
    for i in range(num_layers):
        # Post-attention comparison
        # HF post-attention = embed_out + attn_out = layer input + attention residual
        # The HF intermediates capture: layer_X_attn_out (just attn output, before residual)
        # and we can reconstruct post-attention as: previous layer output + attn_output
        # But simpler: use the Neuron post-attn directly vs reconstruct from HF data

        # For HF: post-attention state = previous_layer_out + attn_out
        if i == 0:
            hf_prev = hf_data["embed_out"][:, -1:, :]
        else:
            hf_prev = hf_data[f"layer_{i-1}_out"][:, -1:, :]
        hf_attn_out = hf_data[f"layer_{i}_attn_out"][:, -1:, :]
        hf_post_attn = (hf_prev + hf_attn_out).float()

        neuron_pa = neuron_post_attn[i]
        m_attn = compare_layer(neuron_pa, hf_post_attn, i)
        attn_metrics.append(m_attn)

        # Post-layer comparison
        hf_layer_out = hf_data[f"layer_{i}_out"][:, -1:, :]
        neuron_pl = neuron_post_layer[i]
        m_layer = compare_layer(neuron_pl, hf_layer_out, i)
        layer_metrics.append(m_layer)

        is_dense = " D" if i == 0 else ""
        print(f"  {i:<4}{is_dense} {m_attn['cosine_sim']:>10.6f} {m_attn['max_abs_err']:>10.4f} "
              f"{m_attn['neuron_norm']:>8.2f} {m_attn['hf_norm']:>8.2f}"
              f" {m_layer['cosine_sim']:>10.6f} {m_layer['max_abs_err']:>10.4f} "
              f"{m_layer['neuron_norm']:>8.2f} {m_layer['hf_norm']:>8.2f}")

    # Summary
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")

    attn_cosines = [m["cosine_sim"] for m in attn_metrics]
    layer_cosines = [m["cosine_sim"] for m in layer_metrics]

    print(f"  Embedding cos sim: {embed_m['cosine_sim']:.6f}")
    print(f"  Post-attn cos sim: min={min(attn_cosines):.6f} (layer {attn_cosines.index(min(attn_cosines))}), "
          f"mean={sum(attn_cosines)/len(attn_cosines):.6f}")
    print(f"  Post-layer cos sim: min={min(layer_cosines):.6f} (layer {layer_cosines.index(min(layer_cosines))}), "
          f"mean={sum(layer_cosines)/len(layer_cosines):.6f}")

    # Attention vs MLP contribution to error
    print(f"\n  Per-layer error attribution (cos sim drop from post-attn to post-layer):")
    print(f"  {'Layer':<8} {'PostAttn':>10} {'PostLayer':>10} {'MLP drop':>10} {'Attn drop':>10}")
    for i in range(num_layers):
        prev_cos = embed_m["cosine_sim"] if i == 0 else layer_cosines[i - 1]
        attn_drop = prev_cos - attn_cosines[i]
        mlp_drop = attn_cosines[i] - layer_cosines[i]
        label = "dense" if i == 0 else "MoE"
        print(f"  {i:<6} ({label:>5}) {attn_cosines[i]:>10.6f} {layer_cosines[i]:>10.6f} "
              f"{mlp_drop:>+10.6f} {attn_drop:>+10.6f}")

    # Compare final logits
    hf_logits = hf_data["logits"][:, -1:, :].float()
    cos_logits = torch.nn.functional.cosine_similarity(
        neuron_logits.reshape(1, -1), hf_logits.reshape(1, -1)
    ).item()
    print(f"\n  Final logits cosine sim: {cos_logits:.6f}")

    # Top-5 comparison
    n_top5_vals, n_top5_idx = neuron_logits.squeeze().topk(5)
    h_top5_vals, h_top5_idx = hf_logits.squeeze().topk(5)
    print(f"\n  Neuron top-5: {[(idx.item(), tokenizer.decode([idx.item()])) for idx in n_top5_idx]}")
    print(f"  HF     top-5: {[(idx.item(), tokenizer.decode([idx.item()])) for idx in h_top5_idx]}")

    # Save results
    results_path = "/tmp/moonlight_divergence_results.pt"
    torch.save({
        "attn_metrics": attn_metrics,
        "layer_metrics": layer_metrics,
        "embed_metrics": embed_m,
        "neuron_logits": neuron_logits,
        "neuron_embed": neuron_embed,
        "neuron_post_attn": neuron_post_attn,
        "neuron_post_layer": neuron_post_layer,
        "prompt": args.prompt,
    }, results_path)
    print(f"\n  Results saved to {results_path}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
