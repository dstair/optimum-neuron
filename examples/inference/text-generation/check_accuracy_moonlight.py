#!/usr/bin/env python3
"""
Logit divergence test for Moonlight 16B-A3B on Trainium via Optimum Neuron.

Compares the Neuron model's output logits against a HuggingFace BF16 CPU
reference model at each decode position, reporting per-token normalized error
and top-1 agreement.

Requires a separate compilation with on_device_sampling=False so the Neuron
model returns raw logits instead of sampled token IDs.  The script handles
this automatically, caching the accuracy-check model separately from the
interactive model.

Usage (trn2.3xlarge):
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python check_accuracy_moonlight.py --num_tokens 5

    # Use a different prompt
    python check_accuracy_moonlight.py --prompt "Einstein was born in" --num_tokens 10
"""

import argparse
import gc
import glob
import os
import shutil
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

from optimum.neuron import NeuronModelForCausalLM


def patch_dynamic_cache():
    """Patch DynamicCache for older HF model code compatibility."""
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None


def export_accuracy_model(model_path, compiled_path, tp_degree, seq_length):
    """Compile model with on_device_sampling=False for logit access."""
    print(f"Compiling accuracy-check model (on_device_sampling=False)...")
    print(f"  This takes ~5 min (cached NEFFs may speed it up).")
    t0 = time.time()

    neuron_config = NeuronModelForCausalLM.get_neuron_config(
        model_name_or_path=model_path,
        batch_size=1,
        sequence_length=seq_length,
        tensor_parallel_size=tp_degree,
    )
    # Override: disable on-device sampling so forward() returns logits
    neuron_config.on_device_sampling = False

    model = NeuronModelForCausalLM.export(
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


def load_neuron_model(model_path, compiled_path, tp_degree, seq_length, recompile):
    """Load or compile the accuracy-check Neuron model."""
    config_file = os.path.join(compiled_path, "config.json")
    if recompile or not os.path.isfile(config_file):
        return export_accuracy_model(model_path, compiled_path, tp_degree, seq_length)

    print(f"Loading compiled accuracy model from {compiled_path}...")
    t0 = time.time()
    model = NeuronModelForCausalLM.from_pretrained(compiled_path, trust_remote_code=True)
    print(f"  Loaded in {time.time() - t0:.0f}s")
    return model


def load_hf_model(model_path):
    """Load HF reference model in BF16 on CPU."""
    print(f"Loading HF reference model in BF16...")
    t0 = time.time()
    patch_dynamic_cache()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.0f}s")
    return model


def neuron_forward_context(neuron_model, input_ids):
    """Run context encoding on Neuron, return logits for last prompt token."""
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
    # outputs shape: (batch, 1, vocab_size) — logits for last prompt position
    return outputs


def neuron_forward_token(neuron_model, token_id, position):
    """Run single-token decode on Neuron, return logits."""
    from optimum.neuron.models.inference.backend.modules.generation.generation_utils import (
        prepare_sampling_params,
    )

    input_ids = token_id.view(1, 1)
    position_ids = torch.tensor([[position]])
    seq_ids = torch.arange(1)
    sampling_params = prepare_sampling_params(batch_size=1, top_k=1, top_p=1.0, temperature=1.0)

    with torch.inference_mode():
        outputs = neuron_model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
        )
    return outputs


def hf_forward(hf_model, input_ids, attention_mask=None, past_key_values=None):
    """Run HF model forward, return (logits_last_pos, past_key_values)."""
    with torch.inference_mode():
        out = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
    # out.logits: (batch, seq_len, vocab_size) — take last position
    logits = out.logits[:, -1:, :].float()
    return logits, out.past_key_values


def compute_metrics(neuron_logits, hf_logits, tokenizer=None, top_k=50):
    """Compute logit comparison metrics between Neuron and HF outputs.

    Returns dict with:
      - top1_match: whether argmax agrees
      - neuron_token / hf_token: argmax token IDs
      - cosine_sim: cosine similarity of full logit vectors
      - top{k}_max_norm_error: max |diff|/max(|hf|) over top-K HF logits
      - top{k}_mean_norm_error: mean |diff|/max(|hf|) over top-K HF logits
    """
    n_logits = neuron_logits.squeeze().float()
    h_logits = hf_logits.squeeze().float()

    neuron_tok = n_logits.argmax().item()
    hf_tok = h_logits.argmax().item()

    # Cosine similarity
    cos = torch.nn.functional.cosine_similarity(n_logits.unsqueeze(0), h_logits.unsqueeze(0)).item()

    # Top-K normalized error (NXDI-style)
    _, topk_idx = h_logits.topk(top_k)
    n_topk = n_logits[topk_idx]
    h_topk = h_logits[topk_idx]
    diffs = (n_topk - h_topk).abs()
    scale = h_logits.abs().max().clamp(min=1e-6)
    norm_errors = diffs / scale
    max_norm = norm_errors.max().item()
    mean_norm = norm_errors.mean().item()

    # Debug: print top-5 tokens from each model
    if tokenizer is not None:
        n_top5_vals, n_top5_idx = n_logits.topk(5)
        h_top5_vals, h_top5_idx = h_logits.topk(5)
        print(f"  [debug] Neuron top-5: {[(idx.item(), f'{val.item():.3f}', tokenizer.decode([idx.item()])) for idx, val in zip(n_top5_idx, n_top5_vals)]}")
        print(f"  [debug] HF     top-5: {[(idx.item(), f'{val.item():.3f}', tokenizer.decode([idx.item()])) for idx, val in zip(h_top5_idx, h_top5_vals)]}")

    return {
        "top1_match": neuron_tok == hf_tok,
        "neuron_token": neuron_tok,
        "hf_token": hf_tok,
        "cosine_sim": cos,
        f"top{top_k}_max_norm_error": max_norm,
        f"top{top_k}_mean_norm_error": mean_norm,
    }


def main():
    parser = argparse.ArgumentParser(description="Moonlight logit divergence test (Optimum Neuron)")
    parser.add_argument("--model_path", type=str,
                        default="/home/ubuntu/environment/models/Moonlight-16B-A3B")
    parser.add_argument("--compiled_model_path", type=str,
                        default="/tmp/moonlight_accuracy_compiled",
                        help="Path for accuracy-check compiled model (on_device_sampling=False)")
    parser.add_argument("--tp_degree", type=int, default=2)
    parser.add_argument("--sequence_length", type=int, default=4096)
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--num_tokens", type=int, default=5,
                        help="Number of decode tokens to compare (1 = context encoding only)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-K logits to use for normalized error")
    parser.add_argument("--max_norm_error_tol", type=float, default=0.50,
                        help="Max allowed top-K normalized error per token (BF16 MoE default: 0.50)")
    parser.add_argument("--recompile", action="store_true",
                        help="Force recompilation of accuracy model")
    args = parser.parse_args()

    print("=" * 60)
    print("Moonlight 16B Logit Divergence Test (Optimum Neuron)")
    print("=" * 60)
    print(f"  Prompt         : {args.prompt!r}")
    print(f"  Tokens to check: {args.num_tokens}")
    print(f"  Top-K          : {args.top_k}")
    print(f"  Tolerance      : {args.max_norm_error_tol}")
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens["input_ids"]
    prompt_len = input_ids.shape[1]
    print(f"  Prompt tokens  : {prompt_len}")
    print()

    # Load Neuron model (compile if needed)
    neuron_model = load_neuron_model(
        args.model_path, args.compiled_model_path,
        args.tp_degree, args.sequence_length, args.recompile,
    )

    # Load HF reference model
    hf_model = load_hf_model(args.model_path)

    # === Context encoding (prefill) ===
    print(f"\n{'='*60}")
    print("Step 0: Context encoding (prefill)")
    print(f"{'='*60}")

    neuron_logits = neuron_forward_context(neuron_model, input_ids)
    print(f"  [debug] Neuron raw output: type={type(neuron_logits)}, shape={neuron_logits.shape if hasattr(neuron_logits, 'shape') else 'N/A'}, dtype={neuron_logits.dtype if hasattr(neuron_logits, 'dtype') else 'N/A'}")
    # neuron_logits: (batch, 1, vocab) or (batch, seq, vocab)
    # We want the last position
    if neuron_logits.dim() == 3 and neuron_logits.shape[1] > 1:
        neuron_logits = neuron_logits[:, -1:, :]
    neuron_logits = neuron_logits.float().cpu()

    hf_attn_mask = torch.ones(1, prompt_len, dtype=torch.long)
    hf_logits, hf_kv = hf_forward(hf_model, input_ids.to(torch.long), attention_mask=hf_attn_mask)
    hf_logits = hf_logits.cpu()

    m = compute_metrics(neuron_logits, hf_logits, tokenizer=tokenizer, top_k=args.top_k)
    neuron_tok = m["neuron_token"]
    hf_tok = m["hf_token"]
    print(f"  Neuron top-1: {neuron_tok} ({tokenizer.decode([neuron_tok])!r})")
    print(f"  HF     top-1: {hf_tok} ({tokenizer.decode([hf_tok])!r})")
    print(f"  Top-1 match : {m['top1_match']}")
    print(f"  Cosine sim  : {m['cosine_sim']:.6f}")
    print(f"  Top-{args.top_k} max norm error: {m[f'top{args.top_k}_max_norm_error']:.6f}")
    print(f"  Top-{args.top_k} mean norm error: {m[f'top{args.top_k}_mean_norm_error']:.6f}")

    all_metrics = [m]
    # Use the HF top-1 token as the next input for both models (teacher-forced)
    next_token = torch.tensor([hf_tok])
    current_pos = prompt_len

    # === Token generation (decode) ===
    for step in range(1, args.num_tokens):
        print(f"\n{'='*60}")
        print(f"Step {step}: Token generation (pos={current_pos})")
        print(f"{'='*60}")

        # Neuron decode
        neuron_logits = neuron_forward_token(neuron_model, next_token, current_pos)
        if neuron_logits.dim() == 3:
            neuron_logits = neuron_logits[:, -1:, :]
        neuron_logits = neuron_logits.float().cpu()

        # HF decode
        hf_input = next_token.view(1, 1).to(torch.long)
        hf_attn_mask = torch.ones(1, prompt_len + step, dtype=torch.long)
        hf_logits, hf_kv = hf_forward(hf_model, hf_input, attention_mask=hf_attn_mask, past_key_values=hf_kv)
        hf_logits = hf_logits.cpu()

        m = compute_metrics(neuron_logits, hf_logits, tokenizer=tokenizer, top_k=args.top_k)
        neuron_tok = m["neuron_token"]
        hf_tok = m["hf_token"]
        print(f"  Input token : {next_token.item()} ({tokenizer.decode([next_token.item()])!r})")
        print(f"  Neuron top-1: {neuron_tok} ({tokenizer.decode([neuron_tok])!r})")
        print(f"  HF     top-1: {hf_tok} ({tokenizer.decode([hf_tok])!r})")
        print(f"  Top-1 match : {m['top1_match']}")
        print(f"  Cosine sim  : {m['cosine_sim']:.6f}")
        print(f"  Top-{args.top_k} max norm error: {m[f'top{args.top_k}_max_norm_error']:.6f}")
        print(f"  Top-{args.top_k} mean norm error: {m[f'top{args.top_k}_mean_norm_error']:.6f}")

        all_metrics.append(m)
        next_token = torch.tensor([hf_tok])
        current_pos += 1

    # === Summary ===
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    top1_matches = sum(1 for m in all_metrics if m["top1_match"])
    max_errors = [m[f"top{args.top_k}_max_norm_error"] for m in all_metrics]
    mean_errors = [m[f"top{args.top_k}_mean_norm_error"] for m in all_metrics]
    cosines = [m["cosine_sim"] for m in all_metrics]
    worst_error = max(max_errors)

    print(f"  Tokens checked  : {len(all_metrics)}")
    print(f"  Top-1 matches   : {top1_matches}/{len(all_metrics)}")
    print(f"  Cosine sim      : min={min(cosines):.6f}, mean={sum(cosines)/len(cosines):.6f}")
    print(f"  Top-{args.top_k} max norm err: worst={worst_error:.6f}, mean={sum(max_errors)/len(max_errors):.6f}")
    print(f"  Top-{args.top_k} mean norm err: worst={max(mean_errors):.6f}, mean={sum(mean_errors)/len(mean_errors):.6f}")
    print(f"  Tolerance       : {args.max_norm_error_tol}")

    if worst_error > args.max_norm_error_tol:
        print(f"\n  FAIL: worst top-{args.top_k} max norm error ({worst_error:.6f}) > tolerance ({args.max_norm_error_tol})")
        sys.exit(1)
    else:
        print(f"\n  PASS: all tokens within tolerance")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
