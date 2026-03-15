#!/usr/bin/env python3
"""Capture per-layer hidden states from the HF Moonlight CPU reference model.

Runs the HF model on CPU in BF16 and saves intermediate hidden states at every
layer boundary. The saved tensors can then be compared against Neuron model
outputs to identify where divergence originates.

Captured tensors (all in float32 for precision):
  - embed_out:          after embed_tokens
  - layer_{i}_attn_in:  input to attention (after input_layernorm)
  - layer_{i}_attn_out: output of attention (before residual add)
  - layer_{i}_mlp_in:   input to MLP/MoE (after post_attention_layernorm)
  - layer_{i}_mlp_out:  output of MLP/MoE (before residual add)
  - layer_{i}_out:      full layer output (after both residual adds)
  - norm_out:           after final RMSNorm
  - logits:             lm_head output

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python capture_hf_intermediates.py
    python capture_hf_intermediates.py --prompt "Einstein was born in" --output /tmp/hf_states.pt
"""

import argparse
import time
from collections import OrderedDict

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache


def patch_dynamic_cache():
    """Patch DynamicCache for older HF model code compatibility."""
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None


def capture_intermediates(model, input_ids, attention_mask):
    """Run HF model forward with hooks to capture all intermediate hidden states."""
    captured = OrderedDict()
    hooks = []

    # Hook: embed_tokens output
    def hook_embed(module, input, output):
        captured["embed_out"] = output.detach().float().cpu()

    hooks.append(model.model.embed_tokens.register_forward_hook(hook_embed))

    # Hooks: per-layer sub-components
    for i, layer in enumerate(model.model.layers):
        def make_layer_hooks(idx, layer_module):
            def hook_input_layernorm(module, input, output):
                captured[f"layer_{idx}_attn_in"] = output.detach().float().cpu()
            hooks.append(layer_module.input_layernorm.register_forward_hook(hook_input_layernorm))

            def hook_self_attn(module, input, output):
                # output is a tuple; first element is the attention output
                attn_out = output[0] if isinstance(output, tuple) else output
                captured[f"layer_{idx}_attn_out"] = attn_out.detach().float().cpu()
            hooks.append(layer_module.self_attn.register_forward_hook(hook_self_attn))

            def hook_post_attn_layernorm(module, input, output):
                captured[f"layer_{idx}_mlp_in"] = output.detach().float().cpu()
            hooks.append(layer_module.post_attention_layernorm.register_forward_hook(hook_post_attn_layernorm))

            def hook_mlp(module, input, output):
                mlp_out = output[0] if isinstance(output, tuple) else output
                captured[f"layer_{idx}_mlp_out"] = mlp_out.detach().float().cpu()
            hooks.append(layer_module.mlp.register_forward_hook(hook_mlp))

            def hook_layer(module, input, output):
                layer_out = output[0] if isinstance(output, tuple) else output
                captured[f"layer_{idx}_out"] = layer_out.detach().float().cpu()
            hooks.append(layer_module.register_forward_hook(hook_layer))

        make_layer_hooks(i, layer)

    # Hook: final norm
    def hook_norm(module, input, output):
        captured["norm_out"] = output.detach().float().cpu()
    hooks.append(model.model.norm.register_forward_hook(hook_norm))

    # Hook: lm_head
    def hook_lm_head(module, input, output):
        captured["logits"] = output.detach().float().cpu()
    hooks.append(model.lm_head.register_forward_hook(hook_lm_head))

    # Forward pass
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    return captured


def print_summary(captured, tokenizer):
    """Print a summary of captured tensors."""
    print(f"\n{'='*70}")
    print(f"{'Key':<30} {'Shape':<25} {'Mean':>10} {'Std':>10} {'Max':>10}")
    print(f"{'='*70}")
    for key, tensor in captured.items():
        shape_str = str(list(tensor.shape))
        print(
            f"{key:<30} {shape_str:<25} "
            f"{tensor.mean().item():>10.4f} {tensor.std().item():>10.4f} {tensor.abs().max().item():>10.4f}"
        )

    # Print top-5 predicted tokens from final logits
    logits = captured["logits"]
    # Last position logits
    last_logits = logits[0, -1, :]
    top5_vals, top5_idx = last_logits.topk(5)
    print(f"\nTop-5 predictions (last position):")
    for i, (idx, val) in enumerate(zip(top5_idx, top5_vals)):
        token_str = tokenizer.decode([idx.item()])
        print(f"  {i+1}. token={idx.item():>6d} ({token_str!r:<15}) logit={val.item():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Capture HF Moonlight per-layer hidden states")
    parser.add_argument("--model_path", type=str,
                        default="/home/ubuntu/environment/models/Moonlight-16B-A3B")
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--output", type=str, default="/tmp/moonlight_hf_intermediates.pt",
                        help="Path to save captured tensors")
    args = parser.parse_args()

    print(f"Model : {args.model_path}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Output: {args.output}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    print(f"Tokens: {input_ids.shape[1]} -> {input_ids.tolist()}")

    # Load HF model
    print(f"\nLoading HF model in BF16 on CPU...")
    patch_dynamic_cache()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.0f}s")

    # Capture intermediates
    print(f"\nRunning forward pass with hooks...")
    t0 = time.time()
    captured = capture_intermediates(model, input_ids, attention_mask)
    print(f"  Forward pass took {time.time() - t0:.1f}s")
    print(f"  Captured {len(captured)} tensors")

    # Save
    # Also save metadata for reproducibility
    captured["_meta"] = {
        "prompt": args.prompt,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "model_path": args.model_path,
    }
    torch.save(captured, args.output)
    print(f"\n  Saved to {args.output}")

    # Summary
    print_summary({k: v for k, v in captured.items() if k != "_meta"}, tokenizer)


if __name__ == "__main__":
    main()
