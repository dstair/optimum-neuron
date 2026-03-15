#!/usr/bin/env python3
"""
Interactive shell for Moonlight 16B-A3B on Trainium via Optimum Neuron.

Exports (compiles) the model on first run, then loads from cache on subsequent
runs. Enters a REPL where you can type prompts and get text completions.

Usage (trn2.3xlarge, 4 NeuronCores):
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    pip install -e /home/ubuntu/environment/optimum-neuron  # first time only
    python interactive_moonlight.py \\
        --model_path /home/ubuntu/environment/models/Moonlight-16B-A3B \\
        --compiled_model_path /tmp/moonlight_optimum_compiled \\
        --tp_degree 2

To force recompilation (e.g. after code changes):
    python interactive_moonlight.py ... --recompile
"""

import argparse
import glob
import os
import readline  # noqa: F401 — enables line editing in input()
import shutil
import sys
import time

import torch
from transformers import AutoTokenizer

from optimum.neuron import NeuronModelForCausalLM


def export_model(args):
    """Compile (export) the model to Neuron format and save it."""
    print(f"Exporting model from {args.model_path} (tp={args.tp_degree}, seq={args.sequence_length})...")
    print("This will take a while (30-60 minutes for first compilation)...")
    t0 = time.time()

    neuron_config = NeuronModelForCausalLM.get_neuron_config(
        model_name_or_path=args.model_path,
        batch_size=1,
        sequence_length=args.sequence_length,
        tensor_parallel_size=args.tp_degree,
    )

    model = NeuronModelForCausalLM.export(
        model_id=args.model_path,
        neuron_config=neuron_config,
        load_weights=True,
        trust_remote_code=True,
    )

    print(f"  Export + load completed in {time.time() - t0:.1f}s")
    print(f"  Saving compiled model to {args.compiled_model_path}...")
    model.save_pretrained(args.compiled_model_path)
    # Copy custom code files (configuration_deepseek.py, etc.) needed by auto_map
    for py_file in glob.glob(os.path.join(args.model_path, "*.py")):
        shutil.copy2(py_file, args.compiled_model_path)
    print(f"  Saved.")
    return model


def load_model(args):
    """Load model — export if needed, otherwise load from compiled cache."""
    compiled_config = os.path.join(args.compiled_model_path, "config.json")
    need_export = args.recompile or not os.path.isfile(compiled_config)

    if need_export:
        return export_model(args)

    print(f"Loading compiled model from {args.compiled_model_path}...")
    t0 = time.time()
    model = NeuronModelForCausalLM.from_pretrained(
        args.compiled_model_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model


def repl(model, tokenizer, args):
    """Interactive read-eval-print loop."""
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_p = args.top_p
    do_sample = temperature > 0

    stop_ids = {tokenizer.eos_token_id}
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id != tokenizer.unk_token_id:
        stop_ids.add(im_end_id)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"\nMoonlight 16B Interactive Shell (Optimum Neuron)")
    print(f"  sequence_length={args.sequence_length}, max_new_tokens={max_new_tokens}")
    print(f"  temperature={temperature}, top_p={top_p}")
    print(f"\n  Commands:")
    print(f"    /quit              — exit")
    print(f"    /tokens N          — set max_new_tokens")
    print(f"    /temp F            — set temperature (0=greedy)")
    print(f"    /top_p F           — set top-p")
    print(f"    /settings          — show current settings")
    print()

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue

        if prompt in ("/quit", "/exit"):
            print("Bye!")
            break

        if prompt == "/settings":
            print(f"  max_new_tokens = {max_new_tokens}")
            print(f"  temperature    = {temperature}")
            print(f"  top_p          = {top_p}")
            continue

        if prompt.startswith("/tokens "):
            try:
                max_new_tokens = int(prompt.split()[1])
                print(f"  max_new_tokens = {max_new_tokens}")
            except (IndexError, ValueError):
                print("  Usage: /tokens N")
            continue

        if prompt.startswith("/temp "):
            try:
                temperature = float(prompt.split()[1])
                do_sample = temperature > 0
                print(f"  temperature = {temperature} ({'sampling' if do_sample else 'greedy'})")
            except (IndexError, ValueError):
                print("  Usage: /temp F")
            continue

        if prompt.startswith("/top_p "):
            try:
                top_p = float(prompt.split()[1])
                print(f"  top_p = {top_p}")
            except (IndexError, ValueError):
                print("  Usage: /top_p F")
            continue

        if prompt.startswith("/"):
            print(f"  Unknown command: {prompt.split()[0]}")
            continue

        # Tokenize
        tokens = tokenizer(prompt, return_tensors="pt", padding=True)

        # Generate
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = 50

        t0 = time.time()
        with torch.inference_mode():
            output_ids = model.generate(**tokens, **gen_kwargs)
        elapsed = time.time() - t0

        # Decode only the generated tokens (skip the prompt)
        prompt_len = tokens["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_len:]
        # Filter stop tokens
        filtered = [t.item() for t in generated_ids if t.item() not in stop_ids]
        text = tokenizer.decode(filtered)

        print(text)
        num_tokens = len(generated_ids)
        ms_per_tok = (elapsed * 1000) / num_tokens if num_tokens > 0 else 0
        print(f"  [{num_tokens} tokens, {elapsed:.1f}s, {ms_per_tok:.1f}ms/tok]")


def main():
    parser = argparse.ArgumentParser(description="Moonlight 16B Interactive Shell (Optimum Neuron)")
    parser.add_argument("--model_path", type=str,
                        default="/home/ubuntu/environment/models/Moonlight-16B-A3B",
                        help="Path to HuggingFace Moonlight-16B-A3B checkpoint")
    parser.add_argument("--compiled_model_path", type=str,
                        default="/tmp/moonlight_optimum_compiled",
                        help="Path to save/load compiled model")
    parser.add_argument("--tp_degree", type=int, default=2,
                        help="Tensor parallel degree (number of NeuronCores)")
    parser.add_argument("--sequence_length", type=int, default=4096,
                        help="Max sequence length (prompt + generation)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Default max new tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0=greedy)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p nucleus sampling")
    parser.add_argument("--recompile", action="store_true",
                        help="Force recompilation even if compiled model exists")
    args = parser.parse_args()

    model = load_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    repl(model, tokenizer, args)


if __name__ == "__main__":
    main()
