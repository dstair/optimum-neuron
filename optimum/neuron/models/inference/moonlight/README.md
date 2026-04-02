# Moonlight 16B-A3B on AWS Trainium

Inference support for [MoonshotAI/Moonlight-16B-A3B](https://huggingface.co/moonshotai/Moonlight-16B-A3B)
on AWS Trainium via [optimum-neuron](https://huggingface.co/docs/optimum-neuron).

## Background

Moonlight 16B-A3B is a 16-billion parameter language model using the DeepSeek V3 architecture.
It activates only 3B parameters per token through sparse Mixture-of-Experts (MoE) routing,
achieving strong performance with efficient inference.

### Architecture highlights

| Feature | Details |
|---------|---------|
| **Layers** | 27 total: 1 dense (layer 0) + 26 MoE layers |
| **Attention** | Multi-head Latent Attention (MLA) with KV compression (`kv_lora_rank=512`) |
| **MoE** | 64 routed experts + 2 shared experts, top-6 per token, sigmoid routing |
| **Active params** | 3B per token (of 16B total) |
| **RoPE** | Standard rotary embeddings, `rope_theta=50000`, 8K context |
| **Vocabulary** | 163,840 tokens |

**MLA** compresses the key-value cache using low-rank projections, reducing memory
from `num_heads * head_dim` to a single compressed vector of dimension 576
(`rope_dim=64 + kv_lora_rank=512`). This enables longer sequences with less memory.

**MoE routing** uses sigmoid-based `noaux_tc` routing with `e_score_correction_bias` —
a learned per-expert bias that adjusts expert selection without affecting routing weights.

## Usage

### Direct inference with optimum-neuron

```python
from optimum.neuron import NeuronModelForCausalLM
from transformers import AutoTokenizer

model_id = "moonshotai/Moonlight-16B-A3B"

# Configure and compile for Neuron
nc = NeuronModelForCausalLM.get_neuron_config(
    model_name_or_path=model_id,
    batch_size=1,
    sequence_length=4096,
    tensor_parallel_size=2,
)
model = NeuronModelForCausalLM.export(
    model_id, neuron_config=nc, load_weights=True, trust_remote_code=True
)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
inputs = tokenizer("The capital of France is", return_tensors="pt")
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

First run compiles the model (~5 min on trn2.3xlarge). Subsequent loads from
a saved compiled model are fast (~80s).

### Save and reload compiled model

```python
# Save after compilation
model.save_pretrained("/tmp/moonlight_compiled")

# Reload without recompilation
model = NeuronModelForCausalLM.from_pretrained("/tmp/moonlight_compiled")
```

### Serving with vLLM

Moonlight can be served via vLLM using the optimum-neuron plugin:

```bash
VLLM_PLUGINS=optimum_neuron \
vllm serve moonshotai/Moonlight-16B-A3B \
  --served_model_name Moonlight-16B-A3B \
  --port 8080 \
  --tensor-parallel-size 2 \
  --max-num-seqs 1 \
  --max-model-len 512 \
  --dtype bfloat16 \
  --trust-remote-code \
  --model-loader-extra-config allow_non_cached_model \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill
```

For longer sequences, use tp=4 with LNC=1:

```bash
NEURON_RT_LOGICAL_NC_PER_DEVICE=1 \
VLLM_PLUGINS=optimum_neuron \
vllm serve moonshotai/Moonlight-16B-A3B \
  --served_model_name Moonlight-16B-A3B \
  --port 8080 \
  --tensor-parallel-size 4 \
  --max-num-seqs 1 \
  --max-model-len 1024 \
  --dtype bfloat16 \
  --trust-remote-code \
  --model-loader-extra-config allow_non_cached_model \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill
```

Then query the server:

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Moonlight-16B-A3B",
    "prompt": "The capital of France is",
    "max_tokens": 50,
    "temperature": 0
  }'
```

Chat completions and streaming are also supported:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Moonlight-16B-A3B",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50,
    "temperature": 0
  }'
```

## Benchmarks

All benchmarks measured on trn2.3xlarge (2 NeuronCores, tp=2, batch_size=1, BF16).

### Direct inference

| Metric | Value |
|--------|-------|
| First token (context encoding) | ~176 ms |
| Per-token decode | ~17 ms |
| Compilation time | ~5 min |
| Compiled model reload | ~80 s |

### vLLM serving (GuideLLM, 20 requests, ~200 prompt tokens, ~50 output tokens)

| Metric | Median | p95 |
|--------|--------|-----|
| Request latency | 1.0 s | 1.0 s |
| Time-to-first-token (TTFT) | 175.8 ms | 178.1 ms |
| Inter-token latency (ITL) | 16.6 ms | 16.7 ms |
| Time per output token (TPOT) | 19.8 ms | 19.9 ms |
| Output tokens/sec | 50.5 tok/s | -- |

### Numerical accuracy (FP32, teacher-forced logit comparison)

HF FP32 (CPU) vs Neuron FP32 (on-device), 1024 positions x 3 prompts:

| Prompt | Max logit diff | Abs mean diff | Top-1 match |
|--------|---------------|---------------|-------------|
| "The capital of France is" | 0.0432 | 0.0003 | 1024/1024 (100%) |
| "1 + 1 =" | 0.1688 | 0.0017 | 1024/1024 (100%) |
| "The largest planet in the solar system is" | 0.0731 | 0.0005 | 1024/1024 (100%) |

In FP32, the Neuron implementation matches HF across all 3072 positions with 100% top-1
agreement. Max logit diff grows with sequence length due to floating-point accumulation
but never changes the top prediction.

## Tests

Run from the `optimum-neuron` root:

```bash
# CPU smoke tests (no hardware needed)
MOONLIGHT_MODEL_PATH=moonshotai/Moonlight-16B-A3B \
  pytest optimum/neuron/models/inference/moonlight/tests/test_moonlight_smoke.py -v

# On-device tests (requires trn2 + model weights)
MOONLIGHT_MODEL_PATH=/path/to/Moonlight-16B-A3B \
  pytest optimum/neuron/models/inference/moonlight/tests/test_moonlight_on_device.py -v -s

# Export/save/reload cycle (requires trn2, ~16 min)
MOONLIGHT_MODEL_PATH=/path/to/Moonlight-16B-A3B \
  pytest optimum/neuron/models/inference/moonlight/tests/test_moonlight_export.py -v -s

# FP32 logit divergence (requires trn2, ~8 min for 32 tokens)
MOONLIGHT_MODEL_PATH=/path/to/Moonlight-16B-A3B \
  pytest optimum/neuron/models/inference/moonlight/tests/test_moonlight_logit_divergence.py -v -s
```

## Hardware requirements

| Configuration | Minimum hardware |
|---------------|-----------------|
| BF16, tp=2 | trn2.3xlarge (2 NeuronCores) |
| FP32, tp=4 | trn2.3xlarge with `NEURON_RT_LOGICAL_NC_PER_DEVICE=1` |

## Known limitations

- **vLLM max sequence length** depends on the TP configuration:
  - `tp=2` (LNC=2, default): `--max-model-len 512` (compiler errors at seq_len >= 1024)
  - `tp=4` (LNC=1, via `NEURON_RT_LOGICAL_NC_PER_DEVICE=1`): `--max-model-len 1024` confirmed working
  - Direct inference supports longer sequences (tested at seq_len=4096)
