"""Qwen3.5-9B (LLM) behind an OpenAI-compatible endpoint on Modal, for the llm baseline in configs/models.yaml.

    pip install modal && modal setup                                  # once
    QWEN_API_KEY=$(openssl rand -hex 24) modal deploy deploy/qwen_vllm_modal.py
    curl -L --max-time 1200 https://<workspace>--qwen-vllm-serve.modal.run/v1/models -H "authorization: Bearer $QWEN_API_KEY"
    modal app stop qwen-vllm --yes                                    # take it down

vLLM serves Qwen/Qwen3.5-9B text-only (--language-model-only) with native tool calling (qwen3_coder parser). Thinking
is switched off per request by the benchmark adapter (chat_template_kwargs.enable_thinking=false). The container scales
to zero after 5 idle minutes; weights are cached on the `qwen-hf-cache` volume so only the first cold start downloads them.
"""
import os
import subprocess

import modal

MODEL = "Qwen/Qwen3.5-9B"
VLLM_VERSION = "0.30.0"
GPU = os.environ.get("QWEN_GPU", "L40S")       # 48 GB: bf16 weights (~19 GB) + KV cache
MAX_MODEL_LEN = 32768                           # benchmark prompts are a few thousand tokens
# The key reaches the container as VLLM_API_KEY (from the secret), so read it under both names: this file is evaluated
# again inside the container and must declare the same secret there, or Modal refuses to start the function.
API_KEY = os.environ.get("QWEN_API_KEY") or os.environ.get("VLLM_API_KEY")
SECRET = {"VLLM_API_KEY": API_KEY} if API_KEY else {}

app = modal.App("qwen-vllm")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
    # The slim image has no CUDA toolkit: FlashInfer's sampler would JIT-compile with nvcc on the first request and the
    # engine dies with "Could not find nvcc". vLLM's own PyTorch sampler needs no compiler.
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
cache = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)


@app.function(image=image, gpu=GPU, volumes={"/hf": cache}, secrets=[modal.Secret.from_dict(SECRET)] if SECRET else [],
              scaledown_window=300, timeout=3600, min_containers=0)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=1800)
def serve():
    cmd = ["vllm", "serve", MODEL, "--host", "0.0.0.0", "--port", "8000", "--served-model-name", "qwen3.5-9b",
           "--max-model-len", str(MAX_MODEL_LEN), "--language-model-only",
           "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
           "--gpu-memory-utilization", "0.90"]
    # vLLM reads VLLM_API_KEY from the environment and then requires "Authorization: Bearer <key>" on /v1/*.
    subprocess.Popen(cmd)
