# Source me before running vLLM experiments on this host.
#   source /root/vllm-work/env.sh
#
# Why each line exists (see lab2/precheck notes):
#  - ninja lives in the venv's bin/, so a bare `python` subprocess cannot find it
#    and FlashInfer's sampler JIT dies with FileNotFoundError: 'ninja'.
#  - CUDA_HOME must point at the toolkit or FlashInfer cannot compile kernels.
#  - The sampling module takes ~71 s to JIT-build on first use, then ~0 ms from
#    cache. A benchmark that starts before the cache is warm silently eats that
#    cost, which is what produced the 10x cold/warm gap in the old
#    /root/vllm-learning results.

export VLLM_VENV=/opt/qwen3-asr/venv
export CUDA_HOME=/usr/local/cuda
export PATH="$VLLM_VENV/bin:$PATH"

# Keep HuggingFace/ModelScope behaviour consistent with the host default.
export USE_MODELSCOPE_HUB=${USE_MODELSCOPE_HUB:-1}

# Deterministic sampler choice for benchmarking:
#   1 = FlashInfer (fast path; REQUIRES a warm JIT cache)
#   0 = native torch sampler (slower but zero JIT dependency)
# Set to 0 only for parity with the old pre-ninja numbers.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-1}

_omni_env_note() {
  echo "vllm env: venv=$VLLM_VENV CUDA_HOME=$CUDA_HOME flashinfer_sampler=$VLLM_USE_FLASHINFER_SAMPLER"
  command -v ninja >/dev/null && echo "  ninja: $(ninja --version)" || echo "  ninja: MISSING"
  command -v nvcc >/dev/null && echo "  nvcc : $(nvcc --version | tail -1)" || echo "  nvcc : MISSING"
}
_omni_env_note
