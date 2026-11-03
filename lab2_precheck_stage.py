#!/usr/bin/env python
"""Lab 2 pre-check: can Qwen3-ASR be driven as a vLLM-Omni stage?

Registers an out-of-tree ``qwen3_asr`` single-stage pipeline (the documented
``register_pipeline`` extension point -- site-packages is NOT modified), then
asks AsyncOmni to bring it up and report what the capability probe produced.

Interpretation:
  * engine starts        -> model is stage-compatible; Lab 2 = registration + probe
  * engine fails         -> the traceback names the missing adapter layer
"""

import json
import os
import traceback

MODEL = "/opt/qwen3-asr/models/Qwen3-ASR-0.6B"


def main() -> int:
    from vllm_omni.config.pipeline_registry import register_pipeline
    from vllm_omni.config.stage_config import (
        PipelineConfig,
        StageExecutionType,
        StagePipelineConfig,
    )

    qwen3_asr_pipeline = PipelineConfig(
        model_type="qwen3_asr",
        model_arch="Qwen3ASRForConditionalGeneration",
        hf_architectures=("Qwen3ASRForConditionalGeneration",),
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="qwen3_asr",
                execution_type=StageExecutionType.LLM_AR,
                input_sources=(),
                final_output=True,
                final_output_type="text",
                owns_tokenizer=True,
                requires_multimodal_data=True,
                engine_output_type="text",
                model_arch="Qwen3ASRForConditionalGeneration",
                sampling_constraints={"detokenize": True},
            ),
        ),
    )
    register_pipeline(qwen3_asr_pipeline)
    print("[precheck] registered pipeline 'qwen3_asr'", flush=True)

    from vllm_omni.entrypoints.async_omni import AsyncOmni

    try:
        engine = AsyncOmni(
            model=MODEL,
            trust_remote_code=True,
            init_timeout=300,
            stage_init_timeout=300,
        )
    except Exception:
        print("[precheck] RESULT: engine construction FAILED", flush=True)
        traceback.print_exc()
        return 2

    try:
        tasks = engine.engine.supported_tasks
        stage_cfgs = engine.engine.stage_configs
        stage_meta = engine.engine.stage_metadata
        print("[precheck] RESULT: engine constructed OK", flush=True)
        print(f"[precheck] supported_tasks      = {tasks}", flush=True)
        print(f"[precheck] num_stages           = {len(stage_cfgs)}", flush=True)
        for i, sc in enumerate(stage_cfgs):
            print(
                f"[precheck] stage{i}: type={sc.stage_type} "
                f"is_comprehension={getattr(sc, 'is_comprehension', '?')} "
                f"final_output={getattr(sc, 'final_output', '?')} "
                f"final_output_type={getattr(sc, 'final_output_type', '?')}",
                flush=True,
            )
        for i, m in enumerate(stage_meta):
            print(f"[precheck] meta{i}: {m}", flush=True)
        print(
            "[precheck] 'transcription' advertised = "
            f"{'transcription' in tasks}",
            flush=True,
        )
        print(json.dumps({"supported_tasks": list(tasks)}), flush=True)
    finally:
        try:
            engine.shutdown()
            print("[precheck] engine shut down", flush=True)
        except Exception:
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
