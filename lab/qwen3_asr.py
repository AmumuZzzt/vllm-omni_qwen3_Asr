# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-ASR: a single-stage speech-to-text pipeline.

Qwen3-ASR is a native vLLM model (``Qwen3ASRForConditionalGeneration``) that
implements ``SupportsTranscription``. Before this pipeline existed, Omni had
no topology that could carry it, so ``supported_tasks`` never contained
``"transcription"`` and ``/v1/audio/transcriptions`` was not mounted.

The ``extras={"supports_transcription": True}`` declaration is what makes the
capability visible: ``omni_engine_base`` folds a stage's ``yaml_extras`` into
the engine task list, which in turn gates the OpenAI transcription routes.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)


QWEN3_ASR_PIPELINE = PipelineConfig(
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
            # Single-stage ASR: this stage *is* the transcription endpoint.
            extras={"supports_transcription": True},
        ),
    ),
)
