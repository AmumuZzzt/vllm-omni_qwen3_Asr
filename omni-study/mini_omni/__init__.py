# SPDX-License-Identifier: Apache-2.0
"""手搓版 vllm-omni 学习项目：mini_omni。

模块对照：
  mini_omni.config        ← vllm_omni/config/{stage_config,pipeline_registry,resolver}.py
  mini_omni.orchestrator  ← vllm_omni/engine/{orchestrator,stage_pool,messages}.py
  mini_omni.engine        ← vllm_omni/engine/{omni_engine_base,async_omni_engine}.py
  mini_omni.entrypoints   ← vllm_omni/entrypoints/{async_omni,async_omni_base}.py
"""

from .config import (
    DeployConfig,
    OmniConfigResolution,
    PipelineConfig,
    StageConfig,
    StageExecutionType,
    StagePipelineConfig,
    load_deploy_config,
    merge_pipeline_deploy,
    register_pipeline,
    resolve_omni_config,
    resolve_pipeline_config,
)
from .orchestrator import (
    AbortRequestMessage,
    ErrorMessage,
    Orchestrator,
    OrchestratorRequestState,
    OutputMessage,
    ShutdownRequestMessage,
    StageSubmissionMessage,
)

__all__ = [
    "DeployConfig",
    "OmniConfigResolution",
    "PipelineConfig",
    "StageConfig",
    "StageExecutionType",
    "StagePipelineConfig",
    "load_deploy_config",
    "merge_pipeline_deploy",
    "register_pipeline",
    "resolve_omni_config",
    "resolve_pipeline_config",
    "AbortRequestMessage",
    "ErrorMessage",
    "Orchestrator",
    "OrchestratorRequestState",
    "OutputMessage",
    "ShutdownRequestMessage",
    "StageSubmissionMessage",
]
