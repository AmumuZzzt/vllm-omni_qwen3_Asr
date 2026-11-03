# SPDX-License-Identifier: Apache-2.0
"""手搓版 vllm_omni 配置层（config layer）。

对照源码：
  - vllm_omni/config/stage_config.py    (StagePipelineConfig / PipelineConfig /
                                          DeployConfig / merge_pipeline_deploy)
  - vllm_omni/config/pipeline_registry.py (OMNI_PIPELINES / register_pipeline)
  - vllm_omni/config/resolver.py         (resolve_omni_config / OmniConfigResolution)

本文件只保留"拓扑与部署合并"这条主干，去掉了 vLLM/transformers/OmegaConf 依赖，
使得它可以在纯 Python 环境下阅读、运行和测试。
"""

from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

import yaml

__all__ = [
    "StageExecutionType",
    "StagePipelineConfig",
    "PipelineConfig",
    "StageDeployConfig",
    "DeployConfig",
    "StageConfig",
    "OmniConfigResolution",
    "OMNI_PIPELINES",
    "register_pipeline",
    "resolve_pipeline_config",
    "pipeline_cfg_resolver",
    "resolve_deploy_yaml",
    "load_deploy_config",
    "merge_pipeline_deploy",
    "resolve_omni_config",
    "build_stage_runtime_overrides",
    "normalize_pipeline_cli_overrides",
]


# --------------------------------------------------------------------------
# 拓扑（PipelineConfig 一侧）—— 对应 stage_config.py:236-535
# --------------------------------------------------------------------------


class StageExecutionType(str, Enum):
    """一个 stage 用哪类执行引擎跑（源码里合并了旧的 StageType + WorkerType）。"""

    LLM_AR = "llm_ar"  # 自回归解码（vLLM scheduler + KV cache）
    LLM_GENERATION = "llm_generation"  # 非自回归生成（如 Code2Wav 声码器）
    DIFFUSION = "diffusion"  # 扩散去噪


@dataclass(frozen=True)
class StagePipelineConfig:
    """一个 stage 的固定拓扑（frozen，用户不可改）。源码 stage_config.py:349。"""

    stage_id: int
    model_stage: str
    execution_type: StageExecutionType = StageExecutionType.LLM_AR
    input_sources: tuple[int, ...] = ()
    final_output: bool = False
    final_output_type: str | None = None
    owns_tokenizer: bool = False
    requires_multimodal_data: bool = False
    hf_config_name: str | None = None
    engine_output_type: str | None = None
    model_arch: str | None = None
    sampling_constraints: dict[str, Any] = field(default_factory=dict)
    # 跨 stage 输入处理函数（点分路径字符串，运行时 import）。
    custom_process_input_func: str | None = None
    custom_process_next_stage_input_func: str | None = None
    async_chunk_process_next_stage_input_func: str | None = None
    sync_process_input_func: str | None = None
    requires_full_payload_input: bool = False
    recompute_preemption: Literal["allow", "fail"] = "allow"
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    """一个模型的完整流水线拓扑（frozen）。源码 stage_config.py:441。"""

    model_type: str
    model_arch: str = ""
    stages: tuple[StagePipelineConfig, ...] = ()
    single_stage_async_chunk: bool = False
    hf_architectures: tuple[str, ...] = ()
    hf_config_predicate: Callable[[Any], bool] | None = None
    default_deploy_config_name: str | None = None
    # 全局 CLI 拼写 -> (stage id, stage 局部拼写)
    stage_cli_aliases: dict[str, tuple[int, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        errors = self.get_validation_errors()
        if errors:
            raise ValueError(
                "PipelineConfig initialization failed with the following error(s):\n\t"
                + "\n\t".join(errors)
            )

    def get_stage(self, stage_id: int) -> StagePipelineConfig | None:
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        return None

    def get_validation_errors(self) -> list[str]:
        """拓扑校验。对应源码 stage_config.py:512-535，逐条保持一致。"""
        errors: list[str] = []
        if not self.stages:
            errors.append("Pipeline has no stages defined")
            return errors
        stage_ids = [s.stage_id for s in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            errors.append("Duplicate stage IDs found")
        stage_id_set = set(stage_ids)
        for stage in self.stages:
            for src in stage.input_sources:
                if src not in stage_id_set:
                    errors.append(f"Stage {stage.stage_id} references non-existent input source {src}")
                if src == stage.stage_id:
                    errors.append(f"Stage {stage.stage_id} references itself")
        # 必须存在入口 stage（input_sources 为空）
        if not any(not s.input_sources for s in self.stages):
            errors.append("No entry point (stage with empty input_sources)")
        # 必须存在终点 stage（final_output=True），否则请求永远无法收尾
        if not any(s.final_output for s in self.stages):
            errors.append("No terminal stage (stage with final_output=True)")
        return errors


# --------------------------------------------------------------------------
# 注册表 —— 对应 config/pipeline_registry.py
# --------------------------------------------------------------------------

PipelineResolverFunc = Callable[[Any], "PipelineConfig | None"]


def pipeline_cfg_resolver(config_type: type):
    """装饰器：hf_config 类型不匹配时 resolver 返回 None（源码 stage_config.py:84）。"""

    def builder(func: PipelineResolverFunc) -> PipelineResolverFunc:
        def wrapper(hf_config: Any) -> PipelineConfig | None:
            if hf_config is None or not isinstance(hf_config, config_type):
                return None
            return func(hf_config)

        wrapper.__name__ = getattr(func, "__name__", "resolver")
        return wrapper

    return builder


# model_type -> PipelineConfig 或 resolver 可调用对象
OMNI_PIPELINES: dict[str, "PipelineConfig | PipelineResolverFunc"] = {}


def register_pipeline(pipeline: "PipelineConfig | PipelineResolverFunc", model_type: str | None = None):
    """注册树外 pipeline。对应 pipeline_registry.py:249。"""
    if isinstance(pipeline, PipelineConfig):
        model_type = model_type if model_type is not None else pipeline.model_type
    elif model_type is None:
        raise ValueError("Model type must be explicitly provided when registering a pipeline resolver")
    OMNI_PIPELINES[model_type] = pipeline


def resolve_pipeline_config(model_type: str, hf_config: Any = None) -> PipelineConfig | None:
    """按 model_type 解析出具体 PipelineConfig。对应 pipeline_registry.py:266。"""
    if model_type not in OMNI_PIPELINES:
        return None
    pipeline = OMNI_PIPELINES[model_type]
    return pipeline(hf_config) if callable(pipeline) else pipeline


# --------------------------------------------------------------------------
# 部署（DeployConfig 一侧）—— 对应 stage_config.py:538-1139
# --------------------------------------------------------------------------

# 这些 key 不进 engine 参数，而是 stage 运行时摆放（reserved）
_STAGE_RESERVED_KEYS = frozenset(
    {
        "async_chunk",
        "model_runner",
        "stage_id",
        "devices",
        "num_replicas",
        "env",
        "output_connectors",
        "input_connectors",
        "default_sampling_params",
        "default_pooling_params",
        "engine_extras",
        "engine_args",
        "runtime",
    }
)

# 顶层 pipeline-wide 字段，会均匀下放到每个 stage
PIPELINE_WIDE_ENGINE_FIELDS: tuple[str, ...] = (
    "trust_remote_code",
    "distributed_executor_backend",
    "dtype",
    "quantization",
    "enable_prefix_caching",
    "enable_chunked_prefill",
    "data_parallel_size",
    "pipeline_parallel_size",
    "custom_voice_dir",
)

# 深合并 key：薄 overlay 不应丢掉 base 里的兄弟 key
_DEEP_MERGE_KEYS = frozenset(
    {
        "default_sampling_params",
        "default_pooling_params",
        "subtalker_sampling_params",
        "engine_extras",
        "engine_args",
    }
)
_DEPLOY_DEEP_MERGE_KEYS = frozenset({"speech_cache"})


@dataclass
class StageDeployConfig:
    """每个 stage 的部署旋钮。源码 stage_config.py:538（此处只保留常用字段子集）。"""

    stage_id: int
    devices: str | None = None
    num_replicas: int = 1
    env: dict[str, Any] | None = None

    async_chunk: bool | None = None
    model_runner: Literal["v1", "v2"] | None = None

    output_connectors: dict[str, str] | None = None
    input_connectors: dict[str, str] | None = None
    default_sampling_params: dict[str, Any] | None = None
    default_pooling_params: dict[str, Any] | None = None

    # 常见 stage 引擎参数（真实源码里还有几十个字段，此处按 deploy yaml 常用项收录）
    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    max_model_len: int | None = None
    attention_backend: str | None = None
    moe_backend: str | None = None
    enable_prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    enforce_eager: bool | None = None
    async_scheduling: bool | None = None
    trust_remote_code: bool | None = None
    compilation_config: dict[str, Any] | None = None
    hf_overrides: dict[str, Any] | None = None

    # 未收录字段统一进 engine_extras（与源码 _parse_stage_deploy 行为一致）
    engine_extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeployConfig:
    """deploy/<model>.yaml —— 用户唯一需要编辑的配置文件。源码 stage_config.py:838。"""

    async_chunk: bool | None = None
    session_mode: str = "turn"
    model_runner: Literal["v1", "v2"] = "v1"
    connectors: dict[str, Any] | None = None
    stages: list[StageDeployConfig] = field(default_factory=list)
    platforms: dict[str, Any] | None = None
    pipeline: str | None = None

    # pipeline-wide 引擎设置（字段集与 PIPELINE_WIDE_ENGINE_FIELDS 对齐）
    trust_remote_code: bool | None = None
    distributed_executor_backend: str | None = None
    dtype: str | None = None
    quantization: str | None = None
    enable_prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    data_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    custom_voice_dir: str | None = None


def _get_recursively_merged_dict(original: dict, update: dict) -> dict:
    """递归合并两个 dict，返回新 dict。对应 stage_config.py:999。"""
    merged = original.copy()
    for k, update_v in update.items():
        orig_v = merged.get(k)
        if isinstance(orig_v, dict) and isinstance(update_v, dict):
            merged[k] = _get_recursively_merged_dict(orig_v, update_v)
        else:
            merged[k] = update_v
    return merged


def _merge_config_fields(base: dict, overlay: dict, *, deep_merge_keys: frozenset[str]) -> dict:
    """选定字段递归合并，其余字段 overlay 直接替换。对应 stage_config.py:986。"""
    base_nested = {k: v for k, v in base.items() if k in deep_merge_keys}
    overlay_nested = {k: v for k, v in overlay.items() if k in deep_merge_keys}
    merged_nested = _get_recursively_merged_dict(base_nested, overlay_nested)
    return {**base, **overlay, **merged_nested}


def _deep_merge_stage(base: dict, overlay: dict) -> dict:
    return _merge_config_fields(base, overlay, deep_merge_keys=_DEEP_MERGE_KEYS)


def _merge_stage_lists(
    base_stages: list[dict[str, Any]] | None,
    overlay_stages: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """按 stage_id 合并两份 stages 列表（overlay 字段胜出）。对应 stage_config.py:1020。"""
    by_id: dict[int, dict[str, Any]] = {s["stage_id"]: s for s in (base_stages or [])}
    for overlay_stage in overlay_stages or []:
        sid = overlay_stage["stage_id"]
        if sid in by_id:
            by_id[sid] = _deep_merge_stage(by_id[sid], overlay_stage)
        else:
            by_id[sid] = overlay_stage
    return list(by_id.values())


def _merge_connectors(base: dict | None, overlay: dict | None) -> dict | None:
    if base is None and overlay is None:
        return None
    return _get_recursively_merged_dict(base or {}, overlay or {})


def _merge_platforms(base: dict | None, overlay: dict | None) -> dict | None:
    if base is None and overlay is None:
        return None
    merged = _get_recursively_merged_dict(base or {}, overlay or {})
    # platforms.<plat>.stages 同样按 stage_id 合并
    for plat in set(base or {}) | set(overlay or {}):
        b_stages = (base or {}).get(plat, {}).get("stages")
        o_stages = (overlay or {}).get(plat, {}).get("stages")
        if b_stages is not None or o_stages is not None:
            merged.setdefault(plat, {})
            merged[plat]["stages"] = _merge_stage_lists(b_stages, o_stages)
    return merged


def resolve_deploy_yaml(path: str | Path) -> dict[str, Any]:
    """加载 deploy YAML，支持 ``base_config`` 继承。对应 stage_config.py:1064。"""
    with open(path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    base_path = raw_dict.pop("base_config", None)
    if base_path is None:
        return raw_dict

    # 相对 overlay 文件所在目录解析
    base_path = Path(path).parent / base_path
    base_dict = resolve_deploy_yaml(base_path)

    # 顶层标量：overlay 胜出；结构化 section 单独合并，避免薄 overlay 丢契约
    merged = _merge_config_fields(
        base_dict,
        {k: v for k, v in raw_dict.items() if k not in ("connectors", "stages", "platforms")},
        deep_merge_keys=_DEPLOY_DEEP_MERGE_KEYS,
    )
    merged_connectors = _merge_connectors(base_dict.get("connectors"), raw_dict.get("connectors"))
    if merged_connectors is not None:
        merged["connectors"] = merged_connectors
    merged["stages"] = _merge_stage_lists(base_dict.get("stages"), raw_dict.get("stages"))
    merged_platforms = _merge_platforms(base_dict.get("platforms"), raw_dict.get("platforms"))
    if merged_platforms is not None:
        merged["platforms"] = merged_platforms
    return merged


def _parse_stage_deploy(stage_data: dict[str, Any]) -> StageDeployConfig:
    """把 deploy YAML 的一条 stage 记录解析成 StageDeployConfig。对应 stage_config.py:930。"""
    flat_args = {k: v for k, v in stage_data.items() if k not in _STAGE_RESERVED_KEYS}
    explicit_engine_extras = dict(stage_data.get("engine_extras") or {})
    runtime_cfg = dict(stage_data.get("runtime", {}))
    devices = runtime_cfg.get("devices", stage_data.get("devices"))
    num_replicas = runtime_cfg.get("num_replicas", stage_data.get("num_replicas", 1))
    env = runtime_cfg.get("env", stage_data.get("env"))

    if "engine_args" in stage_data:
        for k, v in stage_data["engine_args"].items():
            existing = flat_args.get(k)
            if isinstance(v, dict) and isinstance(existing, dict):
                flat_args[k] = _get_recursively_merged_dict(existing, v)
            else:
                flat_args[k] = v

    stage_runner = stage_data.get("model_runner")
    if stage_runner is not None and stage_runner not in ("v1", "v2"):
        raise ValueError(f"stage {stage_data['stage_id']}: model_runner must be 'v1' or 'v2', got {stage_runner!r}")

    kwargs: dict[str, Any] = {
        "stage_id": stage_data["stage_id"],
        "devices": devices,
        "num_replicas": int(num_replicas),
        "env": env,
        "async_chunk": stage_data.get("async_chunk"),
        "model_runner": stage_runner,
        "output_connectors": stage_data.get("output_connectors"),
        "input_connectors": stage_data.get("input_connectors"),
        "default_sampling_params": stage_data.get("default_sampling_params"),
        "default_pooling_params": stage_data.get("default_pooling_params"),
    }
    known = {f.name for f in fields(StageDeployConfig)} - _STAGE_RESERVED_KEYS - {"engine_extras"}
    for name in list(flat_args):
        if name in known:
            kwargs[name] = flat_args.pop(name)
    kwargs["engine_extras"] = _get_recursively_merged_dict(explicit_engine_extras, flat_args)
    return StageDeployConfig(**kwargs)


def load_deploy_config(path: str | Path) -> DeployConfig:
    """加载 deploy YAML（含 base_config 继承）。对应 stage_config.py:1094。"""
    raw_dict = resolve_deploy_yaml(path)
    if "stage_args" in raw_dict:
        raise ValueError(
            f"Deploy config {path} uses the removed `stage_args` schema; "
            "define topology in PipelineConfig and deployment overrides under `stages`."
        )

    stages = [_parse_stage_deploy(s) for s in raw_dict.get("stages", [])]

    model_runner = raw_dict.get("model_runner", "v1")
    if model_runner not in ("v1", "v2"):
        raise ValueError(f"model_runner must be one of ('v1', 'v2'), got {model_runner!r}")

    kwargs: dict[str, Any] = {
        "async_chunk": raw_dict.get("async_chunk"),
        "session_mode": raw_dict.get("session_mode", "turn"),
        "model_runner": model_runner,
        "connectors": raw_dict.get("connectors", None),
        "stages": stages,
        "platforms": raw_dict.get("platforms", None),
        "pipeline": raw_dict.get("pipeline", None),
    }
    for name in PIPELINE_WIDE_ENGINE_FIELDS:
        if name in raw_dict:
            kwargs[name] = raw_dict[name]
    return DeployConfig(**kwargs)


def _extract_platform_overrides(ps: dict[str, Any]) -> tuple[dict[str, Any], str | None, dict | None]:
    """取平台覆盖里的 (overrides, devices, env)。对应 stage_config.py:1148。"""
    if "engine_args" in ps:
        overrides = dict(ps["engine_args"])
        runtime_cfg = ps.get("runtime", {})
        if "num_replicas" in runtime_cfg:
            overrides["num_replicas"] = runtime_cfg["num_replicas"]
        return overrides, runtime_cfg.get("devices"), runtime_cfg.get("env")
    overrides = {k: v for k, v in ps.items() if k not in ("stage_id", "devices", "env")}
    return overrides, ps.get("devices"), ps.get("env")


def _apply_platform_overrides(deploy: DeployConfig, platform: str) -> DeployConfig:
    """把 platforms.<platform> 下的覆盖合并进 deploy。对应 stage_config.py:1172。"""
    platform_section = (deploy.platforms or {}).get(platform)
    if platform_section is None:
        return deploy
    if "model_runner" in platform_section:
        model_runner = platform_section["model_runner"]
        if model_runner not in ("v1", "v2"):
            raise ValueError(f"platform model_runner must be one of ('v1', 'v2'), got {model_runner!r}")
        deploy.model_runner = model_runner
        if model_runner == "v1":
            # 平台的 V1 回退也覆盖那些显式选了 V2 的 stage
            for stage in deploy.stages:
                if stage.model_runner == "v2":
                    stage.model_runner = None

    base_by_id = {s.stage_id: s for s in deploy.stages}
    for ps in platform_section.get("stages", []):
        base = base_by_id.get(ps["stage_id"])
        if base is None:
            continue
        overrides, devices, env = _extract_platform_overrides(ps)
        if devices is not None:
            base.devices = devices
        if env is not None:
            if isinstance(base.env, dict) and isinstance(env, dict):
                base.env = {**base.env, **env}
            else:
                base.env = env
        for key, val in overrides.items():
            if hasattr(base, key):
                if key in _DEEP_MERGE_KEYS and isinstance(val, dict):
                    base_val = getattr(base, key, None)
                    if isinstance(base_val, dict):
                        setattr(base, key, {**base_val, **val})
                        continue
                setattr(base, key, val)
            else:
                base.engine_extras[key] = val

    for stage in deploy.stages:
        if stage.model_runner is not None and stage.model_runner not in ("v1", "v2"):
            raise ValueError(f"stage {stage.stage_id}: model_runner must be 'v1' or 'v2', got {stage.model_runner!r}")
    return deploy


# --------------------------------------------------------------------------
# async_chunk 决策 —— 对应 stage_config.py:1294-1523
# --------------------------------------------------------------------------


def resolve_stage_async_chunk(deploy: DeployConfig, stage: StageDeployConfig | None) -> bool:
    """deploy.async_chunk AND (stage 未显式关闭)。对应 stage_config.py:1294。"""
    if not isinstance(deploy.async_chunk, bool):
        raise ValueError("async_chunk must be a boolean")
    if stage is not None and stage.async_chunk is not None and not isinstance(stage.async_chunk, bool):
        raise ValueError(f"Stage {stage.stage_id} async_chunk must be a boolean or null")
    return deploy.async_chunk and (stage is None or stage.async_chunk is not False)


def _resolve_pipeline_async_chunk_enabled(pipeline: PipelineConfig, deploy: DeployConfig) -> bool:
    """给定 pipeline + deploy 判定 async_chunk 生效值。对应 stage_config.py:1463。"""
    if len(pipeline.stages) <= 1:
        if pipeline.single_stage_async_chunk:
            if deploy.async_chunk is False:
                raise ValueError(f"Pipeline {pipeline.model_type!r} requires async_chunk=True")
            return True
        if deploy.async_chunk:
            warnings.warn(
                "Deploy config set async_chunk=True, but async chunk is inapplicable "
                "to single stage models. As such, it will be disabled.",
                stacklevel=2,
            )
        return False

    has_inter_stage_edges = any(stage.input_sources for stage in pipeline.stages)
    has_next_stage_inps = any(
        stage.async_chunk_process_next_stage_input_func for stage in pipeline.stages
    )
    has_kv_transfer_cfg = any(
        stage.engine_extras.get("kv_transfer_config") for stage in deploy.stages
    )
    supports_async_chunk = has_inter_stage_edges and has_next_stage_inps

    if deploy.async_chunk is not None:
        if deploy.async_chunk:
            if not supports_async_chunk:
                warnings.warn(
                    "Deploy config set async_chunk=True, but the pipeline config does not "
                    "support it; disabling it.",
                    stacklevel=2,
                )
                return False
            if has_kv_transfer_cfg:
                warnings.warn(
                    "Native AR-to-DiT KV transfer requires async_chunk=False; async chunk "
                    "will be disabled.",
                    stacklevel=2,
                )
                return False
        return deploy.async_chunk

    return supports_async_chunk and not has_kv_transfer_cfg


def validate_stage_async_chunk_edges(pipeline: PipelineConfig, deploy: DeployConfig) -> None:
    """同一条 connector 边两端的 async_chunk 必须一致。对应 stage_config.py:1508。"""
    stages = {stage.stage_id: stage for stage in pipeline.stages}
    deploy_by_id = {stage.stage_id: stage for stage in deploy.stages}
    for consumer in pipeline.stages:
        for source_id in consumer.input_sources:
            producer = stages[source_id]
            if not producer.async_chunk_process_next_stage_input_func:
                continue
            producer_async = resolve_stage_async_chunk(deploy, deploy_by_id.get(source_id))
            consumer_async = resolve_stage_async_chunk(deploy, deploy_by_id.get(consumer.stage_id))
            if producer_async != consumer_async:
                raise ValueError(
                    f"Pipeline {pipeline.model_type!r} has incompatible async_chunk settings on "
                    f"connector edge {source_id} -> {consumer.stage_id}. "
                    "Set the same async_chunk mode on both stages or disable pipeline-wide async_chunk."
                )


def update_deploy_config_async_chunk_enabled(pipeline: PipelineConfig, deploy: DeployConfig) -> None:
    """写入 pipeline-wide async_chunk 并校验边一致性。对应 stage_config.py:1454。"""
    deploy.async_chunk = _resolve_pipeline_async_chunk_enabled(pipeline, deploy)
    validate_stage_async_chunk_edges(pipeline, deploy)


# --------------------------------------------------------------------------
# 合并产物 StageConfig —— 对应 stage_config.py:1526-1715
# --------------------------------------------------------------------------


@dataclass
class StageConfig:
    """合并 pipeline + deploy 后的每 stage 运行配置（源码里的 legacy StageConfig）。"""

    stage_id: int
    model_stage: str
    session_mode: str = "turn"
    stage_type: str = "llm"
    input_sources: list[int] = field(default_factory=list)
    custom_process_input_func: str | None = None
    final_output: bool = False
    final_output_type: str | None = None
    worker_type: str | None = None
    scheduler_cls: str | None = None
    hf_config_name: str | None = None
    sampling_constraints: dict[str, Any] = field(default_factory=dict)
    yaml_engine_args: dict[str, Any] = field(default_factory=dict)
    yaml_runtime: dict[str, Any] = field(default_factory=dict)
    yaml_extras: dict[str, Any] = field(default_factory=dict)


_EXECUTION_TYPE_TO_STAGE_WORKER: dict[StageExecutionType, tuple[str, str | None]] = {
    StageExecutionType.LLM_AR: ("llm", "ar"),
    StageExecutionType.LLM_GENERATION: ("llm", "generation"),
    StageExecutionType.DIFFUSION: ("diffusion", None),
}

# AR stage 两种调度器（源码 stage_config.py:322）
_ARY_SCHEDULERS = {
    True: "vllm_omni.core.sched.omni_ar_scheduler.OmniARAsyncScheduler",
    False: "vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler",
}
_GENERATION_SCHEDULER = "vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler"


def _resolve_scheduler(execution_type: StageExecutionType, async_scheduling: bool = True) -> str | None:
    """按 execution_type 选调度器类路径。对应 stage_config.py:322。"""
    if execution_type == StageExecutionType.LLM_AR:
        return _ARY_SCHEDULERS[bool(async_scheduling)]
    if execution_type == StageExecutionType.LLM_GENERATION:
        return _GENERATION_SCHEDULER
    return None


def _select_processor_funcs(
    ps: StagePipelineConfig,
    async_chunk: bool,
) -> tuple[str | None, str | None]:
    """按 async_chunk 模式挑选 (input_proc, next_stage_proc)。对应 stage_config.py:1302。"""
    next_stage_proc = ps.custom_process_next_stage_input_func
    input_proc = ps.custom_process_input_func
    if async_chunk and ps.async_chunk_process_next_stage_input_func:
        next_stage_proc = ps.async_chunk_process_next_stage_input_func
    elif not async_chunk and ps.sync_process_input_func:
        input_proc = ps.sync_process_input_func
    return input_proc, next_stage_proc


def merge_sampling_constraints(
    sampling_params: dict[str, Any] | None,
    constraints: dict[str, Any],
) -> dict[str, Any]:
    """标量约束覆盖采样参数，stop_token_ids 取并集且不改输入。对应 stage_config.py:1410。"""
    resolved = dict(constraints)
    if "stop_token_ids" in resolved:
        caller = (sampling_params or {}).get("stop_token_ids") or []
        required = resolved["stop_token_ids"] or []
        resolved["stop_token_ids"] = list(dict.fromkeys([*caller, *required]))
    return {**(sampling_params or {}), **resolved}


def _build_engine_args(
    ps: StagePipelineConfig,
    ds: StageDeployConfig | None,
    pipeline: PipelineConfig,
    deploy: DeployConfig,
    next_stage_proc: str | None,
) -> dict[str, Any]:
    """拼一个 stage 的 yaml_engine_args。对应 stage_config.py:1316（精简版）。"""
    engine_args: dict[str, Any] = {
        "model_arch": ps.model_arch or pipeline.model_arch or None,
        "retains_state_across_chunks": False,
    }
    if ps.engine_output_type is not None:
        engine_args["engine_output_type"] = ps.engine_output_type
    if next_stage_proc:
        engine_args["custom_process_next_stage_input_func"] = next_stage_proc

    # pipeline-wide 字段均匀下发
    for name in PIPELINE_WIDE_ENGINE_FIELDS:
        value = getattr(deploy, name)
        if value is not None:
            engine_args[name] = value

    # 每 stage 的 deploy 值覆盖 pipeline-wide
    if ds is not None:
        for k, v in asdict(ds).items():
            if k in _STAGE_RESERVED_KEYS or v is None:
                continue
            engine_args[k] = v
        engine_args.update(ds.engine_extras)

    engine_args["async_chunk"] = resolve_stage_async_chunk(deploy, ds)
    engine_args["session_mode"] = deploy.session_mode
    stage_runner = (ds.model_runner if ds is not None and ds.model_runner else deploy.model_runner)
    engine_args["use_v2_model_runner"] = stage_runner == "v2"
    engine_args["requires_full_payload_input"] = ps.requires_full_payload_input
    if ps.execution_type is StageExecutionType.LLM_AR and ps.recompute_preemption != "allow":
        engine_args["recompute_preemption"] = ps.recompute_preemption

    async_scheduling = True
    if ds is not None and ds.async_scheduling is not None:
        async_scheduling = ds.async_scheduling
    if ps.execution_type == StageExecutionType.LLM_AR:
        engine_args["async_scheduling"] = bool(async_scheduling)
    return engine_args


def _build_extras(ps: StagePipelineConfig, ds: StageDeployConfig | None) -> dict[str, Any]:
    """拼 yaml_extras（采样参数 + connectors）。对应 stage_config.py:1423。"""
    extras: dict[str, Any] = {}
    sampling = merge_sampling_constraints(
        ds.default_sampling_params if ds is not None else None,
        ps.sampling_constraints,
    )
    if sampling:
        extras["default_sampling_params"] = sampling
    if ds is not None and ds.default_pooling_params:
        extras["default_pooling_params"] = dict(ds.default_pooling_params)
    if ds is not None and ds.output_connectors:
        extras["output_connectors"] = dict(ds.output_connectors)
    if ds is not None and ds.input_connectors:
        extras["input_connectors"] = dict(ds.input_connectors)
    if ps.extras:
        extras.update(ps.extras)
    return extras


def merge_pipeline_deploy(
    pipeline: PipelineConfig,
    deploy: DeployConfig,
    *,
    platform: str = "cuda",
) -> list[StageConfig]:
    """Merge pipeline + deploy + platform overrides → list[StageConfig]。

    对应 stage_config.py:1526。
    """
    deploy = _apply_platform_overrides(deploy, platform)
    deploy_by_id = {s.stage_id: s for s in deploy.stages}
    update_deploy_config_async_chunk_enabled(pipeline, deploy)

    result: list[StageConfig] = []
    for ps in pipeline.stages:
        ds = deploy_by_id.get(ps.stage_id)
        stage_type, worker_type = _EXECUTION_TYPE_TO_STAGE_WORKER[ps.execution_type]
        stage_async_chunk = resolve_stage_async_chunk(deploy, ds)
        input_proc, next_stage_proc = _select_processor_funcs(ps, stage_async_chunk)
        engine_args = _build_engine_args(ps, ds, pipeline, deploy, next_stage_proc)
        if not ps.requires_multimodal_data:
            engine_args.setdefault("skip_mm_profiling", True)
        sched_cls = _resolve_scheduler(ps.execution_type, engine_args.get("async_scheduling", True))
        extras = _build_extras(ps, ds)

        runtime: dict[str, Any] = {"process": True}
        if ds is not None:
            if ds.devices is not None:
                runtime["devices"] = ds.devices
            runtime["num_replicas"] = ds.num_replicas
            if ds.env is not None:
                runtime["env"] = ds.env
        runtime["requires_multimodal_data"] = ps.requires_multimodal_data

        result.append(
            StageConfig(
                stage_id=ps.stage_id,
                model_stage=ps.model_stage,
                session_mode=deploy.session_mode,
                stage_type=stage_type,
                input_sources=list(ps.input_sources),
                custom_process_input_func=input_proc,
                final_output=ps.final_output,
                final_output_type=ps.final_output_type,
                worker_type=worker_type,
                scheduler_cls=sched_cls,
                hf_config_name=ps.hf_config_name,
                sampling_constraints=dict(ps.sampling_constraints),
                yaml_engine_args=engine_args,
                yaml_runtime=runtime,
                yaml_extras=extras,
            )
        )
    return result


# --------------------------------------------------------------------------
# CLI override 归一化 —— 对应 stage_config.py:99-178
# --------------------------------------------------------------------------

_STAGE_OVERRIDE_PATTERN = re.compile(r"^stage_(\d+)_(.+)$")


def build_stage_runtime_overrides(
    stage_id: int,
    cli_overrides: dict[str, Any],
    *,
    internal_keys: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """从全局 + ``stage_<id>_*`` kwargs 构造某个 stage 的 runtime override。

    对应 stage_config.py:99：非本 stage 的 ``stage_<id>_*`` 被丢弃，
    其余全局 key 原样透传。
    """
    internal_keys = internal_keys or set()
    result: dict[str, Any] = {}
    for key, value in cli_overrides.items():
        if value is None or key in internal_keys:
            continue
        match = _STAGE_OVERRIDE_PATTERN.match(key)
        if match is not None:
            override_stage_id = int(match.group(1))
            param_name = match.group(2)
            if override_stage_id == stage_id and param_name not in internal_keys:
                result[param_name] = value
            continue
        result[key] = value
    return result


def normalize_pipeline_cli_overrides(
    pipeline: PipelineConfig,
    cli_overrides: dict[str, Any],
) -> dict[str, Any]:
    """把 pipeline 级全局 CLI 别名翻译成 stage 级 override。对应 stage_config.py:144。

    例如 Qwen3-TTS 的 ``--speaker`` 属于 stage 1 的 ``--talker_speaker``，
    则 ``speaker=x`` 会被改写成 ``stage_1_speaker=x``；若用户又显式给了
    ``stage_1_speaker``，则 stage 级优先并给出警告。
    """
    normalized = dict(cli_overrides)
    for source, (stage_id, target) in pipeline.stage_cli_aliases.items():
        invalid = [
            key
            for key, value in normalized.items()
            if value is not None
            and (m := _STAGE_OVERRIDE_PATTERN.match(key)) is not None
            and m.group(2) == source
            and int(m.group(1)) != stage_id
        ]
        if invalid:
            raise ValueError(
                f"{', '.join(sorted(invalid))} cannot be set for pipeline {pipeline.model_type!r}; "
                f"{source} belongs to stage {stage_id} as {target}."
            )
        value = normalized.pop(source, None)
        if value is None:
            continue
        stage_key = f"stage_{stage_id}_{target}"
        stage_value = normalized.get(stage_key)
        if stage_value is not None and stage_value != value:
            warnings.warn(
                f"Ignoring {source}={value!r} because {stage_key}={stage_value!r} takes precedence.",
                stacklevel=2,
            )
            continue
        normalized[stage_key] = value
    return normalized


# --------------------------------------------------------------------------
# 解析入口 —— 对应 resolver.py:196 resolve_omni_config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OmniConfigResolution:
    """配置解析的唯一启动交接对象。对应 resolver.py:25。"""

    config_path: str | None
    stage_configs: tuple[StageConfig, ...]
    pipeline_config: PipelineConfig | None = None

    def stage_by_id(self, stage_id: int) -> StageConfig:
        for stage in self.stage_configs:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(f"no stage {stage_id}; resolved stages: {[s.stage_id for s in self.stage_configs]}")


def resolve_omni_config(
    model_type: str,
    *,
    deploy_config_path: str | None = None,
    hf_config: Any = None,
    cli_overrides: dict[str, Any] | None = None,
    stage_overrides: dict[int, dict[str, Any]] | None = None,
    platform: str = "cuda",
) -> OmniConfigResolution:
    """registry/deploy 输入 -> 类型化启动交接对象（resolve_omni_config 精简版）。

    真实源码（resolver.py:196）还处理 HF config 推断、原生单文件 checkpoint、
    strategy specs、通用 diffusion 回退等；这里保留主干：
      registry 解析 pipeline → 加载 deploy → merge → OmniConfigResolution。
    """
    registry_overrides = dict(cli_overrides or {})
    # stage_<id>_* 形式的 per-stage override 先摊平（对应 resolver._flatten_stage_overrides）
    if stage_overrides:
        for sid, overrides in stage_overrides.items():
            if not isinstance(overrides, dict):
                raise TypeError(f"stage override {sid!r} must be a mapping")
            for key, value in overrides.items():
                registry_overrides[f"stage_{sid}_{key}"] = value

    pipeline_cfg = resolve_pipeline_config(model_type, hf_config)
    if pipeline_cfg is None:
        raise ValueError(
            f"Model type {model_type!r} did not resolve to a registered Omni pipeline. "
            f"Available: {sorted(OMNI_PIPELINES)}"
        )

    # CLI 全局别名 → stage 级
    registry_overrides = normalize_pipeline_cli_overrides(pipeline_cfg, registry_overrides)

    if deploy_config_path is not None:
        deploy = load_deploy_config(deploy_config_path)
    elif pipeline_cfg.default_deploy_config_name is not None:
        deploy_path = Path(__file__).resolve().parent.parent / "deploy" / pipeline_cfg.default_deploy_config_name
        deploy = load_deploy_config(deploy_path) if deploy_path.exists() else DeployConfig()
    else:
        deploy = DeployConfig()

    # per-stage CLI override：stage_<id>_* 下发（对应 stage_config.build_stage_runtime_overrides）
    stage_cli: dict[int, dict[str, Any]] = {}
    for key, value in registry_overrides.items():
        m = _STAGE_OVERRIDE_PATTERN.match(key)
        if m is not None and value is not None:
            stage_cli.setdefault(int(m.group(1)), {})[m.group(2)] = value

    stage_configs = merge_pipeline_deploy(pipeline_cfg, deploy, platform=platform)

    # runtime override 应用到 yaml_engine_args / yaml_runtime（CLI > YAML）
    for sc in stage_configs:
        overrides = stage_cli.get(sc.stage_id, {})
        for key, value in overrides.items():
            if key in ("devices", "num_replicas", "max_batch_size"):
                sc.yaml_runtime[key] = value
                continue
            existing = sc.yaml_engine_args.get(key)
            if key != "omni_kv_config" and isinstance(existing, dict) and isinstance(value, dict):
                sc.yaml_engine_args[key] = _get_recursively_merged_dict(existing, value)
            else:
                sc.yaml_engine_args[key] = value

    return OmniConfigResolution(
        config_path=str(deploy_config_path) if deploy_config_path else None,
        stage_configs=tuple(stage_configs),
        pipeline_config=pipeline_cfg,
    )


# --------------------------------------------------------------------------
# 内置示例 pipeline：Qwen3-Omni 风格的 Thinker → Talker → Code2Wav
# （对应 vllm_omni/model_executor/models/qwen3_omni/pipeline.py）
# --------------------------------------------------------------------------

_Q3O_PROC = "mini_omni.stage_input_processors.qwen3_omni"

QWEN3_OMNI_PIPELINE = PipelineConfig(
    model_type="qwen3_omni_moe",
    model_arch="Qwen3OmniMoeForConditionalGeneration",
    default_deploy_config_name="qwen3_omni_moe.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            hf_config_name="thinker_config",
            engine_output_type="latent",
            custom_process_next_stage_input_func=f"{_Q3O_PROC}.thinker2talker_full_payload",
            async_chunk_process_next_stage_input_func=f"{_Q3O_PROC}.thinker2talker_async_chunk",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            hf_config_name="talker_config",
            engine_output_type="latent",
            sync_process_input_func=f"{_Q3O_PROC}.thinker2talker_token_only",
            custom_process_next_stage_input_func=f"{_Q3O_PROC}.talker2code2wav_full_payload",
            async_chunk_process_next_stage_input_func=f"{_Q3O_PROC}.talker2code2wav_async_chunk",
            sampling_constraints={"detokenize": False, "stop_token_ids": [2150]},
            requires_full_payload_input=True,
            recompute_preemption="fail",
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            hf_config_name="thinker_config",
            engine_output_type="audio",
            requires_full_payload_input=True,
            sampling_constraints={"detokenize": True},
        ),
    ),
)

# 只跑 Thinker 的变体（对应 QWEN3_OMNI_THINKER_ONLY_PIPELINE）
QWEN3_OMNI_THINKER_ONLY_PIPELINE = PipelineConfig(
    model_type="qwen3_omni_moe_thinker_only",
    model_arch="Qwen3OmniMoeForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            hf_config_name="thinker_config",
            engine_output_type="text",
            sampling_constraints={"detokenize": True},
        ),
    ),
)


class _Qwen3OmniHfConfig:
    """模拟 transformers.Qwen3OmniMoeConfig（只保留判定用到的字段）。"""

    architectures = ["Qwen3OmniMoeForConditionalGeneration"]

    def __init__(self, enable_audio_output: bool = True) -> None:
        self.enable_audio_output = enable_audio_output


@pipeline_cfg_resolver(config_type=_Qwen3OmniHfConfig)
def resolve_qwen3_omni_pipeline(hf_config: _Qwen3OmniHfConfig) -> PipelineConfig:
    """按 HF config 选 pipeline 变体（对应 qwen3_omni/pipeline.py:119）。"""
    if not hf_config.enable_audio_output:
        return QWEN3_OMNI_THINKER_ONLY_PIPELINE
    return QWEN3_OMNI_PIPELINE


# 与上游一致：qwen3_omni_moe 注册 resolver（按 hf_config 选变体），
# thinker_only 变体单独注册（pipeline_registry.py:230-231）。
register_pipeline(resolve_qwen3_omni_pipeline, "qwen3_omni_moe")
register_pipeline(QWEN3_OMNI_THINKER_ONLY_PIPELINE)
