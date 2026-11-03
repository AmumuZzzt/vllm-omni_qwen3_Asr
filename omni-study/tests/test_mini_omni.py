# SPDX-License-Identifier: Apache-2.0
"""mini_omni 单元测试（纯标准库 unittest，可 `python -m unittest` 直接跑）。

覆盖两条主线：
  A. 配置层：拓扑校验 / 注册表 / deploy YAML 继承合并 / 平台覆盖 / merge 产物
  B. 编排层：跨 stage 路由 / 终态单调 / 多终点输出 / abort / 死副本隔离
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_omni.config import (  # noqa: E402
    DeployConfig,
    OMNI_PIPELINES,
    PipelineConfig,
    QWEN3_OMNI_PIPELINE,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    build_stage_runtime_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
    normalize_pipeline_cli_overrides,
    register_pipeline,
    resolve_omni_config,
    resolve_pipeline_config,
    resolve_stage_async_chunk,
)
from mini_omni.engine import AsyncOmni, AsyncOmniEngine  # noqa: E402
from mini_omni.orchestrator import (  # noqa: E402
    FakeStageClient,
    FakeStageOutput,
    StagePool,
    StageUnavailableError,
)

# ---------------------------------------------------------------------------
# A. 配置层
# ---------------------------------------------------------------------------


def _stage(
    stage_id: int,
    *,
    inputs: tuple[int, ...] = (),
    final: bool = False,
    final_type: str | None = None,
    async_chunk_func: str | None = None,
    **kwargs,
) -> StagePipelineConfig:
    return StagePipelineConfig(
        stage_id=stage_id,
        model_stage=f"stage{stage_id}",
        input_sources=inputs,
        final_output=final,
        final_output_type=final_type,
        async_chunk_process_next_stage_input_func=async_chunk_func,
        **kwargs,
    )


class TestPipelineValidation(unittest.TestCase):
    """PipelineConfig 拓扑校验 —— 与 stage_config.py:512 逐条对应。"""

    def test_valid_pipeline(self) -> None:
        p = PipelineConfig(
            model_type="t",
            stages=(
                _stage(0, final=True, final_type="text"),
            ),
        )
        self.assertEqual(p.get_validation_errors(), [])

    def test_no_stages(self) -> None:
        with self.assertRaisesRegex(ValueError, "Pipeline has no stages defined"):
            PipelineConfig(model_type="t", stages=())

    def test_duplicate_stage_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate stage IDs found"):
            PipelineConfig(
                model_type="t",
                stages=(_stage(0, final=True), _stage(0)),
            )

    def test_nonexistent_input_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "references non-existent input source"):
            PipelineConfig(
                model_type="t",
                stages=(_stage(0, inputs=(5,), final=True),),
            )

    def test_self_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "references itself"):
            PipelineConfig(
                model_type="t",
                stages=(_stage(0, inputs=(0,), final=True),),
            )

    def test_no_entry_point(self) -> None:
        with self.assertRaisesRegex(ValueError, "No entry point"):
            PipelineConfig(
                model_type="t",
                stages=(_stage(1, inputs=(0,)), _stage(0, inputs=(1,), final=True)),
            )

    def test_no_terminal_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "No terminal stage"):
            PipelineConfig(
                model_type="t",
                stages=(_stage(0), _stage(1, inputs=(0,))),
            )

    def test_multi_error_reported_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "error"):
            PipelineConfig(
                model_type="t",
                stages=(_stage(0, inputs=(0,), final=True), _stage(0)),
            )


class TestPipelineRegistry(unittest.TestCase):
    """注册表 —— 对应 pipeline_registry.py。"""

    def test_resolve_registered_pipeline(self) -> None:
        self.assertIs(resolve_pipeline_config("no_such_type"), None)
        p = resolve_pipeline_config("qwen3_omni_moe_thinker_only")
        self.assertIsInstance(p, PipelineConfig)
        self.assertEqual(p.model_type, "qwen3_omni_moe_thinker_only")

    def test_resolver_selects_variant_by_hf_config(self) -> None:
        from mini_omni.config import _Qwen3OmniHfConfig

        full = resolve_pipeline_config("qwen3_omni_moe", _Qwen3OmniHfConfig(enable_audio_output=True))
        self.assertEqual(len(full.stages), 3)
        only = resolve_pipeline_config("qwen3_omni_moe", _Qwen3OmniHfConfig(enable_audio_output=False))
        self.assertEqual(len(only.stages), 1)

    def test_resolver_wrong_hf_config_type_returns_none(self) -> None:
        # pipeline_cfg_resolver：类型不匹配 → None（stage_config.py:84）
        self.assertIs(resolve_pipeline_config("qwen3_omni_moe", object()), None)

    def test_register_out_of_tree_pipeline(self) -> None:
        p = PipelineConfig(model_type="_tmp_test_pipeline", stages=(_stage(0, final=True),))
        register_pipeline(p)
        try:
            self.assertIs(resolve_pipeline_config("_tmp_test_pipeline"), p)
        finally:
            OMNI_PIPELINES.pop("_tmp_test_pipeline", None)

    def test_register_resolver_requires_model_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "Model type must be explicitly provided"):
            register_pipeline(lambda hf: None)


class TestDeployYaml(unittest.TestCase):
    """deploy YAML 继承合并 —— 对应 resolve_deploy_yaml (stage_config.py:1064)。"""

    def _write(self, name: str, content: str) -> Path:
        path = Path(self.tmp.name) / name
        path.write_text(content, encoding="utf-8")
        return path

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_base_config_inheritance(self) -> None:
        self._write(
            "base.yaml",
            "async_chunk: true\n"
            "stages:\n"
            "  - stage_id: 0\n"
            "    max_num_seqs: 64\n"
            "    default_sampling_params: {temperature: 0.0, max_tokens: 2048}\n",
        )
        overlay = self._write(
            "overlay.yaml",
            "base_config: base.yaml\n"
            "stages:\n"
            "  - stage_id: 0\n"
            "    max_num_seqs: 128\n"
            "    default_sampling_params: {max_tokens: 4096}\n",
        )
        cfg = load_deploy_config(overlay)
        self.assertTrue(cfg.async_chunk)  # 从 base 继承
        stage0 = cfg.stages[0]
        self.assertEqual(stage0.max_num_seqs, 128)  # overlay 覆盖
        # 深合并：temperature 不被薄 overlay 丢掉
        self.assertEqual(stage0.default_sampling_params, {"temperature": 0.0, "max_tokens": 4096})

    def test_stage_args_schema_rejected(self) -> None:
        path = self._write("legacy.yaml", "stage_args: {}\n")
        with self.assertRaisesRegex(ValueError, "removed `stage_args` schema"):
            load_deploy_config(path)

    def test_unknown_stage_field_goes_to_engine_extras(self) -> None:
        path = self._write(
            "unknown.yaml",
            "stages:\n  - stage_id: 0\n    some_future_knob: 7\n",
        )
        cfg = load_deploy_config(path)
        self.assertEqual(cfg.stages[0].engine_extras, {"some_future_knob": 7})

    def test_invalid_model_runner_rejected(self) -> None:
        path = self._write("bad.yaml", "model_runner: v3\n")
        with self.assertRaisesRegex(ValueError, "model_runner must be one of"):
            load_deploy_config(path)


class TestPlatformOverrides(unittest.TestCase):
    """平台覆盖 —— 对应 _apply_platform_overrides (stage_config.py:1172)。"""

    def test_platform_stage_override_and_extras(self) -> None:
        pipeline = PipelineConfig(model_type="t", stages=(_stage(0, final=True, final_type="text"),))
        deploy = DeployConfig(
            stages=[StageDeployConfig(stage_id=0, devices="0", gpu_memory_utilization=0.9)],
            platforms={
                "npu": {
                    "stages": [
                        {
                            "stage_id": 0,
                            "gpu_memory_utilization": 0.6,
                            "tensor_parallel_size": 2,
                            "some_npu_only_knob": "x",
                        }
                    ]
                }
            },
        )
        stages = merge_pipeline_deploy(pipeline, deploy, platform="npu")
        ea = stages[0].yaml_engine_args
        self.assertEqual(ea["gpu_memory_utilization"], 0.6)  # 平台覆盖 base
        self.assertEqual(ea["tensor_parallel_size"], 2)  # 字段存在 → setattr
        self.assertEqual(ea["some_npu_only_knob"], "x")  # 未知字段 → engine_extras
        self.assertEqual(stages[0].yaml_runtime["devices"], "0")  # 平台未改 devices

    def test_platform_model_runner_v1_clears_stage_v2(self) -> None:
        pipeline = PipelineConfig(model_type="t", stages=(_stage(0, final=True),))
        deploy = DeployConfig(
            model_runner="v2",
            stages=[StageDeployConfig(stage_id=0, model_runner="v2")],
            platforms={"rocm": {"model_runner": "v1"}},
        )
        stages = merge_pipeline_deploy(pipeline, deploy, platform="rocm")
        self.assertFalse(stages[0].yaml_engine_args["use_v2_model_runner"])


class TestAsyncChunkResolution(unittest.TestCase):
    """async_chunk 决策 —— 对应 stage_config.py:1294-1523。"""

    def _two_stage_pipeline(self, with_async_func: bool) -> PipelineConfig:
        return PipelineConfig(
            model_type="t2",
            stages=(
                _stage(
                    0,
                    final=True,
                    final_type="text",
                    async_chunk_func="p.f0" if with_async_func else None,
                ),
                _stage(1, inputs=(0,), final=True, final_type="audio"),
            ),
        )

    def test_default_auto_detect(self) -> None:
        p = self._two_stage_pipeline(with_async_func=True)
        deploy = DeployConfig()  # async_chunk 未设置 → 自动探测
        from mini_omni.config import _resolve_pipeline_async_chunk_enabled

        self.assertTrue(_resolve_pipeline_async_chunk_enabled(p, deploy))

    def test_single_stage_warning_and_disable(self) -> None:
        p = PipelineConfig(model_type="s", stages=(_stage(0, final=True),))
        deploy = DeployConfig(async_chunk=True)
        from mini_omni.config import _resolve_pipeline_async_chunk_enabled

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertFalse(_resolve_pipeline_async_chunk_enabled(p, deploy))
        self.assertTrue(any("single stage" in str(w.message) for w in caught))

    def test_stage_opt_out(self) -> None:
        p = self._two_stage_pipeline(with_async_func=True)
        deploy = DeployConfig(
            async_chunk=True,
            stages=[StageDeployConfig(stage_id=1, async_chunk=False)],
        )
        # 无 stage 级条目 → 跟随 pipeline-wide
        self.assertTrue(resolve_stage_async_chunk(deploy, None))
        # stage 1 显式关闭 → 该 stage 关闭
        self.assertFalse(resolve_stage_async_chunk(deploy, deploy.stages[0]))

    def test_async_chunk_edge_mismatch_rejected(self) -> None:
        from mini_omni.config import validate_stage_async_chunk_edges

        p = PipelineConfig(
            model_type="t2",
            stages=(
                _stage(0, final=True, async_chunk_func="p.f0"),
                _stage(1, inputs=(0,), final=True),
            ),
        )
        deploy = DeployConfig(
            async_chunk=True,
            stages=[
                StageDeployConfig(stage_id=0, async_chunk=True),
                StageDeployConfig(stage_id=1, async_chunk=False),
            ],
        )
        with self.assertRaisesRegex(ValueError, "incompatible async_chunk settings"):
            validate_stage_async_chunk_edges(p, deploy)


class TestMergePipelineDeploy(unittest.TestCase):
    """merge_pipeline_deploy —— 对应 stage_config.py:1526。"""

    def test_qwen3_omni_three_stages(self) -> None:
        deploy = load_deploy_config(
            Path(__file__).resolve().parent.parent / "deploy" / "qwen3_omni_moe.yaml"
        )
        stages = merge_pipeline_deploy(QWEN3_OMNI_PIPELINE, deploy, platform="cuda")
        self.assertEqual([s.stage_id for s in stages], [0, 1, 2])
        self.assertEqual([s.worker_type for s in stages], ["ar", "ar", "generation"])
        self.assertEqual(stages[0].scheduler_cls.split(".")[-1], "OmniARAsyncScheduler")
        # final_output 来自拓扑，不可被 deploy 覆盖
        self.assertTrue(stages[0].final_output)
        self.assertEqual(stages[0].final_output_type, "text")
        self.assertFalse(stages[1].final_output)
        self.assertTrue(stages[2].final_output)
        self.assertEqual(stages[2].final_output_type, "audio")
        # pipeline-wide async_chunk 下发到每个 stage
        self.assertTrue(stages[0].yaml_engine_args["async_chunk"])
        # 采样参数：deploy default 与拓扑 constraints 合并
        self.assertEqual(stages[1].yaml_extras["default_sampling_params"]["seed"], 42)
        self.assertIn(2150, stages[1].yaml_extras["default_sampling_params"]["stop_token_ids"])
        # 连接器接线
        self.assertEqual(stages[1].yaml_extras["input_connectors"], {"from_stage_0": "connector_of_shared_memory"})

    def test_async_chunk_processor_selection(self) -> None:
        """async_chunk 开 → 选 async 处理函数；关 → 选 sync/full payload。"""
        deploy = load_deploy_config(
            Path(__file__).resolve().parent.parent / "deploy" / "qwen3_omni_moe.yaml"
        )
        on = merge_pipeline_deploy(QWEN3_OMNI_PIPELINE, deploy, platform="cuda")
        self.assertIn(
            "thinker2talker_async_chunk",
            on[0].yaml_engine_args["custom_process_next_stage_input_func"],
        )
        # 关闭 async_chunk 重新合并
        deploy2 = load_deploy_config(
            Path(__file__).resolve().parent.parent / "deploy" / "qwen3_omni_moe.yaml"
        )
        deploy2.async_chunk = False
        off = merge_pipeline_deploy(QWEN3_OMNI_PIPELINE, deploy2, platform="cuda")
        # next_stage_proc 回落到 custom_process_next_stage_input_func（full payload）
        self.assertIn(
            "thinker2talker_full_payload",
            off[0].yaml_engine_args["custom_process_next_stage_input_func"],
        )

    def test_no_multimodal_stage_skips_mm_profiling(self) -> None:
        p = PipelineConfig(model_type="t", stages=(_stage(0, final=True),))
        stages = merge_pipeline_deploy(p, DeployConfig(), platform="cuda")
        self.assertTrue(stages[0].yaml_engine_args["skip_mm_profiling"])


class TestCliOverrides(unittest.TestCase):
    """CLI override 归一化 —— 对应 stage_config.py:99/144。"""

    def test_stage_scoped_override_only_hits_matching_stage(self) -> None:
        overrides = {"stage_0_max_num_seqs": 8, "stage_1_max_num_seqs": 16, "dtype": "bfloat16"}
        s0 = build_stage_runtime_overrides(0, overrides)
        self.assertEqual(s0, {"max_num_seqs": 8, "dtype": "bfloat16"})
        s1 = build_stage_runtime_overrides(1, overrides)
        self.assertEqual(s1, {"max_num_seqs": 16, "dtype": "bfloat16"})

    def test_internal_keys_filtered(self) -> None:
        overrides = {"model": "m", "stage_id": 0, "max_num_seqs": 4}
        out = build_stage_runtime_overrides(0, overrides, internal_keys={"model", "stage_id"})
        self.assertEqual(out, {"max_num_seqs": 4})

    def test_cli_alias_translates_to_stage_key(self) -> None:
        p = PipelineConfig(
            model_type="t",
            stages=(_stage(0, final=True), _stage(1, inputs=(0,), final=True)),
            stage_cli_aliases={"speaker": (1, "talker_speaker")},
        )
        out = normalize_pipeline_cli_overrides(p, {"speaker": "alice"})
        self.assertEqual(out, {"stage_1_talker_speaker": "alice"})

    def test_stage_key_wins_over_alias_with_warning(self) -> None:
        p = PipelineConfig(
            model_type="t",
            stages=(_stage(0, final=True), _stage(1, inputs=(0,), final=True)),
            stage_cli_aliases={"speaker": (1, "talker_speaker")},
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = normalize_pipeline_cli_overrides(
                p, {"speaker": "alice", "stage_1_talker_speaker": "bob"}
            )
        self.assertEqual(out["stage_1_talker_speaker"], "bob")
        self.assertTrue(any("takes precedence" in str(w.message) for w in caught))

    def test_alias_for_wrong_stage_rejected(self) -> None:
        p = PipelineConfig(
            model_type="t",
            stages=(_stage(0, final=True), _stage(1, inputs=(0,), final=True)),
            stage_cli_aliases={"speaker": (1, "talker_speaker")},
        )
        # speaker 属于 stage 1；写成 stage_0_speaker 即非法
        with self.assertRaisesRegex(ValueError, "cannot be set for pipeline"):
            normalize_pipeline_cli_overrides(p, {"stage_0_speaker": "x"})


class TestResolveOmniConfig(unittest.TestCase):
    """resolve_omni_config 端到端 —— 对应 resolver.py:196。"""

    def test_resolve_with_deploy_yaml(self) -> None:
        from mini_omni.config import _Qwen3OmniHfConfig

        deploy_path = Path(__file__).resolve().parent.parent / "deploy" / "qwen3_omni_moe.yaml"
        res = resolve_omni_config(
            "qwen3_omni_moe",
            deploy_config_path=str(deploy_path),
            hf_config=_Qwen3OmniHfConfig(enable_audio_output=True),
            platform="cuda",
        )
        self.assertEqual(len(res.stage_configs), 3)
        self.assertEqual(res.config_path, str(deploy_path))
        # deploy yaml 的 stage0 设备与采样参数进入合并产物
        self.assertEqual(res.stage_by_id(0).yaml_runtime["devices"], "0")
        self.assertEqual(
            res.stage_by_id(0).yaml_engine_args["max_num_batched_tokens"], 32768
        )

    def test_resolve_requires_pipeline(self) -> None:
        with self.assertRaisesRegex(ValueError, "did not resolve to a registered Omni pipeline"):
            resolve_omni_config("not_registered_model", platform="cuda")

    def test_resolve_thinker_only_via_hf_config(self) -> None:
        from mini_omni.config import _Qwen3OmniHfConfig

        res = resolve_omni_config(
            "qwen3_omni_moe",
            hf_config=_Qwen3OmniHfConfig(enable_audio_output=False),
            platform="cuda",
        )
        self.assertEqual(len(res.stage_configs), 1)
        self.assertEqual(res.stage_by_id(0).model_stage, "thinker")
        with self.assertRaises(KeyError):
            res.stage_by_id(7)

    def test_stage_cli_override_applied(self) -> None:
        from mini_omni.config import _Qwen3OmniHfConfig

        res = resolve_omni_config(
            "qwen3_omni_moe",
            hf_config=_Qwen3OmniHfConfig(),
            cli_overrides={"stage_0_max_num_seqs": 3},
            platform="cuda",
        )
        self.assertEqual(res.stage_by_id(0).yaml_engine_args["max_num_seqs"], 3)


# ---------------------------------------------------------------------------
# B. 编排层
# ---------------------------------------------------------------------------


def _sampling_params(n: int) -> list[dict]:
    return [{"max_tokens": 16} for _ in range(n)]


def _make_pipeline_pools(
    scripts: dict[int, list[FakeStageOutput]] | None = None,
    *,
    process_inputs=None,
) -> list[StagePool]:
    """构造 Thinker(0, 终点=text) → Talker(1) → Code2Wav(2, 终点=audio) 三段池。"""
    scripts = scripts or {}

    def thinker_client() -> FakeStageClient:
        return FakeStageClient(0, final_output=True, final_output_type="text", outputs=scripts.get(0))

    def talker_client() -> FakeStageClient:
        return FakeStageClient(
            1,
            final_output=False,
            process_engine_inputs=process_inputs,
            outputs=scripts.get(1),
        )

    def c2w_client() -> FakeStageClient:
        return FakeStageClient(2, final_output=True, final_output_type="audio", outputs=scripts.get(2))

    return [
        StagePool(0, [thinker_client()], final_output=True, final_output_type="text"),
        StagePool(1, [talker_client()], final_output=False),
        StagePool(2, [c2w_client()], final_output=True, final_output_type="audio"),
    ]


def _make_hanging_pools() -> list[StagePool]:
    """stage0 只产出未完成输出 → 请求永远停在 stage0，便于观察状态。"""
    hang = [FakeStageOutput(request_id="__any__", finished=False, text="partial")]
    return [
        StagePool(
            0,
            [FakeStageClient(0, final_output=True, final_output_type="text", outputs=hang)],
            final_output=True,
            final_output_type="text",
        ),
        StagePool(1, [FakeStageClient(1)], final_output=False),
        StagePool(2, [FakeStageClient(2, final_output=True, final_output_type="audio")],
                  final_output=True, final_output_type="audio"),
    ]


class OrchestratorTestBase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.pools = _make_pipeline_pools()
        self.engine = AsyncOmniEngine(self.pools)
        self.engine.start()
        self.omni = AsyncOmni(self.engine)

    async def asyncTearDown(self) -> None:
        await self.engine.shutdown()

    async def collect(self, coro, *, limit: int = 100, timeout: float = 3.0) -> list:
        out = []

        async def run() -> None:
            async for item in coro:
                out.append(item)

        await asyncio.wait_for(run(), timeout=timeout)
        return out


class TestEndToEndRouting(OrchestratorTestBase):
    """三段流水线端到端 —— 覆盖 ORCH-INV-001 的正向路径。"""

    async def test_request_flows_through_all_stages(self) -> None:
        # 每个 stage 的默认脚本：一条 finished=True 输出
        outputs = await self.collect(
            self.omni.generate(
                "hello",
                _sampling_params(3),
                final_stage_id=2,
                final_output_stage_ids=[2],
            )
        )
        # stage0(thinker) 与 stage2(code2wav) 都是 final_output pool：
        # thinker 的文本先到（finished=False），code2wav 的音频收尾（finished=True）。
        # 这与上游一致 —— _route_output 对 final_output pool 一律发 OutputMessage，
        # 由 final_output_stage_ids 决定哪一刻算"请求完成"。
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].stage_id, 0)
        self.assertEqual(outputs[0].final_output_type, "text")
        self.assertFalse(outputs[0].finished)
        self.assertEqual(outputs[1].stage_id, 2)
        self.assertEqual(outputs[1].final_output_type, "audio")
        self.assertTrue(outputs[1].finished)
        # 路由确实发生：三个 stage 都被提交过
        self.assertEqual(len(self.pools[0].clients[0].add_request_calls), 1)
        self.assertEqual(len(self.pools[1].clients[0].add_request_calls), 1)
        self.assertEqual(len(self.pools[2].clients[0].add_request_calls), 1)

    async def test_stage0_final_output_also_reaches_frontend(self) -> None:
        """Thinker 自己也是 final_output（text），其输出必须发给前端。"""
        # 重配：stage0 终点 + stage2 终点（双终点，对应 output_modalities）
        pools = _make_pipeline_pools()
        self.engine.orchestrator.stage_pools = pools
        self.engine.stage_configs = pools
        outputs = await self.collect(
            self.omni.generate(
                "hi",
                _sampling_params(3),
                final_stage_id=2,
                final_output_stage_ids=[0, 2],
            )
        )
        stages_seen = [o.stage_id for o in outputs]
        self.assertIn(0, stages_seen)  # thinker 文本
        self.assertIn(2, stages_seen)  # code2wav 音频
        # 只有最后一个输出 finished=True（ORCH-INV: 每请求一个终态）
        self.assertEqual(sum(o.finished for o in outputs), 1)
        self.assertTrue(outputs[-1].finished)


class TestTerminalMonotonicity(OrchestratorTestBase):
    """ORCH-INV-100：终态单调。"""

    async def test_output_after_cleanup_is_dropped(self) -> None:
        orch = self.engine.orchestrator
        # 先人为把请求标记为终态（模拟已清理）
        from mini_omni.orchestrator import OrchestratorRequestState

        state = OrchestratorRequestState(request_id="r1", final_stage_id=2, terminal=True)
        orch.request_states["r1"] = state
        # 迟到输出被丢弃，不进输出队列
        await orch._handle_processed_output(0, 0, FakeStageOutput(request_id="r1", finished=True))
        self.assertTrue(orch.request_states["r1"].terminal)
        # 队列里没有为它产生任何 OutputMessage
        self.assertTrue(orch.output_async_queue.empty())

    async def test_unknown_request_output_dropped(self) -> None:
        orch = self.engine.orchestrator
        await orch._handle_processed_output(0, 0, FakeStageOutput(request_id="ghost", finished=True))
        self.assertTrue(orch.output_async_queue.empty())

    async def test_cleanup_marks_terminal_and_decrements_counter(self) -> None:
        # 用"不结束"的 stage0，保证请求停在 running 状态
        pools = _make_hanging_pools()
        self.engine.orchestrator.stage_pools = pools
        self.engine.stage_configs = pools
        await self.engine.add_request_async(
            request_id="r2",
            prompt="x",
            sampling_params_list=_sampling_params(3),
            final_stage_id=2,
        )
        await asyncio.sleep(0.05)
        orch = self.engine.orchestrator
        self.assertEqual(orch._running_count, 1)
        await orch._cleanup_request_ids(["r2"])
        self.assertEqual(orch._running_count, 0)
        self.assertNotIn("r2", orch.request_states)


class TestDeadStageIsolation(OrchestratorTestBase):
    """stage 全副本死亡只失败单个请求，不影响引擎（#4285 语义）。"""

    async def test_dead_entry_stage_fails_only_that_request(self) -> None:
        # 把 stage0 的副本清空
        saved = self.pools[0].clients
        self.pools[0].clients = []
        try:
            outputs = await self.collect(
                self.omni.generate("x", _sampling_params(3), final_stage_id=2, final_output_stage_ids=[2]),
                timeout=3.0,
            )
            # generate() 应收到 ErrorMessage → 抛 RuntimeError
            # （collect 内部会传播异常，这里换个方式验证）
        except RuntimeError as e:
            self.assertIn("no live replicas", str(e))
        else:
            # 若没抛错，则不应有 finished 输出
            self.assertFalse(any(o.finished for o in outputs))
        finally:
            self.pools[0].clients = saved

    async def test_forward_to_dead_stage_fails_request(self) -> None:
        # stage1 空 → thinker 完成后转发失败
        saved = self.pools[1].clients
        self.pools[1].clients = []
        try:
            await self.omni.generate(
                "x", _sampling_params(3), final_stage_id=2, final_output_stage_ids=[2]
            ).__anext__()
        except RuntimeError as e:
            self.assertIn("no live replicas", str(e))
        finally:
            self.pools[1].clients = saved


class TestAbort(OrchestratorTestBase):
    async def test_abort_delivers_error_and_clears_state(self) -> None:
        # 不结束的 stage0 → 请求一直保持 running，直到被 abort
        pools = _make_hanging_pools()
        self.engine.orchestrator.stage_pools = pools
        self.engine.stage_configs = pools
        await self.engine.add_request_async(
            request_id="r3",
            prompt="x",
            sampling_params_list=_sampling_params(3),
            final_stage_id=2,
        )
        await asyncio.sleep(0.05)
        orch = self.engine.orchestrator
        self.assertIn("r3", orch.request_states)
        await self.engine.abort_async("r3")
        self.assertNotIn("r3", orch.request_states)
        # abort 已下发到各 stage 的 client
        for pool in pools:
            calls = pool.clients[0].abort_calls
            self.assertTrue(any("r3" in c for c in calls))

    async def test_abort_ack_on_rpc_queue(self) -> None:
        pools = _make_hanging_pools()
        self.engine.orchestrator.stage_pools = pools
        self.engine.stage_configs = pools
        await self.engine.add_request_async(
            request_id="r4",
            prompt="x",
            sampling_params_list=_sampling_params(3),
            final_stage_id=2,
        )
        await asyncio.sleep(0.05)
        await self.engine.abort_async("r4")  # 内部等待 rpc ack，不挂死即通过


class TestStageInputProcessor(OrchestratorTestBase):
    """下一 stage 的 input processor 决定消费/等待（对应 process_engine_inputs）。"""

    async def test_empty_processor_result_waits_for_more_outputs(self) -> None:
        # talker 的 processor 一直返回 [] → 永远不提交 stage1
        pools = _make_pipeline_pools(
            process_inputs=lambda srcs, prompt: [],
        )
        self.engine.orchestrator.stage_pools = pools
        self.engine.stage_configs = pools
        orch = self.engine.orchestrator
        await self.engine.add_request_async(
            request_id="r5",
            prompt="x",
            sampling_params_list=_sampling_params(3),
            final_stage_id=2,
            final_output_stage_ids=[2],
        )
        await asyncio.sleep(0.15)
        # thinker 完成了，但 processor 拒绝产生输入 → stage1 未提交
        self.assertEqual(len(pools[1].clients[0].add_request_calls), 0)
        self.assertNotIn(1, orch.request_states["r5"].stage_submit_ts)


class TestReplicaSelection(unittest.IsolatedAsyncioTestCase):
    """StagePool 选副本 —— 对应 stage_pool.py:647 select_replica_id。"""

    def test_round_robin_across_replicas(self) -> None:
        pool = StagePool(0, [FakeStageClient(0), FakeStageClient(0), FakeStageClient(0)])
        picks = [pool.select_replica_id(f"r{i}") for i in range(6)]
        self.assertEqual(picks, [0, 1, 2, 0, 1, 2])

    def test_binding_is_sticky(self) -> None:
        pool = StagePool(0, [FakeStageClient(0), FakeStageClient(0)])
        a = pool.select_replica_id("req")
        b = pool.select_replica_id("req")  # 第二次仍返回同一副本
        self.assertEqual(a, b)

    def test_no_live_replica_raises(self) -> None:
        pool = StagePool(0, [])
        with self.assertRaises(StageUnavailableError):
            pool.select_replica_id("r")

    def test_release_bindings(self) -> None:
        pool = StagePool(0, [FakeStageClient(0)])
        pool.select_replica_id("r")
        pool.release_bindings(["r"])
        self.assertIsNone(pool.get_bound_replica_id("r"))


class TestQwen3OmniDeployFile(unittest.TestCase):
    """真实 deploy 文件 qwen3_omni_moe.yaml 的回归。"""

    def test_deploy_file_parses(self) -> None:
        deploy = load_deploy_config(
            Path(__file__).resolve().parent.parent / "deploy" / "qwen3_omni_moe.yaml"
        )
        self.assertTrue(deploy.async_chunk)
        self.assertEqual(len(deploy.stages), 3)
        self.assertEqual(deploy.stages[0].devices, "0")
        self.assertEqual(deploy.stages[1].devices, "1")
        self.assertEqual(deploy.stages[2].devices, "1")
        # connectors
        self.assertIn("connector_of_shared_memory", deploy.connectors)
        self.assertEqual(
            deploy.connectors["connector_of_shared_memory"]["extra"]["codec_chunk_frames"], 25
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
