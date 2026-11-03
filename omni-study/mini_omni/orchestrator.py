# SPDX-License-Identifier: Apache-2.0
"""手搓版 vllm_omni 编排器（Orchestrator）。

对照源码：
  - vllm_omni/engine/orchestrator.py  (OrchestratorBase / Orchestrator,
                                        OrchestratorRequestState, _route_output,
                                        _forward_to_next_stage, _handle_add_request,
                                        _cleanup_request_ids, _handle_abort)
  - vllm_omni/engine/stage_pool.py     (StagePool 选副本 / 提交 / abort)
  - vllm_omni/engine/messages.py       (队列消息类型)

保留的核心契约（见 docs/design/module/engine_orchestration.md）：
  ORCH-INV-001  只有 Orchestrator 拥有跨 stage 路由权（entrypoint / stage client 不许自己转发）
  ORCH-INV-100  终态单调：请求一旦进入终态，不再为其转发任何新工作

与上游的差异（为了能在纯 Python 下跑）：
  * stage 引擎用 ``FakeStageClient`` 替代真实 EngineCore 进程；
  * 输出队列用 asyncio.Queue 替代 janus 队列（上游跑在后台线程的 loop 里）；
  * 省略 CFG companion、PD 分离、duplex session、first-audio 排序等旁路。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("mini_omni.orchestrator")

__all__ = [
    "EngineQueueMessage",
    "StageSubmissionMessage",
    "AbortRequestMessage",
    "ShutdownRequestMessage",
    "OutputMessage",
    "ErrorMessage",
    "StageMetricsMessage",
    "OrchestratorRequestState",
    "StageUnavailableError",
    "Orchestrator",
]


# --------------------------------------------------------------------------
# 队列消息 —— 对应 engine/messages.py（msgspec.Struct → dataclass）
# --------------------------------------------------------------------------


@dataclass
class EngineQueueMessage:
    type: str = ""


@dataclass
class StageSubmissionMessage(EngineQueueMessage):
    type: str = "add_request"
    request_id: str = ""
    prompt: Any = None
    original_prompt: Any = None
    sampling_params_list: list[Any] = field(default_factory=list)
    final_stage_id: int = -1
    request_timestamp: float = 0.0
    final_output_stage_ids: list[int] | None = None
    entry_stage_id: int = 0


@dataclass
class AbortRequestMessage(EngineQueueMessage):
    type: str = "abort"
    request_ids: list[str] = field(default_factory=list)
    # 设置后 orchestrator 完成 abort 后回一条确认（上游 rpc_id 语义）
    rpc_id: str | None = None


@dataclass
class ShutdownRequestMessage(EngineQueueMessage):
    type: str = "shutdown"


@dataclass
class OutputMessage(EngineQueueMessage):
    type: str = "output"
    request_id: str = ""
    stage_id: int = -1
    replica_id: int | None = None
    engine_outputs: Any = None
    metrics: Any = None
    finished: bool = False
    stage_submit_ts: float | None = None


@dataclass
class ErrorMessage(EngineQueueMessage):
    type: str = "error"
    error: str = ""
    status_code: int | None = None
    request_id: str | None = None
    stage_id: int | None = None
    fatal: bool = False


@dataclass
class StageMetricsMessage(EngineQueueMessage):
    type: str = "stage_metrics"
    request_id: str = ""
    stage_id: int = -1
    replica_id: int | None = None
    metrics: Any = None
    stage_submit_ts: float | None = None


@dataclass
class AbortResultMessage(EngineQueueMessage):
    type: str = "abort_result"
    rpc_id: str = ""
    success: bool = True
    error: str | None = None


class StageUnavailableError(RuntimeError):
    """stage 没有可用副本（对应 stage_pool.StageUnavailableError）。"""


# --------------------------------------------------------------------------
# 请求状态 —— 对应 orchestrator.py:239 OrchestratorRequestState（精简）
# --------------------------------------------------------------------------


@dataclass
class OrchestratorRequestState:
    request_id: str
    prompt: Any = None
    sampling_params_list: list[Any] = field(default_factory=list)
    final_stage_id: int = -1
    entry_stage_id: int = 0
    # 请求结束需要哪些 final_output stage 全部完成（多模态输出可有多个）
    final_output_stage_ids: set[int] = field(default_factory=set)
    finished_final_output_stage_ids: set[int] = field(default_factory=set)
    # 所有提交过的 stage 中，已回报完成的（用于 pending_final_output 判定）
    finished_stage_ids: set[int] = field(default_factory=set)
    pending_final_output: OutputMessage | None = None
    request_timestamp: float = 0.0
    # 每个 stage 的提交时间戳（也是"该 stage 已提交"的标记）
    stage_submit_ts: dict[int, float] = field(default_factory=dict)
    # 终态单调：进入终态后拒绝一切新转发（ORCH-INV-100）
    terminal: bool = False


# --------------------------------------------------------------------------
# Fake stage 引擎 —— 替代真实 EngineCore
# --------------------------------------------------------------------------


@dataclass
class FakeStageOutput:
    """一个 stage 引擎产出的"processed output"。

    对应上游经 output processor 之后的 OmniRequestOutput 子集。
    """

    request_id: str
    finished: bool
    text: str = ""
    payload: Any = None  # 跨 stage 传递的 latent / codec / waveform
    error: str | None = None


class FakeStageClient:
    """单个 stage 的假引擎：按提交顺序吐出配置好的输出序列。"""

    def __init__(
        self,
        stage_id: int,
        *,
        final_output: bool = False,
        final_output_type: str | None = None,
        outputs: list[FakeStageOutput] | None = None,
        process_engine_inputs: Callable[[list[Any], Any], list[Any]] | None = None,
        requires_multimodal_data: bool = False,
    ) -> None:
        self.stage_id = stage_id
        self.final_output = final_output
        self.final_output_type = final_output_type
        self.requires_multimodal_data = requires_multimodal_data
        self.process_engine_inputs = process_engine_inputs
        # request_id -> 待产出的输出队列
        self._pending: dict[str, list[FakeStageOutput]] = {}
        self._default_outputs = list(outputs or [])
        self.add_request_calls: list[str] = []
        self.abort_calls: list[list[str]] = []
        self.submitted: set[str] = set()

    # --- orchestrator 面向的接口（对应 StagePoolClient） ---

    def add_request(self, request_id: str, prompt: Any, **kwargs: Any) -> None:
        self.add_request_calls.append(request_id)
        self.submitted.add(request_id)
        if request_id not in self._pending:
            # prompt 里可带 "script" 覆盖；否则用构造时的默认输出序列；
            # 都没有则合成一条 finished=True 的输出
            script = None
            if isinstance(prompt, dict):
                script = prompt.get("script")
            if script is None:
                script = self._default_outputs or [
                    FakeStageOutput(
                        request_id=request_id,
                        finished=True,
                        text=f"stage{self.stage_id}-done",
                    )
                ]
            self._pending[request_id] = [
                FakeStageOutput(
                    request_id=request_id,
                    finished=r.finished,
                    text=r.text,
                    payload=r.payload,
                    error=r.error,
                )
                for r in script
            ]

    def poll(self, request_id: str) -> FakeStageOutput | None:
        queue = self._pending.get(request_id)
        if not queue:
            return None
        out = queue.pop(0)
        if not queue:
            self._pending.pop(request_id, None)
        return out

    def abort(self, request_ids: list[str]) -> None:
        self.abort_calls.append(list(request_ids))
        for rid in request_ids:
            self._pending.pop(rid, None)
            self.submitted.discard(rid)


# --------------------------------------------------------------------------
# StagePool —— 对应 engine/stage_pool.py（选副本 / 绑定 / 提交 / abort）
# --------------------------------------------------------------------------


class StagePool:
    def __init__(
        self,
        stage_id: int,
        clients: list[FakeStageClient],
        *,
        stage_type: str = "llm",
        final_output: bool = False,
        final_output_type: str | None = None,
    ) -> None:
        self.stage_id = stage_id
        self.clients = clients
        self.stage_type = stage_type
        self.final_output = final_output
        self.final_output_type = final_output_type
        self._bindings: dict[str, int] = {}
        self._next_replica_id = 0
        self.stage_vllm_config = None

    @property
    def num_replicas(self) -> int:
        return len(self.clients)

    def live_replica_ids(self) -> list[int]:
        return list(range(len(self.clients)))

    @property
    def stage_client(self) -> FakeStageClient | None:
        return self.clients[0] if self.clients else None

    def select_replica_id(self, request_id: str) -> int:
        """选副本：绑定优先，否则对 live 副本轮转。对应 stage_pool.py:647。"""
        cached = self._bindings.get(request_id)
        if cached is not None and cached in self.live_replica_ids():
            return cached
        live = self.live_replica_ids()
        if not live:
            raise StageUnavailableError(f"stage {self.stage_id} has no live replicas")
        if len(live) == 1:
            chosen = live[0]
        else:
            chosen = live[self._next_replica_id % len(live)]
            self._next_replica_id = (self._next_replica_id + 1) % len(live)
        self._bindings[request_id] = chosen
        return chosen

    def get_bound_replica_id(self, request_id: str) -> int | None:
        return self._bindings.get(request_id)

    def release_bindings(self, request_ids: list[str]) -> None:
        for rid in request_ids:
            self._bindings.pop(rid, None)

    async def submit_initial(
        self,
        request_id: str,
        req_state: OrchestratorRequestState,
        request: Any,
        *,
        prompt_text: Any = None,
        expected_replica_id: int | None = None,
    ) -> int:
        params = req_state.sampling_params_list[self.stage_id]
        replica_id = self.select_replica_id(request_id)
        if expected_replica_id is not None and replica_id != expected_replica_id:
            self.release_bindings([request_id])
            raise StageUnavailableError(
                f"stage {self.stage_id} replica {expected_replica_id} became unavailable before submit"
            )
        client = self.clients[replica_id]
        try:
            client.add_request(request_id, request, params=params, prompt_text=prompt_text)
        except BaseException:
            self.release_bindings([request_id])
            raise
        return replica_id

    async def abort_requests(self, request_ids: list[str]) -> list[tuple[str, Any]]:
        """把 abort 下发到每个副本；返回 (request_id, output) 对（AR 终点 stage 的部分输出）。"""
        results: list[tuple[str, Any]] = []
        for client in self.clients:
            client.abort(request_ids)
        return results

    def poll_outputs(self) -> list[tuple[int, FakeStageOutput]]:
        """轮询本 pool 所有副本，返回 (replica_id, output) 列表。"""
        got: list[tuple[int, FakeStageOutput]] = []
        for replica_id, client in enumerate(self.clients):
            for rid in list(client.submitted):
                out = client.poll(rid)
                if out is not None:
                    got.append((replica_id, out))
        return got


# --------------------------------------------------------------------------
# Orchestrator —— 对应 engine/orchestrator.py
# --------------------------------------------------------------------------


class Orchestrator:
    """事件驱动编排器：收 add_request / abort，轮询 stage 输出并做跨 stage 路由。

    上游默认是事件驱动 loop（orchestrator.py:1214），这里统一用
    "request handler 任务 + output handler 任务"的双任务结构（run()，
    orchestrator.py:491），输出侧用 1ms 轮询（对应 legacy loop:1154）。
    """

    def __init__(
        self,
        request_async_queue: asyncio.Queue[EngineQueueMessage],
        output_async_queue: asyncio.Queue[EngineQueueMessage],
        rpc_async_queue: asyncio.Queue[EngineQueueMessage],
        stage_pools: list[StagePool],
        *,
        async_chunk: bool = False,
        poll_interval: float = 0.001,
    ) -> None:
        self.request_async_queue = request_async_queue
        self.output_async_queue = output_async_queue
        self.rpc_async_queue = rpc_async_queue
        self.stage_pools = stage_pools
        self.num_stages = len(stage_pools)
        self.async_chunk = async_chunk
        self.poll_interval = poll_interval

        self.request_states: dict[str, OrchestratorRequestState] = {}
        self._shutdown_event = asyncio.Event()
        self._running_count = 0  # 上游 OmniRequestCounter 的等价物

    # ---- 生命周期（对应 orchestrator.py:491 run） ----

    async def run(self) -> None:
        logger.info("[Orchestrator] Starting event loop")
        request_task = asyncio.create_task(self._request_handler(), name="orchestrator-request-handler")
        output_task = asyncio.create_task(
            self._orchestration_output_handler(), name="orchestrator-stage-output-handler"
        )
        try:
            await asyncio.gather(request_task, output_task)
        finally:
            self._shutdown_event.set()
            for task in (request_task, output_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(request_task, output_task, return_exceptions=True)

    def shutdown(self) -> None:
        self._shutdown_event.set()

    # ---- 请求队列（对应 orchestrator.py:631 _request_handler） ----

    async def _request_handler(self) -> None:
        while True:
            msg = await self.request_async_queue.get()
            if isinstance(msg, StageSubmissionMessage):
                await self._handle_add_request(msg)
            elif isinstance(msg, AbortRequestMessage):
                await self._handle_abort(msg)
            elif isinstance(msg, ShutdownRequestMessage):
                self._shutdown_event.set()
                break
            else:
                logger.warning("[Orchestrator] Unknown message type: %s", getattr(msg, "type", msg))

    async def _orchestration_output_handler(self) -> None:
        """轮询所有 stage pool，路由逻辑输出。对应 orchestrator.py:1154。"""
        while not self._shutdown_event.is_set():
            idle = True
            for stage_id in range(self.num_stages):
                pool = self.stage_pools[stage_id]
                for replica_id, raw in pool.poll_outputs():
                    await self._handle_processed_output(stage_id, replica_id, raw)
                    idle = False
            if idle:
                await asyncio.sleep(self.poll_interval)
            else:
                await asyncio.sleep(0)

    # ---- add_request（对应 orchestrator.py:3536 _handle_add_request） ----

    async def _handle_add_request(self, msg: StageSubmissionMessage) -> None:
        stage_id = msg.entry_stage_id
        request_id = msg.request_id
        if not msg.sampling_params_list:
            raise ValueError(f"Missing sampling params for stage 0. Got {len(msg.sampling_params_list)} stage params.")

        pool = self.stage_pools[stage_id]
        if not pool.live_replica_ids():
            await self._fail_request_dead_stage(request_id, stage_id)
            return

        final_stage_id = msg.final_stage_id
        final_output_stage_ids = set(msg.final_output_stage_ids or [final_stage_id])

        req_state = OrchestratorRequestState(
            request_id=request_id,
            prompt=msg.original_prompt if msg.original_prompt is not None else msg.prompt,
            sampling_params_list=msg.sampling_params_list,
            final_stage_id=final_stage_id,
            entry_stage_id=stage_id,
            final_output_stage_ids=final_output_stage_ids,
            request_timestamp=msg.request_timestamp or time.time(),
        )
        self.request_states[request_id] = req_state
        self._running_count += 1

        try:
            replica_id = await pool.submit_initial(
                request_id,
                req_state,
                msg.prompt,
                prompt_text=None,
            )
        except StageUnavailableError as e:
            logger.error("[Orchestrator] add_request for req=%s hit unavailable stage-%s: %s", request_id, stage_id, e)
            await self._fail_request_dead_stage(request_id, stage_id)
            return

        req_state.stage_submit_ts[stage_id] = time.time()
        logger.debug("[Orchestrator] req=%s submitted to stage-%s replica-%s", request_id, stage_id, replica_id)

    # ---- 输出路由（对应 orchestrator.py:2093 _route_output） ----

    async def _handle_processed_output(self, stage_id: int, replica_id: int, output: FakeStageOutput) -> None:
        req_state = self.request_states.get(output.request_id)
        if req_state is None:
            # 终态单调 + 取消后的迟到输出：直接丢弃（ORCH-INV-100）
            logger.warning(
                "[Orchestrator] Dropping output for unknown req %s at stage-%s (terminal or aborted)",
                output.request_id,
                stage_id,
            )
            return
        if req_state.terminal:
            logger.warning(
                "[Orchestrator] Dropping post-terminal output for req %s at stage-%s", output.request_id, stage_id
            )
            return

        if output.error is not None:
            await self._handle_stage_error(stage_id, output)
            return

        await self._route_output(stage_id, replica_id, output, req_state)

    async def _route_output(
        self,
        stage_id: int,
        replica_id: int,
        output: FakeStageOutput,
        req_state: OrchestratorRequestState,
    ) -> None:
        """路由一个 processed 输出：发给前端 和/或 转发给下一 stage。"""
        req_id = output.request_id
        finished = output.finished
        submit_ts = req_state.stage_submit_ts.get(stage_id)

        if finished:
            req_state.finished_stage_ids.add(stage_id)

        request_finished = False
        pool = self.stage_pools[stage_id]
        if finished and pool.final_output:
            req_state.finished_final_output_stage_ids.add(stage_id)
            final_output_stage_ids = req_state.final_output_stage_ids or {req_state.final_stage_id}
            request_finished = final_output_stage_ids.issubset(req_state.finished_final_output_stage_ids)

        if pool.final_output:
            message = OutputMessage(
                request_id=req_id,
                stage_id=stage_id,
                replica_id=replica_id,
                engine_outputs=output,
                finished=request_finished,
                stage_submit_ts=submit_ts,
            )
            if (
                request_finished
                and self.async_chunk
                and not set(req_state.stage_submit_ts).issubset(req_state.finished_stage_ids)
            ):
                # async_chunk 直投可能先于上游 terminal 输出到达：
                # 先扣住，等所有已提交 stage 都完成再发，避免前端拿到不完整 usage
                # （对应 orchestrator.py:2135-2146 pending_final_output）
                req_state.pending_final_output = message
                request_finished = False
            else:
                await self.output_async_queue.put(message)
        else:
            # 非终点 stage：只在有指标时发 metrics 消息（此处省略指标）
            pass

        # 跨 stage 转发：终点之前的 stage 在完成时把输出交给下一 stage
        # （对应 orchestrator.py:2169-2197 的转发条件）
        if finished and stage_id < req_state.final_stage_id and not req_state.terminal:
            await self._forward_to_next_stage(req_id, stage_id, output, req_state, src_replica_id=replica_id)

        # 扣住的 pending 输出：所有已提交 stage 完成后补发
        pending = req_state.pending_final_output
        if pending is not None and set(req_state.stage_submit_ts).issubset(req_state.finished_stage_ids):
            req_state.pending_final_output = None
            pending.finished = True
            await self.output_async_queue.put(pending)
            request_finished = True

        if request_finished:
            await self._cleanup_request_ids([req_id])

    # ---- 转发（对应 orchestrator.py:2683 _forward_to_next_stage） ----

    async def _forward_to_next_stage(
        self,
        req_id: str,
        src_stage_id: int,
        output: FakeStageOutput,
        req_state: OrchestratorRequestState,
        *,
        src_replica_id: int | None = None,
    ) -> None:
        """把 src 的输出转给下一 stage；stage 不可用只失败这一个请求。"""
        try:
            await self._forward_to_next_stage_unguarded(req_id, src_stage_id, output, req_state)
        except StageUnavailableError as e:
            logger.error(
                "[Orchestrator] inter-stage forward for req=%s hit unavailable stage-%s: %s",
                req_id,
                src_stage_id + 1,
                e,
            )
            await self._fail_request_dead_stage(req_id, src_stage_id + 1)

    async def _forward_to_next_stage_unguarded(
        self,
        req_id: str,
        src_stage_id: int,
        output: FakeStageOutput,
        req_state: OrchestratorRequestState,
    ) -> None:
        next_logical = src_stage_id + 1
        next_pool = self.stage_pools[next_logical]
        if not next_pool.live_replica_ids():
            await self._fail_request_dead_stage(req_id, next_logical)
            return

        # 幂等：下一 stage 已提交过就不再 submit（对应 _next_stage_already_submitted）
        already_submitted = (next_logical) in req_state.stage_submit_ts
        if already_submitted:
            return

        next_client = next_pool.stage_client
        source_outputs = [output]
        next_inputs: list[Any] = [output]
        if next_client is not None and next_client.process_engine_inputs is not None:
            # 对应上游 process_engine_inputs：由下一 stage 的 input processor
            # 决定消费哪些上游输出（可返回 [] 表示"再等等"）
            next_inputs = next_client.process_engine_inputs(source_outputs, req_state.prompt)
            if not next_inputs:
                logger.debug(
                    "[Orchestrator] req=%s stage-%s produced no inputs for stage-%s; waiting for more outputs",
                    req_id,
                    src_stage_id,
                    next_logical,
                )
                return

        for next_input in next_inputs:
            replica_id = await next_pool.submit_initial(
                req_id,
                req_state,
                next_input,
                prompt_text=None,
            )
            logger.debug(
                "[Orchestrator] req=%s forwarded stage-%s -> stage-%s replica-%s",
                req_id,
                src_stage_id,
                next_logical,
                replica_id,
            )
        req_state.stage_submit_ts[next_logical] = time.time()

    # ---- 失败与清理（对应 orchestrator.py:1530/1657/1852） ----

    async def _handle_stage_error(self, stage_id: int, output: FakeStageOutput) -> None:
        """stage 报错 → 前端可见错误 + 清理请求状态。"""
        await self.output_async_queue.put(
            ErrorMessage(
                error=output.error or "Stage request failed",
                request_id=output.request_id,
                stage_id=stage_id,
            )
        )
        await self._cleanup_request_ids([output.request_id], abort=True)

    async def _fail_request_dead_stage(self, req_id: str, stage_id: int) -> None:
        """stage 全副本不可用 → 只失败这个请求（服务器继续服务其他请求）。"""
        await self.output_async_queue.put(
            ErrorMessage(
                error=f"stage {stage_id} has no live replicas",
                request_id=req_id,
                stage_id=stage_id,
                fatal=False,
            )
        )
        await self._cleanup_request_ids([req_id], abort=True)

    async def _cleanup_request_ids(self, request_ids: list[str], *, abort: bool = False) -> None:
        """释放副本绑定与请求状态。所有收尾路径（完成/错误/abort）都汇到这里。

        对应 orchestrator.py:1852 _cleanup_request_ids（CFG 部分省略）。
        """
        if not request_ids:
            return
        cleanup_ids = list(dict.fromkeys(request_ids))
        if abort:
            for pool in self.stage_pools:
                await pool.abort_requests(cleanup_ids)
        for pool in self.stage_pools:
            pool.release_bindings(cleanup_ids)
        for request_id in cleanup_ids:
            req_state = self.request_states.pop(request_id, None)
            if req_state is not None:
                req_state.terminal = True  # ORCH-INV-100：终态单调
                if self._running_count > 0:
                    self._running_count -= 1

    # ---- abort（对应 orchestrator.py:723 _handle_abort） ----

    async def _handle_abort(self, msg: AbortRequestMessage) -> None:
        error: str | None = None
        try:
            # 先把 abort 下发到各 stage（保留绑定信息用于收集部分输出），
            # 再清理状态 —— 顺序与上游 _cleanup_request_ids(abort=True) 一致。
            await self._cleanup_request_ids(list(msg.request_ids), abort=True)
            logger.info("[Orchestrator] Aborted request(s) %s", msg.request_ids)
        except Exception as exc:  # noqa: BLE001 - 上游也吞掉并回 ack
            error = str(exc)
            logger.exception("[Orchestrator] Abort failed for request(s) %s", msg.request_ids)
        if msg.rpc_id is not None:
            # 有 rpc_id 时必须回执，保证 CorrelatedRpcClient 的等待者不挂死
            await self.rpc_async_queue.put(
                AbortResultMessage(rpc_id=msg.rpc_id, error=error, success=error is None)
            )
