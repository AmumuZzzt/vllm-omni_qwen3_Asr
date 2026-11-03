# SPDX-License-Identifier: Apache-2.0
"""手搓版 AsyncOmniEngine + AsyncOmni 入口（组合根）。

对照源码：
  - vllm_omni/engine/omni_engine_base.py   (OmniEngineBase：组合根 + 输出队列)
  - vllm_omni/engine/async_omni_engine.py  (AsyncOmniEngine：add_request_async)
  - vllm_omni/entrypoints/async_omni.py    (AsyncOmni.generate：面向用户的异步生成器)
  - vllm_omni/entrypoints/async_omni_base.py (_process_orchestrator_results：
                                              每请求队列 → yield OmniRequestOutput)

上游是"orchestrator 后台线程 + janus 双端队列"；这里简化成同一事件循环里的
双任务 + asyncio.Queue，但**消息种类、路由顺序、终态判定保持一致**。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from .orchestrator import (
    AbortRequestMessage,
    EngineQueueMessage,
    ErrorMessage,
    Orchestrator,
    OutputMessage,
    ShutdownRequestMessage,
    StagePool,
    StageSubmissionMessage,
)

logger = logging.getLogger("mini_omni.engine")

__all__ = ["MiniRequestOutput", "AsyncOmniEngine", "AsyncOmni"]


class MiniRequestOutput:
    """对应 OmniRequestOutput 的最小投影（前端 yield 的对象）。"""

    def __init__(
        self,
        request_id: str,
        stage_id: int,
        finished: bool,
        text: str = "",
        payload: Any = None,
        final_output_type: str = "text",
    ) -> None:
        self.request_id = request_id
        self.stage_id = stage_id
        self.finished = finished
        self.text = text
        self.payload = payload
        self.final_output_type = final_output_type

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"MiniRequestOutput(req={self.request_id!r}, stage={self.stage_id}, "
            f"finished={self.finished}, text={self.text!r})"
        )


class AsyncOmniEngine:
    """组合根：持有 orchestrator 任务 + 输出分发（对应 OmniEngineBase）。"""

    def __init__(self, stage_pools: list[StagePool], *, async_chunk: bool = False) -> None:
        self.stage_configs = stage_pools  # 每个 pool 即一个 stage 的运行配置视图
        self.request_queue: asyncio.Queue[EngineQueueMessage] = asyncio.Queue()
        self.output_queue: asyncio.Queue[EngineQueueMessage] = asyncio.Queue()
        self.rpc_queue: asyncio.Queue[EngineQueueMessage] = asyncio.Queue()
        self.orchestrator = Orchestrator(
            self.request_queue,
            self.output_queue,
            self.rpc_queue,
            stage_pools,
            async_chunk=async_chunk,
        )
        self._task: asyncio.Task | None = None
        # request_id -> 该请求的私有输出队列（对应 ClientRequestState.queue）
        self._request_queues: dict[str, asyncio.Queue[EngineQueueMessage]] = {}
        self._router_task: asyncio.Task | None = None

    # ---- 生命周期 ----

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.orchestrator.run(), name="orchestrator")
            self._router_task = asyncio.create_task(self._route_outputs(), name="output-router")

    async def shutdown(self) -> None:
        self.orchestrator.shutdown()
        await self.request_queue.put(ShutdownRequestMessage())
        for task in (self._task, self._router_task):
            if task is not None:
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    task.cancel()
        self._task = None
        self._router_task = None

    async def _route_outputs(self) -> None:
        """把 orchestrator 输出队列的消息分发到每请求私有队列。

        对应 async_omni_base.py:316 _final_output_loop。
        """
        while True:
            try:
                msg = await self.output_queue.get()
            except asyncio.CancelledError:
                raise
            request_id = getattr(msg, "request_id", None)
            if isinstance(msg, ErrorMessage) and request_id is None:
                # 无 request_id 的 fatal 错误 = 引擎级死亡
                if msg.fatal:
                    for q in self._request_queues.values():
                        await q.put(msg)
                continue
            if request_id is None:
                continue
            q = self._request_queues.get(request_id)
            if q is None:
                logger.warning("dropping message for unknown req %s", request_id)
                continue
            await q.put(msg)

    # ---- 提交 ----

    async def add_request_async(
        self,
        *,
        request_id: str,
        prompt: Any,
        sampling_params_list: Sequence[Any],
        final_stage_id: int,
        final_output_stage_ids: list[int] | None = None,
        entry_stage_id: int = 0,
        arrival_time: float | None = None,
    ) -> None:
        """提交请求到 orchestrator（对应 AsyncOmniEngine.add_request_async）。"""
        import time as _time

        self._request_queues[request_id] = asyncio.Queue()
        await self.request_queue.put(
            StageSubmissionMessage(
                request_id=request_id,
                prompt=prompt,
                original_prompt=prompt,
                sampling_params_list=list(sampling_params_list),
                final_stage_id=final_stage_id,
                request_timestamp=arrival_time if arrival_time is not None else _time.time(),
                final_output_stage_ids=final_output_stage_ids,
                entry_stage_id=entry_stage_id,
            )
        )

    async def abort_async(self, request_id: str) -> None:
        """等待 ack 的 abort（对应 OmniEngineBase.abort_async）。"""
        rpc_id = uuid.uuid4().hex
        await self.request_queue.put(AbortRequestMessage(request_ids=[request_id], rpc_id=rpc_id))
        while True:
            msg = await self.rpc_queue.get()
            if getattr(msg, "rpc_id", None) == rpc_id:
                if not getattr(msg, "success", True):
                    raise RuntimeError(msg.error or "abort failed")
                return

    def get_request_queue(self, request_id: str) -> asyncio.Queue[EngineQueueMessage]:
        return self._request_queues[request_id]


class AsyncOmni:
    """面向用户的离线/在线入口（对应 entrypoints/async_omni.py AsyncOmni）。"""

    def __init__(self, engine: AsyncOmniEngine) -> None:
        self.engine = engine

    @staticmethod
    def _get_unique_request_id(external_request_id: str) -> str:
        # 与上游一致：外部 id + 随机后缀，保证内部唯一（async_omni_base.py:232）
        return f"{external_request_id or 'req'}-{uuid.uuid4().hex[:12]}"

    async def generate(
        self,
        prompt: Any,
        sampling_params_list: Sequence[Any],
        *,
        request_id: str = "",
        final_stage_id: int = -1,
        final_output_stage_ids: list[int] | None = None,
        entry_stage_id: int = 0,
    ) -> AsyncGenerator[MiniRequestOutput, None]:
        """产出该请求的输出，直到 finished=True。

        对应 AsyncOmni.generate → _process_orchestrator_results
        （async_omni_base.py:246）：读每请求队列，yield 输出，
        收到 finished 的 OutputMessage 即 break。
        """
        external_request_id = request_id
        request_id = self._get_unique_request_id(external_request_id)

        await self.engine.add_request_async(
            request_id=request_id,
            prompt=prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            final_output_stage_ids=final_output_stage_ids,
            entry_stage_id=entry_stage_id,
        )
        queue = self.engine.get_request_queue(request_id)

        try:
            while True:
                result = await queue.get()

                if isinstance(result, ErrorMessage):
                    if result.fatal:
                        raise RuntimeError(f"engine dead: {result.error}")
                    raise RuntimeError(result.error)

                if not isinstance(result, OutputMessage):
                    logger.warning("dropping unexpected per-request message %r", result)
                    continue

                pool = self.engine.stage_configs[result.stage_id]
                output = result.engine_outputs
                yield MiniRequestOutput(
                    request_id=external_request_id or request_id,
                    stage_id=result.stage_id,
                    finished=result.finished,
                    text=getattr(output, "text", ""),
                    payload=getattr(output, "payload", None),
                    final_output_type=pool.final_output_type or "text",
                )

                if result.finished:
                    break
        finally:
            # 请求结束后释放前端侧状态（上游 _log_summary_and_cleanup 的等价物）
            self.engine._request_queues.pop(request_id, None)

    async def abort(self, request_id: str) -> None:
        await self.engine.abort_async(request_id)
