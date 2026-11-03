# vLLM-Omni 源码学习笔记

> 学习对象：`D:\pi-project\notes\vllm\vllm-omni-main\vllm-omni-main`（vllm-project/vllm-omni 主分支源码）
> 手搓产出：本目录 `mini_omni/`（纯 Python 可运行复现）+ `tests/`（50 个用例，全部通过）
>
> 运行方式：`D:\Py3.10\python.exe -m unittest discover -s tests`（退出码 0）

---

## 一、一句话理解这个项目

vLLM 只会"文本 → 文本"的自回归生成；**vLLM-Omni 把一次推理拆成多个 stage**（如
Thinker → Talker → Code2Wav），每个 stage 可以是 AR、非自回归生成、或扩散模型，
由一个 **Orchestrator** 统一做跨 stage 路由，用 **OmniConnector** 在 stage 之间搬运
tensor/KV，对外仍暴露 OpenAI 兼容 API。

核心分层（源码 `docs/design/architecture_overview.md`）：

```text
模型结构          What is computed（模型各组件怎么连）
    ↓
PipelineConfig    逻辑 stage 拓扑 + 执行类型 + 输入输出边
    ↓
Stage 运行策略    每 stage 自己的 batching / attention / 并行 / 量化
    ↓
DeployConfig      stage 放哪块卡、几个副本、用什么 connector（用户唯一要编辑的 yaml）
```

关键不变量（`docs/design/module/engine_orchestration.md`）：

| 编号 | 含义 |
|---|---|
| ORCH-INV-001 | **只有 Orchestrator 拥有跨 stage 路由权**；entrypoint 和 stage client 都不许自己转发 |
| ORCH-INV-002 | stage client 只做通信和生命周期，不选下一跳 |
| ORCH-INV-100 | **终态单调**：请求一旦到终态，不再为其转发任何新工作 |

---

## 二、源码地图（按调用链）

### 1. 启动/配置链

```text
vllm serve --omni
  └─ entrypoints/cli/serve.py            CLI 解析
      └─ config/resolver.py:resolve_omni_config()      ★ 生产解析唯一入口
          ├─ config/pipeline_registry.py               model_type → PipelineConfig 注册表
          │    └─ 各模型 model_executor/models/*/pipeline.py（拓扑声明，frozen）
          ├─ config/config_factory.py:StageConfigFactory.create_from_model()
          │    └─ 读 HF config → 推断 model_type → 查注册表（可按 hf_config 选变体）
          └─ config/stage_config.py
               ├─ DeployConfig(load_deploy_yaml)      deploy/*.yaml + base_config 继承
               ├─ _apply_platform_overrides()          cuda/npu/rocm/xpu 平台覆盖
               └─ merge_pipeline_deploy()              ★ Pipeline + Deploy → 每 stage 的 StageConfig
  └─ 返回 OmniConfigResolution{pipeline_config, stage_configs}（唯一启动交接对象）
```

### 2. 运行链（请求生命周期）

```text
entrypoints/async_omni.py:AsyncOmni.generate()          用户侧异步生成器
  ├─ 停顿/准入控制（pause/sleep/wake、admission slot）
  ├─ engine.add_request_async(...)                      组合根
  │    └─ 往 request_async_queue 投 StageSubmissionMessage
  └─ _process_orchestrator_results()                    读"每请求私有队列"逐条 yield
       └─ 收到 finished=True 的 OutputMessage 就 break

engine/async_omni_engine.py:AsyncOmniEngine             组合根（engine 与 orchestrator 的粘合）
engine/omni_engine_base.py:OmniEngineBase
  ├─ _initialize_stages()       按 stage_configs 启动各 stage 引擎进程
  ├─ _bootstrap_orchestrator()  起 Orchestrator 后台任务
  └─ get_outputs_async()        前端从输出总线取消息

engine/orchestrator.py:Orchestrator                     ★ 核心状态机（3793 行）
  ├─ run()                       双任务：request_handler + output_handler
  ├─ _request_handler()          消费 add_request / abort / shutdown / rpc
  ├─ _handle_add_request()       建 OrchestratorRequestState → stage0 提交
  ├─ _orchestration_loop()       轮询各 stage 副本输出（默认事件驱动变体）
  ├─ _route_output()             ★ 每条输出：发前端？转发下一 stage？请求结束？
  ├─ _forward_to_next_stage()    交给下一 stage（经 input processor 转换输入）
  ├─ _cleanup_request_ids()      所有收尾路径（完成/错误/abort）统一汇入
  └─ _handle_abort()             abort 下发 + rpc 回执

engine/stage_pool.py:StagePool                          一个逻辑 stage 的副本池
  ├─ select_replica_id()         绑定优先 + 轮转
  ├─ submit_initial/submit_update 提交请求（含输出处理器登记）
  └─ abort_requests()            把 abort 广播到各副本
```

### 3. 数据类型

| 类型 | 位置 | 作用 |
|---|---|---|
| `StageSubmissionMessage` | engine/messages.py | 入队消息：prompt、每 stage 的 sampling_params、final_stage_id |
| `OutputMessage` | engine/messages.py | 出队消息：某 stage 的输出 + finished 标志 |
| `ErrorMessage` | engine/messages.py | 请求级/引擎级错误（fatal 区分） |
| `OmniRequestOutput` | outputs/__init__.py | 继承 vLLM `RequestOutput`，加 stage_id/final_output_type/多模态字段 |
| `OrchestratorRequestState` | engine/orchestrator.py:239 | 每请求编排状态（见下） |

### 4. `OrchestratorRequestState` 的关键字段（读源码必看）

```python
final_stage_id                 # 请求的逻辑终点 stage
final_output_stage_ids         # 哪些 stage 完成才算"请求完成"（可多个，如 text+audio）
finished_final_output_stage_ids# 其中已完成的子集 → 决定 request_finished
finished_stage_ids             # 所有已回报完成的 stage（含非终点）
stage_submit_ts                # 每 stage 提交时间；key 存在 == 该 stage 已提交
pending_final_output           # async_chunk 直投先于上游 terminal 到达时的"扣住"缓冲
```

**请求完成的判定**（`_route_output`，orchestrator.py:2117）：

```python
request_finished = final_output_stage_ids.issubset(finished_final_output_stage_ids)
```

多终点输出（文本 + 音频同时要）就靠这个 subset 判定。

---

## 三、async_chunk：为什么快

`docs/design/feature/async_chunk.md`：顺序模式下必须等上游**整段**输出才提交下游，
TTFP ≈ 各 stage 之和；async_chunk 模式下**每个 chunk 一产生就交给下游**，stage 重叠执行：

```text
顺序:    [Thinker ████████████][Talker ████████████][Code2Wav ████]   TTFP=6.0s
async:   [Thinker ████░░████░░████]
              [Talker ░░████░░████]        TTFP≈0.5s（实测 TTFP 降 ~92%）
                  [Code2Wav ░░░░████]
```

实现要点（源码）：

1. **拓扑声明**：`StagePipelineConfig.async_chunk_process_next_stage_input_func`
   是"chunk 模式下的下一 stage 输入处理器"；与 `custom_process_next_stage_input_func`
   （整段模式）成对存在，`_select_processor_funcs()`（stage_config.py:1302）按开关二选一。
2. **开关解析**：`_resolve_pipeline_async_chunk_enabled()`（stage_config.py:1463）——
   单 stage 模型不支持（除非 `single_stage_async_chunk`）；多 stage 需要"有跨 stage 边 +
   有 async 处理函数"；与原生 AR→DiT KV 传输互斥。
3. **边一致性**：`validate_stage_async_chunk_edges()`——同一条 connector 边两端的
   async_chunk 必须一致，否则启动即报错。
4. **扣住的终态**：`_route_output` 里 `pending_final_output`——async_chunk 直投可能让
   下游先跑完，此时不立刻给前端发 finished，等所有已提交 stage 都完成再发，
   避免 usage/stop_reason 不完整（orchestrator.py:2135-2146）。
5. **连接器参数**：deploy yaml `connectors.extra.codec_chunk_frames=25`、
   `initial_codec_chunk_frames=4`（动态初始 chunk 压 TTFP）。

---

## 四、手搓复现（mini_omni）

### 4.1 对照表

| 手搓文件 | 上游源码 | 保留的核心 |
|---|---|---|
| `mini_omni/config.py` | `config/stage_config.py` + `pipeline_registry.py` + `resolver.py` | PipelineConfig 拓扑校验（6 条错误规则逐条对齐）、注册表+resolver、deploy yaml `base_config` 继承与按 stage_id 深合并、平台覆盖、async_chunk 决策与边校验、`merge_pipeline_deploy`、CLI `stage_<id>_*` override 归一化、`resolve_omni_config` |
| `mini_omni/orchestrator.py` | `engine/orchestrator.py` + `stage_pool.py` + `messages.py` | 双任务 run()、`OrchestratorRequestState`、`_route_output`（final_output 判定 / pending_final_output 扣发 / subset 终态）、`_forward_to_next_stage`（幂等提交、processor 返回空则等待）、统一 `_cleanup_request_ids`、abort+rpc 回执、终态单调丢弃、StagePool 轮转选副本+绑定粘性 |
| `mini_omni/engine.py` | `engine/omni_engine_base.py` + `entrypoints/async_omni*.py` | 组合根、输出总线按 request_id 分发到每请求私有队列、`AsyncOmni.generate()` 逐条 yield 到 finished |

### 4.2 有意简化（避免以假乱真）

* stage 引擎用 `FakeStageClient`（按脚本吐输出）替代真实 EngineCore 进程/调度器/KV；
* 单进程 asyncio 替代"后台线程 + janus 双端队列"；
* 省略 CFG companion、PD 分离、duplex 会话、first-audio 排序、多模态缓存、指标。

### 4.3 测试覆盖（50 个，`tests/test_mini_omni.py`）

**配置层**：拓扑 6 种非法形态全部拒绝；注册表 resolver 按 hf_config 选变体、类型不匹配返回 None；
deploy `base_config` 继承 + 薄 overlay 深合并不丢兄弟 key；`stage_args` 旧 schema 拒绝；
平台覆盖字段三路（已有字段 setattr / 未知字段进 engine_extras / devices 单独处理）；
`model_runner` v2→v1 回退清理 stage 级 v2；async_chunk 自动探测/单 stage 警告关闭/stage 级 opt-out/
边不一致报错；真实 `qwen3_omni_moe.yaml` 三 stage 合并产物（worker_type、scheduler、采样参数并集、
连接器接线）；CLI override 按 stage 隔离 + 别名翻译 + 冲突警告 + 越界拒绝。

**编排层**：三 stage 端到端路由；双终点（text+audio）subset 判定、唯一终态；
终态后迟到输出丢弃（ORCH-INV-100）、未知请求输出丢弃、清理计数归零；
入口 stage 死 → 只失败该请求；转发时下游死 → 只失败该请求；abort 状态清理+rpc 回执+
各 stage 下发；processor 返回空 → 等待不提交；副本轮转/绑定粘性/无副本抛错。

### 4.4 已知差距（下一步可继续手搓）

1. **async chunk 数据面**：现在只复现了"开关与处理器选择"，未实现 chunk 级
   connector 传输（`stage_output_payload_keys`/`OmniConnector`）与 prewarm；
2. **事件驱动输出循环**：上游默认 `_orchestration_loop_event_driven`（每副本 reader task
   + 单 dispatch 队列），手搓版只有 1ms 轮询版；
3. **StageRuntime 启动规划**（`ReplicaInitPlan`、设备锁、进程管理）完全未涉及；
4. **stage 输入处理器真实逻辑**（`stage_input_processors/qwen3_omni.py` 的
   thinker2talker/talker2code2wav，hidden state 切片、codec 行增量）——这是理解
   Qwen3-Omni 数据流的下一站。

---

## 五、阅读顺序建议（给后来的自己）

1. `docs/design/architecture_overview.md`（架构 + 四层配置模型）
2. `vllm_omni/model_executor/models/qwen3_omni/pipeline.py`（131 行，拓扑声明全貌）
3. `vllm_omni/config/stage_config.py` 的 `PipelineConfig.get_validation_errors`
   → `resolve_deploy_yaml` → `merge_pipeline_deploy`（配置主干）
4. `vllm_omni/engine/orchestrator.py` 的 `OrchestratorRequestState` → `_handle_add_request`
   → `_route_output` → `_forward_to_next_stage` → `_cleanup_request_ids`（编排主干，约 600 行）
5. `docs/design/feature/async_chunk.md` + 对应实现（性能主线）
6. 对照本目录 `mini_omni/` 跑测试，验证自己能脱离源码复述每个分支的语义
