#!/usr/bin/env python
"""Insert the 2026-11-02 lab operations into the standalone notebook SEED.

The notebook keeps notes in localStorage; SEED is only read when that storage is
empty. So we do two things:
  1. append the new note to ``SEED.notes`` in notebook.html (fresh browsers), and
  2. emit a matching importable JSON (for browsers that already have state).
"""

from __future__ import annotations

import json
import pathlib

SITE = pathlib.Path(r"D:\pi-project\notes\vllm\omni-learning-site")
HTML = SITE / "notebook.html"
JSON_OUT = SITE / "omni-notebook-lab-2026-11-02.json"

ANCHOR = """          updatedAt: '2026-10-31T02:50:00.000Z'
        }
      ]
    };"""


def rec(rid, chapter, section, stage, label, command, result, status, lesson,
        metrics="", duration="", links="", created="2026-11-02T15:00:00.000Z"):
    return {
        "id": rid,
        "chapter": chapter,
        "section": section,
        "stage": stage,
        "label": label,
        "duration": duration,
        "command": command,
        "result": result,
        "status": status,
        "lesson": lesson,
        "metrics": metrics,
        "links": links,
        "createdAt": created,
    }


CH1 = "一、0.32 参考实现：缺口定位"
CH2 = "二、版本锚点核对"
CH3 = "三、0.26 隔离环境搭建"
CH4 = "四、复现验证"

records = [
    rec(
        "lab026-version-recon", CH1, "只读勘察", "核验",
        "0.32 参考环境的版本与能力探测",
        '/opt/qwen3-asr/venv/bin/python -c "import vllm, vllm_omni, torch; print(\'vllm\', vllm.__version__); print(\'omni\', vllm_omni.__version__); print(\'torch\', torch.__version__); print(\'cap\', torch.cuda.get_device_capability(0))"',
        "vllm 0.32.0\nomni 0.32.0\ntorch 2.13.0+cu130\ncuda True / NVIDIA GeForce RTX 4090\ncap (8, 9)",
        "成功",
        "0.32 的能力探测只看 stage 拓扑标记（owns_tokenizer、final_output_type），结构上不会产出 transcription，因此转录路由不注册。",
        metrics="venv 9.0 G\nGPU 24564 MiB",
    ),
    rec(
        "lab026-precheck", CH1, "能力探测", "验证",
        "out-of-tree qwen3_asr pipeline 冒烟",
        "cd /root/vllm-work/lab2 && python lab2_precheck_stage.py",
        "[precheck] registered pipeline 'qwen3_asr'\n[precheck] RESULT: engine constructed OK\n[precheck] supported_tasks      = ('generate',)\n[precheck] num_stages           = 1\n[precheck] stage0: type=StageType.LLM is_comprehension=True final_output=True final_output_type=text\n[precheck] 'transcription' advertised = False\n[precheck] engine shut down",
        "成功",
        "Qwen3ASRForConditionalGeneration 可以被 Omni stage 正常驱动；断点唯一且明确地落在 capability probe，而不是模型适配层。",
        metrics="torch.compile 26.19 s",
    ),
    rec(
        "lab026-ninja", CH1, "排障", "部署",
        "EngineCore 启动失败的真实根因是 ninja 缺失",
        "# 首次运行 precheck，失败后读子进程日志\nsed -n '60,175p' precheck_run1.log",
        "flashinfer/jit/core.py -> run_ninja -> FileNotFoundError: [Errno 2] No such file or directory: 'ninja'\n\n修复后实测：\nfirst build+load: 71.2s\nsecond call:      0.000s (cache hit, same object)",
        "成功",
        "现象是 Engine core initialization failed，根因埋在子进程 traceback 里；FlashInfer 采样器首次要 JIT 编译，71 s 成本摊进首批请求，正是历史吞吐差一个数量级的来源。",
        metrics="JIT 首次 71.2 s\n命中缓存 0.000 s",
    ),
    rec(
        "lab026-anchor", CH2, "锚点校验", "核验",
        "v0.32.0 标签与线上 wheel 逐文件比对",
        "diff -rq /opt/qwen3-asr/venv/lib/python3.12/site-packages/vllm_omni /root/vllm-work/src/vllm-omni/vllm_omni | grep -v __pycache__",
        "Only in .../site-packages/vllm_omni: _version.py\nOnly in ./vllm_omni/<多个目录>: README.md\n\n（Python 源码无差异）",
        "成功",
        "线上运行的就是 tag v0.32.0 的源码；唯一差异是 wheel 构建生成的 _version.py 与仓库里带、wheel 排除的 README。实验地基可复现。",
    ),
    rec(
        "lab026-patchid", CH2, "等价判定", "核验",
        "study 分支 HEAD 与上游 21495f9e 的 stable patch-id 比对",
        "git describe --tags HEAD\ngit show 21495f9e | git patch-id --stable\ngit diff 1c4aca7f..HEAD | git patch-id --stable\ngit diff --check 1c4aca7f..HEAD",
        "v0.26.0-12-g18649cf7\nupstream 21495f9e          : fc29c9c67459312565d21cbb61f3def5af5195f5\nHEAD (BASE..HEAD aggregate): fc29c9c67459312565d21cbb61f3def5af5195f5\nHEAD (single commit)       : fc29c9c67459312565d21cbb61f3def5af5195f5\nRESULT: MATCH\ngit diff --check: exit=0",
        "成功",
        "内容等价但 commit 对象不同：HEAD 18649cf7 是把 21495f9e 重放到 base 1c4aca7f 之上。base 是 v0.26.0 之后 12 个提交，不是标签本身，所以必须配 vLLM 0.26。",
        metrics="BASE..HEAD = 1 commit\nUP..HEAD = 10 commits",
    ),
    rec(
        "lab026-pip-slow", CH3, "安装方式", "尝试",
        "pip 直装 vllm==0.26.0：60 分钟仅 4.84 GB",
        "python -m pip install vllm==0.26.0   # 已配置清华源",
        "进程 52 分钟时 CPU 1.6%，/proc/1108/io write_bytes ≈ 4.84 GB，venv 仍为 12 M\n下载暂存于 /tmp/pip-unpack-*：torch 506 MB、vllm 290 MB、triton 272 MB、十余个 nvidia_* 包",
        "中断",
        "pip 单连接串行且最后才写入 site-packages，所以中途看起来毫无进展。整套 CUDA 栈体量 7–9 GB，是这个量级决定了耗时。",
        metrics="约 0.64 MB/s\n4.84 GB / 60 min",
    ),
    rec(
        "lab026-mirror", CH3, "换源决策", "验证",
        "镜像吞吐实测（各取 8 MiB range）",
        "/opt/qwen3-asr/venv/bin/python /tmp/mirror_probe.py",
        "tuna      14.13 MiB/s   (8.0 MiB in 0.6s)\naliyun     0.69 MiB/s   (8.0 MiB in 11.6s)\nustc      12.14 MiB/s   (8.0 MiB in 0.7s)\ntencent    1.74 MiB/s   (8.0 MiB in 4.6s)\n\n（半小时后复测：tencent 20.69 MiB/s，uv 实测持续约 47 MB/s）",
        "成功",
        "突发值不能代表持续吞吐：tencent 首测 1.74、复测 20.69。此外 IP 属阿里云不等于阿里云源最快，实测它排最后。",
    ),
    rec(
        "lab026-uv-install", CH3, "核心安装", "部署",
        "改用 uv 并行安装 vllm 0.26.0",
        "uv pip install --python /opt/qwen3-asr/venv-omni-026/bin/python \\\n  --index-url https://mirrors.cloud.tencent.com/pypi/simple \\\n  vllm==0.26.0",
        "Resolved 197 packages in 5.74s\n（清华源阶段约 5 MB/s，切换腾讯源后约 47 MB/s 收尾）\n\n结果：vllm 0.26.0 / torch 2.11.0+cu130 / transformers 5.22.1 / triton 3.6.0\nCUDA True / NVIDIA GeForce RTX 4090\nvenv 7.5 G",
        "成功",
        "换并行下载 + 换源把同一份 7–9 GB 从 60 分钟压到约 20 分钟。装完必须实测 torch.cuda.is_available()，不能只看 pip 成功。",
        metrics="197 packages\nvenv 8.0 G",
    ),
    rec(
        "lab026-editable", CH3, "挂载源码", "部署",
        "editable 挂载 study 分支（--no-deps）",
        "uv pip install --python /opt/qwen3-asr/venv-omni-026/bin/python \\\n  --no-deps -e /opt/qwen3-asr/source-study",
        "+ vllm-omni==0.26.1.dev12+g18649cf74  (from file:///opt/qwen3-asr/source-study)\n\n随后 import 报错：\nModuleNotFoundError: No module named 'aenum'",
        "部分成功",
        "版本号与 git describe 的 v0.26.0-12-g18649cf7 对应，major.minor=(0,26) 与 vllm 对齐。但该树 [project].dependencies 为空，--no-deps 后必须按 requirements/common.txt 补依赖。",
    ),
    rec(
        "lab026-common-deps", CH3, "依赖补齐", "部署",
        "按项目 requirements/common.txt 补齐依赖",
        "uv pip install --python /opt/qwen3-asr/venv-omni-026/bin/python \\\n  -r /opt/qwen3-asr/source-study/requirements/common.txt",
        "+ av==19.0.2  + cache-dit==1.3.0  + diffusers==0.38.0  + janus==2.0.0\n+ msgpack==1.2.4  + omegaconf==2.4.0  + soundfile==0.14.0  + scipy==1.18.1\n+ openai-whisper==20250625  + x-transformers==2.31.11  + prettytable==3.18.0  …\n- huggingface-hub==2.4.0\n+ huggingface-hub==1.33.0",
        "成功",
        "用的是项目自己的 common.txt，不是我猜的清单，便于复现。代价是 huggingface-hub 被降级 2.4.0 → 1.33.0；import 全通，但这是对已有依赖的实际改动，需记录以便回滚。",
    ),
    rec(
        "lab026-pytest", CH4, "测试", "验证",
        "运行 21495f9e 自带的 capability probe 测试",
        "cd /opt/qwen3-asr/source-study\n/opt/qwen3-asr/venv-omni-026/bin/python -m pytest \\\n  tests/engine/test_transcription_capability.py -v --no-header -p no:cacheprovider",
        "collected 21 items\n… 21 passed, 14 warnings in 2.74s\n=== pytest exit=0 ===\n\n（git status 无输出；未生成 .pytest_cache）",
        "成功",
        "测试把上游语义钉死了：探测委托 vLLM 的 supports_transcription 谓词、且目标 stage 必须是文本终态。这证明我此前在 0.32 上写的 extras 声明方案与上游机制不等价，不能算对 #5714 的复现。",
        metrics="21 passed / 0 failed / 0 skipped\n2.74 s",
    ),
]

note = {
    "id": "seed-lab-026-repro",
    "title": "vLLM 0.26 学习环境与 #5714 capability probe 复现",
    "body": (
        "目标：把 vLLM-Omni 固定到与上游 21495f9e 匹配的版本，在隔离环境里复现 #5714 的 capability probe，并跑通它自带的测试。\n\n"
        "隔离方式：新建 /opt/qwen3-asr/venv-omni-026，原 /opt/qwen3-asr/venv（0.32.0）保持只读参考；源码用 study 分支，通过 editable 安装生效。\n\n"
        "版本口径：study 分支 base 是 1c4aca7f（v0.26.0 之后 12 个提交），所以必须配 vLLM 0.26，不能与 0.32 混用。omni 的 version.py 会检查 major.minor 是否一致。\n\n"
        "重要更正：此前在 0.32 上用 extras 声明让转录路由通了的做法，与上游「委托 supports_transcription 谓词 + 要求文本终态」的机制不同，不作为复现成果。\n\n"
        "所有命令与输出均来自实际执行；估算值已标注为估算。"
    ),
    "tags": ["ASR", "vLLM-Omni", "0.26", "复现", "环境隔离"],
    "run": {
        "objective": "在隔离环境中把 vLLM-Omni 固定在 v0.26 line（= 上游 21495f9e），复现 #5714 capability probe 并跑通其 21 个测试。",
        "environment": "RTX 4090 24 GB · driver 595.80 · venv-omni-026（Python 3.12.13）· 根盘剩 28 G",
        "model": "Qwen/Qwen3-ASR-0.6B（本地 /opt/qwen3-asr/models，未重新下载）",
        "workload": "本阶段不做推理或压测；只做版本核对、依赖安装与单元测试",
        "conclusion": "环境就绪：vllm 0.26.0 + vllm_omni 0.26.1.dev12+g18649cf74，21 tests passed；确认 0.32 上的自写补丁与上游机制不等价。",
        "next": "前进到下一个提交 a98d18bc（standalone qwen3_asr pipeline + deploy profile），对照自己的 pipeline 定义与上游差异。",
    },
    "records": records,
    "annotations": [
        {
            "id": "lab026-note-correction",
            "target": "record:lab026-pytest:result",
            "quote": "21 passed",
            "comment": "这一行是分界线：测试证明了上游的判定口径，也证明我之前那套 extras 方案只是让接口能通，不是同一套机制。学习记录里必须区分「能跑」和「复现了对的实现」。",
            "createdAt": "2026-11-02T15:05:00.000Z",
        }
    ],
    "experimentSort": "custom",
    "createdAt": "2026-11-02T15:00:00.000Z",
    "updatedAt": "2026-11-02T15:10:00.000Z",
}


def main() -> int:
    html = HTML.read_text(encoding="utf-8")
    if note["id"] in html:
        print("note already present, aborting")
        return 1
    if ANCHOR not in html:
        print("anchor not found, aborting")
        return 1

    block = json.dumps(note, ensure_ascii=False, indent=2)
    block = "\n".join("        " + line for line in block.splitlines())
    replacement = ANCHOR.replace(
        "        }\n      ]", "        },\n" + block + "\n      ]", 1
    )
    html = html.replace(ANCHOR, replacement, 1)
    HTML.write_text(html, encoding="utf-8")
    print(f"inserted note into SEED ({len(records)} records)")

    payload = {
        "format": "omni-notebook-standalone",
        "version": 1,
        "exportedAt": "2026-11-02T15:10:00.000Z",
        "notes": [note],
    }
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {JSON_OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
