#!/usr/bin/env python
"""Inspect what the *runtime* stage config carries after the lab pipeline loads.

Why a real file: Omni stage replicas use multiprocessing ``spawn``, which
re-executes ``__main__`` by path. A heredoc/stdin script has no path
(``<stdin>``) and every stage init dies in ``_fixup_main_from_path``.
"""

import sys

MODEL = "/opt/qwen3-asr/models/Qwen3-ASR-0.6B"


def main() -> int:
    from vllm_omni.entrypoints.cli.serve import cmd_init

    cmd_init()  # registers the lab qwen3_asr pipeline

    from vllm_omni.entrypoints.async_omni import AsyncOmni

    engine = AsyncOmni(model=MODEL, trust_remote_code=True, init_timeout=300)
    try:
        sc = engine.engine.stage_configs[0]
        print("stage_config type   :", type(sc).__name__, flush=True)
        print("has yaml_extras     :", hasattr(sc, "yaml_extras"), flush=True)
        print("yaml_extras         :", getattr(sc, "yaml_extras", "N/A"), flush=True)
        print("is_comprehension    :", getattr(sc, "is_comprehension", "N/A"), flush=True)
        print("supported_tasks     :", engine.engine.supported_tasks, flush=True)
        print("transcription in it :", "transcription" in engine.engine.supported_tasks, flush=True)
    finally:
        engine.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
