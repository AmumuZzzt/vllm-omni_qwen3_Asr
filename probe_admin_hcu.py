#!/usr/bin/env python
"""Read-only environment probe for the admin-hcu-dev host.

The password is taken from the ADMIN_PW environment variable and is never
written to disk or printed. Only read-only shell commands are executed.
"""

import os
import sys

import paramiko

HOST = "172.16.16.60"
USER = "root"
CWD = "/mnt/admin/zt/vllm_omni_env"

REMOTE = r"""
echo "===== 1. 主机 ====="
hostname; whoami; pwd
echo "--- uname ---"; uname -a
echo "--- os ---"; cat /etc/os-release 2>/dev/null | grep -E "^(PRETTY_NAME|VERSION_ID)" 

echo
echo "===== 2. 目标目录内容 ====="
ls -la /mnt/admin/zt/vllm_omni_env 2>&1 | head -30

echo
echo "===== 3. 是否有 venv ====="
if [ -f /mnt/admin/zt/vllm_omni_env/bin/activate ]; then
  echo "VENV: /mnt/admin/zt/vllm_omni_env/bin/activate 存在"
  PY=/mnt/admin/zt/vllm_omni_env/bin/python
else
  echo "VENV: 目标目录下没有 activate"
  ls -d /mnt/admin/zt/vllm_omni_env/*venv* 2>/dev/null
  PY=$(which python3)
fi
echo "PY=$PY"
$PY -V 2>&1

echo
echo "===== 4. 包版本 ====="
$PY - <<'PYEOF' 2>/dev/null
import importlib.metadata as m
for p in ("vllm", "vllm-omni", "torch", "transformers", "triton"):
    try:
        print(f"  {p:14s}: {m.version(p)}")
    except Exception:
        print(f"  {p:14s}: MISSING")
PYEOF

echo
echo "===== 5. vllm_omni 来源（editable 还是 wheel）====="
$PY -c "import vllm_omni,os;print('  path:',os.path.dirname(vllm_omni.__file__))" 2>&1 | tail -3
ls -d /mnt/admin/zt/vllm_omni_env/lib/python*/site-packages/vllm_omni*.dist-info 2>/dev/null | head -2
cat /mnt/admin/zt/vllm_omni_env/lib/python*/site-packages/vllm_omni*.dist-info/direct_url.json 2>/dev/null

echo
echo "===== 6. GPU ====="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv 2>&1 | head -8

echo
echo "===== 7. CUDA ====="
ls -l /usr/local/cuda 2>/dev/null | head -1
cat /usr/local/cuda/version.json 2>/dev/null | grep -o '"version" : "[^"]*"' | head -1

echo
echo "===== 8. 磁盘 ====="
df -h / /mnt/admin/zt 2>/dev/null | head -5

echo
echo "===== 9. 模型 ====="
ls -d /opt/qwen3-asr/models/* 2>/dev/null || echo "  (/opt/qwen3-asr/models 不存在)"
find /mnt/admin/zt -maxdepth 3 -type d -iname "*Qwen3-ASR*" 2>/dev/null | head -5

echo
echo "===== 10. 关键源码文件 ====="
SP=$($PY -c "import vllm_omni,os;print(os.path.dirname(vllm_omni.__file__))" 2>/dev/null)
echo "site-packages vllm_omni = $SP"
ls -l "$SP/utils/forced_aligner.py" 2>/dev/null || echo "  forced_aligner.py: 不存在"
ls -l "$SP/engine/async_omni_engine.py" 2>/dev/null | head -2

echo
echo "===== 11. git 仓库？ ====="
for d in /mnt/admin/zt/vllm_omni_env /mnt/admin/zt/vllm_omni_env/vllm-omni /mnt/admin/zt; do
  if [ -d "$d/.git" ]; then
    echo "  repo: $d"
    echo -n "    branch: "; git -C "$d" rev-parse --abbrev-ref HEAD 2>/dev/null
    echo -n "    HEAD  : "; git -C "$d" log -1 --oneline 2>/dev/null
    echo -n "    dirty : "; git -C "$d" status --porcelain 2>/dev/null | wc -l
  fi
done
"""


def main() -> int:
    pw = os.environ.get("ADMIN_PW")
    if not pw:
        print("ADMIN_PW not set", file=sys.stderr)
        return 2

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            HOST, port=22, username=USER, password=pw,
            timeout=25, banner_timeout=25, auth_timeout=25,
            look_for_keys=False, allow_agent=False,
        )
    except Exception as exc:
        print(f"CONNECT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    try:
        stdin, stdout, stderr = client.exec_command(
            f"cd {CWD} 2>/dev/null; " + REMOTE, timeout=120
        )
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        print(out)
        if err.strip():
            print("---- STDERR ----")
            print(err[:3000])
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
