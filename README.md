# vllm 实验工作区（本地）

本地目录：`D:\codex_project\vllm`　|　远程实验主机：`vllm4090` = `117.50.218.154:23 (root)`

> 凭据在 [`LOCAL-ONLY-CREDENTIALS.md`](LOCAL-ONLY-CREDENTIALS.md)（已 gitignore）。**绝不上传、不提交、不拷贝到服务器。**
> 私钥：`C:\Users\Godzilla\.ssh\vllm_4090`

---

## 1. 连接（已配置免密，日常不需要密码）

```powershell
ssh vllm4090                       # 交互登录
ssh -o BatchMode=yes vllm4090 "hostname"   # 单条命令
```

SSH 配置块（`C:\Users\Godzilla\.ssh\config`）：

```sshconfig
Host vllm4090
    HostName 117.50.218.154
    Port 23
    User root
    IdentityFile C:/Users/Godzilla/.ssh/vllm_4090
    IdentitiesOnly yes
    PreferredAuthentications publickey
    ServerAliveInterval 30
    ServerAliveCountMax 6
    StrictHostKeyChecking accept-new
```

密码登录仍可用（应急）：`root` / 见凭据文件。若换了机器要重建免密：

```powershell
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\vllm_4090 -C dsh-agent
# 把 .pub 内容追加到服务器 /root/.ssh/authorized_keys（chmod 600）
```

## 2. 传输文件

```powershell
# 本机 -> 远端
scp -P 23 .\script.py vllm4090:/root/vllm-work/
# 远端 -> 本机
scp -P 23 vllm4090:/root/vllm-work/out.json .\
```

`scp` 必须走别名 `vllm4090` 或显式 `-P 23 root@...`；直接 `root@117.50.218.154` 不带 `-P` 会连 22 端口失败/挂起。

## 3. 远程执行（推荐用 `remote.ps1`，避开引号地狱）

Windows OpenSSH 8.1 对嵌套引号很脆弱，且 PowerShell here-string 是 CRLF，远端 bash 会把
`/root/vllm-work` 读成 `/root/vllm-work\r` 而报“无此目录”。本目录的 [`remote.ps1`](remote.ps1)
已处理这两点（LF 规整 + UTF-8 读取 + 进程管道传 stdin）：

```powershell
cd D:\codex_project\vllm
.\remote.ps1 -Command 'nvidia-smi --query-gpu=memory.used --format=csv'
.\remote.ps1 -File .\scripts\bench.sh          # 长脚本走文件，最保险
'df -h /model' | .\remote.ps1                  # 管道也行
```

手工方式等价于（注意必须 LF）：

```powershell
@'
export PATH=/opt/qwen3-asr/venv/bin:$PATH
nvidia-smi --query-gpu=memory.used --format=csv
'@ -replace "`r`n","`n" | ssh vllm4090 "bash -s"
```

注意：`bash -s` 是非交互、非登录 shell，**不读 `.bashrc`** —— 每次显式
`export PATH=/opt/qwen3-asr/venv/bin:$PATH`，或用 `bash -lc`。

## 4. 服务器环境（2026-11-02 实测）

| 项 | 值 |
|---|---|
| 系统 | Ubuntu 22.04.4 LTS, kernel 5.15.0-113 |
| CPU / RAM | 16 vCPU / 62 GB（无 swap） |
| GPU | **RTX 4090 24 GB**（sm_89），驱动 595.80 |
| CUDA | `/usr/local/cuda` -> `cuda-12.6`（12.4 也在） |
| 磁盘 | `/` overlay 79 G（可用 ~43 G）；`/model` 40 T 共享存储（13 T 可用） |
| 时区 | UTC（本地东八区，注意日志时间差 8h） |
| 网络 | huggingface.co 200、pypi 200，可直连 |
| 工作目录 | `/root/vllm-work`（已建，实验用） |

### Python 环境

| 环境 | 说明 |
|---|---|
| `/opt/qwen3-asr/venv` | **现成可用**：Python 3.12.13，vllm 0.32.0，torch 2.13.0+cu130，transformers 5.19.0，vllm-omni 0.32.0 |
| `/usr/local/miniconda3` | base 为 Python 3.13.5，未装 vllm |
| `/root/vllm-learning/vllm` | vllm 源码 checkout（1.3 G，HEAD `89a9a03710`，remote github vllm-project/vllm） |

vllm 已装且 `torch.cuda.is_available() == True`，**无需重复安装**。新实验优先复用 `/opt/qwen3-asr/venv`，避免 43 G 磁盘被多份 torch 吃光：

```bash
/opt/qwen3-asr/venv/bin/vllm serve /model/.../model --port 8000 ...
```

### 模型缓存

- `/model/HuggingFace`（11 T）、`/model/ModelScope`（13 T）—— 大量已下载模型
- 服务器 `.bashrc` 已设 `USE_MODELSCOPE_HUB=1`
- 本机缓存：`/root/.cache/huggingface` 约 15 G

## 5. 既有实验痕迹（Qwen3-ASR 基准）

`/root/vllm-learning/` 下有之前的 ASR 压测：`asr_zh.wav`、`asr_en.wav`、`csgo.wav`，
以及 `openai-audio-infqps-concurrency{8,16,32,64}-Qwen3-ASR-0.6B-*.json` 结果文件。
对应服务命令见 `/root/.bash_history`。

## 6. 注意事项

- **显存只有 24 G**：7B 模型 bf16 约 14 G 权重，`--gpu-memory-utilization` 建议 0.85~0.9，长上下文慎开；32B 需量化。
- **根盘只剩 43 G**：不要 `pip install` 多套 vllm/torch；模型放大文件放 `/model`。
- **无 swap**：大模型编译/量化容易 OOM killer，注意 `dmesg`。
- 后台长任务：服务器**未装** tmux/screen（只有 `nohup`/`setsid`）。要么 `nohup cmd > log 2>&1 &`，要么先 `apt-get install -y tmux` 装上 tmux 再跑。
- `nvidia-smi --format` 在这台机器上只吃 `csv`，**不支持 `csv,noheader`/`key=value` 修饰符**（会报 `Format modifier is not recognized`），过滤表头用 `| tail -n +2`。
- 时区 UTC，报告时间需 +8h 换算。

## 7. 目录约定

| 路径 | 用途 |
|---|---|
| `/root/vllm-work` | 新实验工作区（已建，含 `.inbox/` 接收上传文件） |
| `/root/vllm-work/.inbox` | 本机 `scp` 落点 |
| `/root/vllm-learning` | 旧 ASR 实验（勿动） |
| `/model/HuggingFace`、`/model/ModelScope` | 模型只读缓存，放这里不占根盘 |

本地仓库结构：

```
D:\codex_project\vllm\
├─ README.md                    # 本文（环境与用法）
├─ remote.ps1                   # 远程执行助手
├─ LOCAL-ONLY-CREDENTIALS.md    # 凭据（gitignore，禁止外传）
├─ .gitignore
├─ .agent\                      # 临时探测脚本（gitignore）
└─ omni-study\                  # 学习辅助：手搓版 mini_omni + 测试（不计入正式学习路线）
```

## 8. GitHub 同步工作流（2026-11-03 起）

远端：`https://github.com/AmumuZzzt/vllm-omni_qwen3_Asr`（**Private**）

分工：**本地读代码/写代码并提交推送；服务器只拉取（pull）以运行实验。**

```powershell
# 本地：提交 + 推送。必须显式覆盖代理（全局 http/https.proxy 指向 127.0.0.1:7890，
# 该代理常不在线；git 会把 config 里的空串当未设置而回落到全局，所以不能用 git config 关闭）
cd D:\codex_project\vllm
git add -A
git commit -m "..."
git -c http.proxy= -c https.proxy= push

# 服务器：拉取到工作区（首次 clone）
git clone https://github.com/AmumuZzzt/vllm-omni_qwen3_Asr.git /root/vllm-work/vllm-lab
# 之后更新
cd /root/vllm-work/vllm-lab && git pull --ff-only
```

注意事项：

- 本机全局 git 代理指向 `127.0.0.1:7890`，该代理常不在线；推送一律带 `-c http.proxy= -c https.proxy=`（HTTP_PROXY 环境变量为空，无需处理）。仓库级 `git config --local http.proxy ""` **无效**（git 回落全局），不要用。
- git 凭据助手依赖命名管道，DSH 受限模式下会失败；**推送需要一次性的完全权限**（凭据从 Windows 凭据管理器读取，令牌不落盘、不进日志）。
- `LOCAL-ONLY-CREDENTIALS.md`、`*.cred`、`*.pem`、`id_rsa*` 已在 `.gitignore` 中，**任何情况下不得提交**。
- 服务器上的实验源码树（`/opt/qwen3-asr/source-study` 等）**不在**本仓库管理范围内，仍按学习计划单独处理。
