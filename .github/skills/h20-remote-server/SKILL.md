---
name: h20-remote-server
description: "H20远程服务器开发工作流。Use when: 连接远程服务器、在远端编译/运行/测试代码、同步代码提交、管理远端工作空间 /data/chenz、SSH连接H20服务器、远端执行命令、代码同步。h20-old: ssh -p 30300 root@8.130.174.55（主力训练机，与他人共用，可用卡数不固定）；h20-new: ssh -p 32344 root@8.130.174.55（只能用 GPU 4-7，前4张卡严禁使用）；NAS盘: /data，主工作空间: /data/chenz。"
argument-hint: "可选：指定要执行的具体任务（如：编译、测试、同步代码）"
---

# H20 远程服务器开发工作流

## 服务器信息

| 别名 | SSH 连接命令 | 可用 GPU | 说明 |
|------|-------------|---------|------|
| **h20-old** | `ssh -p 30300 root@8.130.174.55` | 不固定，与他人共用 | 主力训练机，训练前必须先查 `nvidia-smi` 确认空闲卡 |
| **h20-new** | `ssh -p 32344 root@8.130.174.55` | ⚠️ **仅限 GPU 4-7**（前4张卡禁用） | 新机器，可用卡为 GPU 4-7 中的部分或全部 |

> ⚠️ **h20-old**：整机与其他用户共享，可用 GPU 数量不固定。启动训练前必须先运行 `nvidia-smi` 查看哪些卡空闲，再确定 `CUDA_VISIBLE_DEVICES`。

> ⚠️ **h20-new 限制**：GPU 0-3 属于其他用户，**严禁使用**。运行任务时 `CUDA_VISIBLE_DEVICES` 只能从 GPU 4、5、6、7 中选择，不必用满四张，但不得超出这个范围。

| 项目 | 值 |
|------|-----|
| NAS 盘路径 | `/data` |
| **主工作空间** | `/data/chenz`（**唯一允许的工作空间**） |

## 核心原则

1. **本地编辑代码，远端运行代码** — 所有编译、运行、测试必须在远端服务器执行，原则上禁止在本地运行代码。
2. **工作空间唯一** — 只允许使用 `/data/chenz`，不得使用其他目录。
3. **代码提交在本地** — 原则上不允许在远端直接提交代码；由本地提交后，同步更新远端状态。
4. **远端保持干净** — 远端仓库始终保持无脏提交状态；临时文件必须加入 `.gitignore`。

---

## 操作流程

### 1. 连接远端服务器

```bash
ssh -p 30300 root@8.130.174.55
```

连接后确认工作目录：

```bash
cd /data/chenz/<项目名>
```

### 2. 本地修改代码 → 远端编译/运行/测试

**步骤：**

1. 在本地 VS Code 中修改代码。
2. 将改动同步到远端（见下方"代码同步"流程）。
3. 在远端执行编译/运行/测试命令：

```bash
# 示例：编译
ssh -p 30300 root@8.130.174.55 "cd /data/chenz/<项目名> && <编译命令>"

# 示例：运行训练/测试
ssh -p 30300 root@8.130.174.55 "cd /data/chenz/<项目名> && <运行命令>"
```

4. 查看远端输出结果，在本地分析日志和错误。

### 3. 代码同步流程

**本地提交 → 推送远端 → 远端拉取（推荐方式）：**

```bash
# 本地：提交代码
git add .
git commit -m "提交信息"
git push origin <branch>

# 远端：拉取最新代码，保持干净状态
ssh -p 30300 root@8.130.174.55 "cd /data/chenz/<项目名> && git pull origin <branch>"
```

**验证远端状态干净：**

```bash
ssh -p 30300 root@8.130.174.55 "cd /data/chenz/<项目名> && git status"
# 期望输出：nothing to commit, working tree clean
```

### 4. 处理远端临时文件

远端产生的临时文件（日志、输出、缓存等）**必须加入 `.gitignore`**，不得提交到版本库。

**检查远端未跟踪文件：**

```bash
ssh -p 30300 root@8.130.174.55 "cd /data/chenz/<项目名> && git status --short"
```

**将临时文件夹加入 `.gitignore`（在本地操作后同步）：**

```bash
# 本地：编辑 .gitignore，添加临时目录
echo "tmp/" >> .gitignore
echo "outputs/" >> .gitignore
echo "*.log" >> .gitignore

# 本地提交
git add .gitignore
git commit -m "chore: ignore temp files on remote"
git push origin <branch>

# 远端拉取
ssh -p 30300 root@8.130.174.55 "cd /data/chenz/<项目名> && git pull origin <branch>"
```

---

## 禁止事项

| 禁止操作 | 说明 |
|----------|------|
| 本地运行代码 | 所有执行必须在远端进行 |
| 使用 `/data/chenz` 以外的工作空间 | 只允许 `/data/chenz` |
| 在远端直接 `git commit` | 提交只在本地进行 |
| 将临时文件提交到版本库 | 必须加入 `.gitignore` |
| **未经用户明确允许停止训练进程** | 严禁在未获得用户明确指令的情况下 kill、中断或停止任何正在运行的训练进程（包括但不限于 `kill`、`pkill`、`Ctrl+C`、关闭 tmux 会话等操作） |
| **未经用户明确允许重新启动训练** | 严禁在未获得用户明确指令的情况下重新发起训练（包括直接启动 `torchrun`/`python train.py`，或通过脚本间接触发） |
| **禁止干扰其他用户的进程** | 服务器为多人共用，严禁 kill、中断或以任何方式干扰属于其他用户的进程、tmux 会话或训练任务；操作前须通过进程归属（`ps aux`、`tmux list-sessions`）确认目标进程属于当前用户 |

---

## 常用命令速查

```bash
# ── h20-old（主力训练机，GPU 0-7 全部可用）──────────────────
# 连接服务器
ssh -p 30300 root@8.130.174.55

# 在远端执行单条命令
ssh -p 30300 root@8.130.174.55 "cd /data/chenz/<项目名> && <命令>"

# 查看 GPU 状态
ssh -p 30300 root@8.130.174.55 "nvidia-smi"

# 查看 GPU 占用，确认空闲卡再决定 CUDA_VISIBLE_DEVICES
ssh -p 30300 root@8.130.174.55 "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader"

# 多卡训练（实际可用卡号以 nvidia-smi 结果为准）
ssh -p 30300 root@8.130.174.55 "cd /data/chenz/<项目名> && CUDA_VISIBLE_DEVICES=<空闲卡号> /data/chenz/conda_env/<env>/bin/torchrun --nproc_per_node <N> train.py ..."

# ── h20-new（⚠️ 仅 GPU 4-7 可用，前4张卡严禁使用）────────────
# 连接服务器
ssh -p 32344 root@8.130.174.55

# 在远端执行单条命令
ssh -p 32344 root@8.130.174.55 "cd /data/chenz/<项目名> && <命令>"

# 查看 GPU 状态（只看 GPU 4-7）
ssh -p 32344 root@8.130.174.55 "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | awk -F, 'NR>=5'"

# 训练（CUDA_VISIBLE_DEVICES 只能从 4,5,6,7 中选，不必用满4张）
ssh -p 32344 root@8.130.174.55 "cd /data/chenz/<项目名> && CUDA_VISIBLE_DEVICES=4,5,6,7 /data/chenz/conda_env/<env>/bin/torchrun --nproc_per_node 4 train.py ..."

# ── 通用 ──────────────────────────────────────────────────────
# 查看磁盘使用
ssh -p 30300 root@8.130.174.55 "df -h /data"

# 查看工作空间
ssh -p 30300 root@8.130.174.55 "ls /data/chenz"

# 同步代码到远端
ssh -p 30300 root@8.130.174.55 "cd /data/chenz/<项目名> && git pull origin <branch> && git status"
```
