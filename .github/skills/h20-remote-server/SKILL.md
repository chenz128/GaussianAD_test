---
name: h20-remote-server
description: "H20远程服务器开发工作流。Use when: 连接远程服务器、在远端编译/运行/测试代码、同步代码提交、管理远端工作空间 /data/chenz、SSH连接H20服务器、远端执行命令、代码同步。SSH地址: ssh -p 31256 root@8.130.174.55，NAS盘: /data，主工作空间: /data/chenz。"
argument-hint: "可选：指定要执行的具体任务（如：编译、测试、同步代码）"
---

# H20 远程服务器开发工作流

## 服务器信息

| 项目 | 值 |
|------|-----|
| SSH 连接命令 | `ssh -p 31256 root@8.130.174.55` |
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
ssh -p 31256 root@8.130.174.55
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
ssh -p 31256 root@8.130.174.55 "cd /data/chenz/<项目名> && <编译命令>"

# 示例：运行训练/测试
ssh -p 31256 root@8.130.174.55 "cd /data/chenz/<项目名> && <运行命令>"
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
ssh -p 31256 root@8.130.174.55 "cd /data/chenz/<项目名> && git pull origin <branch>"
```

**验证远端状态干净：**

```bash
ssh -p 31256 root@8.130.174.55 "cd /data/chenz/<项目名> && git status"
# 期望输出：nothing to commit, working tree clean
```

### 4. 处理远端临时文件

远端产生的临时文件（日志、输出、缓存等）**必须加入 `.gitignore`**，不得提交到版本库。

**检查远端未跟踪文件：**

```bash
ssh -p 31256 root@8.130.174.55 "cd /data/chenz/<项目名> && git status --short"
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
ssh -p 31256 root@8.130.174.55 "cd /data/chenz/<项目名> && git pull origin <branch>"
```

---

## 禁止事项

| 禁止操作 | 说明 |
|----------|------|
| 本地运行代码 | 所有执行必须在远端进行 |
| 使用 `/data/chenz` 以外的工作空间 | 只允许 `/data/chenz` |
| 在远端直接 `git commit` | 提交只在本地进行 |
| 将临时文件提交到版本库 | 必须加入 `.gitignore` |

---

## 常用命令速查

```bash
# 连接服务器
ssh -p 31256 root@8.130.174.55

# 在远端执行单条命令
ssh -p 31256 root@8.130.174.55 "cd /data/chenz/<项目名> && <命令>"

# 查看远端 GPU 状态
ssh -p 31256 root@8.130.174.55 "nvidia-smi"

# 查看远端磁盘使用
ssh -p 31256 root@8.130.174.55 "df -h /data"

# 查看远端工作空间
ssh -p 31256 root@8.130.174.55 "ls /data/chenz"

# 同步代码到远端
ssh -p 31256 root@8.130.174.55 "cd /data/chenz/<项目名> && git pull origin <branch> && git status"
```
