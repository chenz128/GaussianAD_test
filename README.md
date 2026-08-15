# GaussianAD

## Circle Planner 架构（VADHeadCircleGaussian, planner_v5）

把「ego↔agent → ego↔map → ego↔gaussian(当前) → ego↔gaussian(未来)」这条 4 路递进交叉注意力封装成一个 block，循环 `num_circles` 次（recurrent refinement）：每轮用上一轮融合后的 ego 重新再看一遍 4 路信息，逐轮迭代收敛，实现信息充分融合，重点改善碰撞率 (obj_box_col)。

```mermaid
flowchart TB
    classDef inp  fill:#e3f2fd,stroke:#1976d2,color:#000
    classDef cur  fill:#fff9c4,stroke:#f9a825,color:#000
    classDef fut  fill:#f3e5f5,stroke:#8e24aa,color:#000
    classDef head fill:#c8e6c9,stroke:#2e7d32,color:#000
    classDef loss fill:#ffcdd2,stroke:#c62828,color:#000
    classDef sw   fill:#ffe0b2,stroke:#e65100,color:#000
    classDef loop fill:#e8f5e9,stroke:#1b5e20,color:#000

    %% ===== 输入（来自上游模块）=====
    AG["agent_query<br/>当前帧检测"]:::inp
    MP["map_query<br/>当前帧地图"]:::inp
    GO["gaussian_output<br/>当前帧高斯 (B,G,28)"]:::inp
    OF["offset<br/>flow 位移 (B,G,fut_ts·2)"]:::inp

    %% ===== 未来帧高斯构造（只算一次）=====
    FB["未来帧高斯<br/>xy = 当前高斯.xy + offset_t<br/>其余 26 维不变"]:::fut

    %% ===== KV 源（只编码一次，循环内复用）=====
    subgraph KV["KV 源（只编码一次，循环内复用）"]
        AK["agent_kv (N_agent,B,D)"]:::cur
        MK["map_kv (N_map,B,D)"]:::cur
        GK["gs_kv (G,B,D)"]:::cur
        FK["fut_gs_kv (fut_ts·G,B,D)"]:::fut
    end

    %% ===== circle：4 路递进注意力 × N 次 =====
    subgraph CIRCLE["circle block：4 路递进注意力 × N 次（num_circles=N）"]
        EQ["ego_query<br/>初始 (1,B,D)"]:::cur
        subgraph RD1["第 1 轮（i=1）"]
            A1["ego↔agent"]:::cur
            M1["ego↔map"]:::cur
            G1["ego↔gaussian(当前)"]:::cur
            F1["ego↔gaussian(未来)"]:::fut
        end
        MID["⋮ 第 2 .. N−1 轮 ⋮<br/>（相同操作，ego 逐轮 refine）"]:::loop
        subgraph RDN["第 N 轮（最后一轮）"]
            AN["ego↔agent"]:::cur
            MN["ego↔map"]:::cur
            GN["ego↔gaussian(当前)"]:::cur
            FN["ego↔gaussian(未来)"]:::fut
        end
    end

    CAT["拼接 4 路特征 (4D=512)<br/>只用最后一轮中间态"]:::head
    DEC["ego_fut_decoder MLP<br/>(原输出头，输入 3D→4D)"]:::head
    PRED["ego_fut_preds<br/>(B, ego_fut_mode, fut_ts, 2)"]:::head

    PL["PlanLoss (w=10)"]:::loss

    %% ===== occ_flow 耦合路径 =====
    SEL["_select_plan_ego<br/>按 ego_fut_cmd 选 mode"]:::sw
    DET{"plan_ego_detach?<br/>当前配置 = False"}:::sw
    COMP["occ_flow ego 补偿<br/>means_fut − planner_res"]
    OFL["OccFlowLoss"]:::loss

    %% ---------- 前向 + 梯度 ----------
    AG ==> AK
    MP ==> MK
    GO ==> GK
    GO ==>|✅grad| FB
    OF ==>|✅grad| FB
    FB ==> FK

    EQ ==>|query| A1
    AK ==>|key/val| A1
    A1 ==>|query| M1
    MK ==>|key/val| M1
    M1 ==>|query| G1
    GK ==>|key/val| G1
    G1 ==>|query| F1
    FK ==>|key/val| F1
    F1 ==>|"ego₁"| MID
    MID ==>|"ego_{N−1}"| AN
    AK ==>|key/val 复用| AN
    AN ==>|query| MN
    MK ==>|key/val 复用| MN
    MN ==>|query| GN
    GK ==>|key/val 复用| GN
    GN ==>|query| FN
    FK ==>|key/val 复用| FN
    FN -.->|"ego_N 迭代 → 回到第 1 轮（循环 × N）"| A1

    AN ==> CAT
    MN ==> CAT
    GN ==> CAT
    FN ==> CAT
    CAT ==> DEC ==> PRED
    PRED ==>|✅grad 回传 planner| PL

    %% ---------- occ_flow → planner 的可切换耦合 ----------
    PRED ==> SEL ==> DET
    DET ==>|"detach=False：✅梯度回传 planner（当前）"| COMP
    DET -.->|"detach=True：🚫切断"| COMP
    OF  ==>|✅grad 训练 offset| COMP
    GO  -.->|"🚫detach（flow_grad_scale=0）"| COMP
    COMP ==> OFL

    linkStyle 34 stroke:#c62828,stroke-width:2px,stroke-dasharray:5 5
    linkStyle 36 stroke:#c62828,stroke-width:2px,stroke-dasharray:5 5
```

---

## Planner A：FutAttn（VADHeadFutAttn, planner_v2）

配置文件：`config/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futattn/`（base: `v12_ft_plan`）

保留**当前帧 3 路**（与 `VADHead` 完全一致，结构/权重不变），新增**未来帧分支**：

- 未来帧高斯 = 当前高斯 xy + offset 逐帧平移（与 `gaussian_head.forward_flow` 的 `means_fut = means + offset` 一致，batch=1）；
- 未来 ego token：当前帧融合特征 (3D) 经 `ego_to_fut` 投影 + **逐帧位置编码 `fut_pos` (query 侧)**；
- 先 `fut_self_decoder` 做时间维 self-attention（建模 6 帧时序连贯性）；
- 再 `ego_fut_gaussian_decoder` 逐帧交叉注意力（frames 折叠进 batch 维：query `(1,fut_ts,D)` ↔ key/val `(G,fut_ts,D)`，第 t 帧 ego 只看第 t 帧高斯）；
- 逐时间步回归 `fut_out_mlp` → `(B, ego_fut_mode, fut_ts, 2)`。

```mermaid
flowchart TB
    classDef inp  fill:#e3f2fd,stroke:#1976d2,color:#000
    classDef cur  fill:#fff9c4,stroke:#f9a825,color:#000
    classDef fut  fill:#f3e5f5,stroke:#8e24aa,color:#000
    classDef head fill:#c8e6c9,stroke:#2e7d32,color:#000
    classDef loss fill:#ffcdd2,stroke:#c62828,color:#000
    classDef sw   fill:#ffe0b2,stroke:#e65100,color:#000
    classDef te   fill:#e0f7fa,stroke:#006064,color:#000

    %% ===== 输入（来自上游模块）=====
    AG["agent_query<br/>当前帧检测"]:::inp
    MP["map_query<br/>当前帧地图"]:::inp
    GO["gaussian_output<br/>当前帧高斯 (B=1,G,28)"]:::inp
    OF["offset<br/>flow 位移 (B,G,fut_ts·2)"]:::inp

    %% ===== 当前帧 3 路（复用 VADHead，权重不变）=====
    subgraph CUR["当前帧 3 路（复用 VADHead）"]
        EQ["ego_query"]:::cur
        EA["ego↔agent"]:::cur
        EM["ego↔map"]:::cur
        EG["ego↔gaussian(当前)"]:::cur
    end

    %% ===== 未来帧分支（新增）=====
    subgraph FUT["未来帧分支（新增）"]
        FB["未来帧高斯<br/>xy = 当前.xy + offset_t<br/>(fut_ts, G, 28)"]:::fut
        FMLP["fut_gaussian_fus_mlp<br/>(fut_ts, G, D)"]:::fut
        KV["key/value<br/>(G, fut_ts, D)"]:::fut
        E2F["ego_to_fut<br/>3D=384 → D=128"]:::te
        FPE["+ fut_pos 位置编码<br/>nn.Embedding(fut_ts, D)<br/>🌐 query 侧时间编码"]:::te
        SD["fut_self_decoder<br/>时间维 self-attn"]:::fut
        EFG["ego_fut_gaussian_decoder<br/>cross-attn<br/>q:(1,fut_ts,D) ↔ kv:(G,fut_ts,D)"]:::fut
    end

    CAT["拼 3 路 ego_feats (3D)"]:::head
    OUT["fut_out_mlp 逐帧回归"]:::head
    PRED["ego_fut_preds<br/>(B, ego_fut_mode, fut_ts, 2)"]:::head

    PL["PlanLoss (w=10)"]:::loss

    %% ===== occ_flow 耦合路径 =====
    SEL["_select_plan_ego<br/>按 ego_fut_cmd 选 mode"]:::sw
    DET{"plan_ego_detach?<br/>当前配置 = False"}:::sw
    COMP["occ_flow ego 补偿<br/>means_fut − planner_res"]
    OFL["OccFlowLoss"]:::loss

    %% ---------- 前向 + 梯度 ----------
    EQ ==> EA ==>|query| EM ==>|query| EG
    AG ==>|key/val| EA
    MP ==>|key/val| EM
    GO ==>|key/val| EG

    GO ==>|✅grad| FB
    OF ==>|✅grad| FB
    FB ==> FMLP ==> KV
    EG ==> E2F
    E2F ==> FPE
    FPE ==>|query| SD
    SD ==>|query| EFG
    KV ==>|key/val| EFG
    EA ==> CAT
    EM ==> CAT
    EG ==> CAT
    CAT ==>|init fut token| E2F
    EFG ==> OUT ==> PRED
    PRED ==>|✅grad 回传 planner| PL

    %% ---------- occ_flow → planner 的可切换耦合 ----------
    PRED ==> SEL ==> DET
    DET ==>|"detach=False：✅梯度回传 planner（当前）"| COMP
    DET -.->|"detach=True：🚫切断"| COMP
    OF  ==>|✅grad 训练 offset| COMP
    GO  -.->|"🚫detach（flow_grad_scale=0）"| COMP
    COMP ==> OFL

    linkStyle 25 stroke:#c62828,stroke-width:2px,stroke-dasharray:5 5
    linkStyle 27 stroke:#c62828,stroke-width:2px,stroke-dasharray:5 5
```

---

## Planner B：FutGauTime（VADHeadFutGaussianTime, planner_v4）

配置文件：`config/nuscenes_gs25600_gtbox_oracle_v12_futgau_costime/`（base: `base_plan`，对照 `futgau_detach_false`）

与 FutAttn 的设计路线不同：**不新建未来 ego token**，而是把**未来帧高斯**作为与 agent/map/当前高斯完全对称的**第 4 路 stream** 融合：

- 未来帧高斯 = 当前高斯 xy + offset 逐帧平移，6 帧展平为一个 key 集合 `(fut_ts·G, D)`（batch=1 折叠）；
- ego 单 query 与展平的未来高斯做一次交叉注意力（`ego_fut_gaussian_decoder`）；
- **逐时间步位置编码 `fut_time_pos` 加在 key/value 侧**（即高斯特征上），使每个 key 明确携带「属于未来第几帧」；
- 拼接 4 路特征 (4D=512) → 复用原 `ego_fut_decoder`（输入 3D→4D 加宽）回归轨迹。

```mermaid
flowchart TB
    classDef inp  fill:#e3f2fd,stroke:#1976d2,color:#000
    classDef cur  fill:#fff9c4,stroke:#f9a825,color:#000
    classDef fut  fill:#f3e5f5,stroke:#8e24aa,color:#000
    classDef head fill:#c8e6c9,stroke:#2e7d32,color:#000
    classDef loss fill:#ffcdd2,stroke:#c62828,color:#000
    classDef sw   fill:#ffe0b2,stroke:#e65100,color:#000
    classDef te   fill:#e0f7fa,stroke:#006064,color:#000

    %% ===== 输入（来自上游模块）=====
    AG["agent_query<br/>当前帧检测"]:::inp
    MP["map_query<br/>当前帧地图"]:::inp
    GO["gaussian_output<br/>当前帧高斯 (B=1,G,28)"]:::inp
    OF["offset<br/>flow 位移 (B,G,fut_ts·2)"]:::inp

    %% ===== 当前帧 3 路（复用 VADHead，权重不变）=====
    subgraph CUR["当前帧 3 路（复用 VADHead）"]
        EQ["ego_query"]:::cur
        EA["ego↔agent"]:::cur
        EM["ego↔map"]:::cur
        EG["ego↔gaussian(当前)"]:::cur
    end

    %% ===== 新增第 4 路：未来帧高斯 =====
    subgraph FUT["新增第 4 路 stream：未来帧高斯"]
        FB["未来帧高斯<br/>xy = 当前.xy + offset_t<br/>(fut_ts, G, 28)"]:::fut
        TPE["+ fut_time_pos 位置编码<br/>nn.Embedding(fut_ts, D)<br/>🌐 key/value 侧时间编码"]:::te
        FK["展平 key 集合<br/>(fut_ts·G, B=1, D)"]:::fut
        EFG["ego_fut_gaussian_decoder<br/>cross-attn（ego 单 query）"]:::fut
    end

    CAT["拼接 4 路特征 (4D=512)"]:::head
    DEC["ego_fut_decoder MLP<br/>(原输出头，输入 3D→4D)"]:::head
    PRED["ego_fut_preds<br/>(B, ego_fut_mode, fut_ts, 2)"]:::head

    PL["PlanLoss (w=10)"]:::loss

    %% ===== occ_flow 耦合路径 =====
    SEL["_select_plan_ego<br/>按 ego_fut_cmd 选 mode"]:::sw
    DET{"plan_ego_detach?<br/>当前配置 = False"}:::sw
    COMP["occ_flow ego 补偿<br/>means_fut − planner_res"]
    OFL["OccFlowLoss"]:::loss

    %% ---------- 前向 + 梯度 ----------
    EQ ==> EA ==>|query| EM ==>|query| EG
    AG ==>|key/val| EA
    MP ==>|key/val| EM
    GO ==>|key/val| EG

    GO ==>|✅grad| FB
    OF ==>|✅grad| FB
    FB ==> TPE ==> FK
    EG ==>|query| EFG
    FK ==>|key/val| EFG
    EA ==> CAT
    EM ==> CAT
    EG ==> CAT
    EFG ==> CAT
    CAT ==> DEC ==> PRED
    PRED ==>|✅grad 回传 planner| PL

    %% ---------- occ_flow → planner 的可切换耦合 ----------
    PRED ==> SEL ==> DET
    DET ==>|"detach=False：✅梯度回传 planner（当前）"| COMP
    DET -.->|"detach=True：🚫切断"| COMP
    OF  ==>|✅grad 训练 offset| COMP
    GO  -.->|"🚫detach（flow_grad_scale=0）"| COMP
    COMP ==> OFL

    linkStyle 22 stroke:#c62828,stroke-width:2px,stroke-dasharray:5 5
    linkStyle 24 stroke:#c62828,stroke-width:2px,stroke-dasharray:5 5
```

---

---

## 两个 Planner 的时间编码区别


|                  | **Planner A：FutAttn** (planner_v2)                                                      | **Planner B：FutGauTime** (planner_v4)                                     |
| ---------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 配置文件         | `v12_ft_plan_futattn`                                                                    | `v12_futgau_costime`                                                       |
| 时间编码位置     | **query 侧**（未来 ego token）                                                           | **key/value 侧**（未来帧高斯）                                             |
| 实现             | `fut_pos = nn.Embedding(fut_ts, D)`，加到未来 ego token 逐帧                             | `fut_time_pos = nn.Embedding(fut_ts, D)`，加到未来高斯 `(fut_ts,G,D)` 逐帧 |
| 未来帧组织形式   | 帧折叠进**batch 维**：q `(1,fut_ts,D)` ↔ kv `(G,fut_ts,D)`，第 t 帧 ego 只看第 t 帧高斯 | 帧+高斯**展平**成一个 key 集合 `(fut_ts·G, B, D)`，ego 单 query 看全部    |
| 时序建模         | `fut_self_decoder`（时间维 self-attn）+ `fut_pos` PE                                     | 无 self-attn，仅靠 key 侧 PE + offset 坐标                                 |
| 输出头           | 新分支`fut_out_mlp` **逐时间步回归**（替换原头）                                         | 复用原`ego_fut_decoder`（3D→4D 加宽）                                     |
| 已知取舍         | 3D→D 压缩瓶颈；新头从零学                                                               | 保留预训练输出头作 baseline；注意力无帧隔离                                |
| 无时间编码的对照 | —                                                                                       | `futgau_detach_false`（v3 展平后无任何 PE）                                |

> 简言之：**FutAttn 把时间编码放在「问的一方」（ego query），配合帧间 self-attn 让轨迹回归逐帧感知时刻**；**FutGauTime 把时间编码放在「被看的一方」（未来高斯 key），让展平的未来场景本身携带帧序信息**。两者都从 `v12_fixempty epoch_15` 续训、`plan_ego_detach=False`（occ_flow 梯度可回传 planner）。
