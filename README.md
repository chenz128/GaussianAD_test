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
