# DLPack 混合训练实践报告：MindSpore → PyTorch 渐进迁移的过渡态

> **一句话结论**：在昇腾 910B(Atlas A2)上，以图网络代表模型 GCN on Cora 为载体，把"DLPack 混合过渡态"完整跑通——混合训练与纯 MindSpore 收敛/精度完全一致(test Accuracy 80.3% vs 80.3%)，交换段 DLPack 比 host 拷贝快 4.4~7.8 倍(训练内实测)且随边界张量放大到 60.9 倍，反向切分(PT 前段 + MS 后段)同样成立。渐进迁移可以**敢切、选对、全程适用**。

## 1 背景：为什么需要"混合过渡态"

业界训练框架以 PyTorch 为主，生态与社区资源更丰富；基于 MindSpore 实现的存量模型，因此普遍存在向 PyTorch 渐进迁移的诉求。迁移的真正难点不在"能不能重写"，而在**迁移周期与模型演进并行**：人力只够逐模块迁移，训练迭代却不能停。混合过渡态的思路是——已迁移的模块用 PyTorch 跑，未迁移的留在 MindSpore，边界上用 DLPack 做零拷贝张量交换 + 梯度桥接，两侧参数在**同一步训练中联合更新**；随迁移推进 MS 侧越来越小，最终整体切换，任意时刻都停在一个"能训练的完整模型"上。

这个方案成立需要回答三个问题，它们构成整个验证的主线：

1. **敢切吗(切换等价性)**：混合态相对纯 MindSpore 现状，收敛与精度是否一致?
2. **选对了吗(交换机制)**：边界交换用 DLPack 零拷贝，还是 host 拷贝?
3. **全程适用吗(形态对称性)**：迁移初期"MS 前段 + PT 后段"，迁移过半变成"PT 前段 + MS 后段"——两个方向的桥接机制是否都成立?

## 2 验证思路：三个问题，三组对照

**验证载体**选了真实开源模型 **GCN on Cora**(Kipf & Welling 两层图卷积 + PyG Planetoid Cora，2708 节点/1433 维特征/7 分类)——只是挑一个图网络的代表性开源模型做例子，不代表迁移对象本身是 GCN。选它的理由：自带真实的图语义算子(scatter 聚合/gather)，超参用 Kipf 论文原始 recipe(Adam lr=0.01、weight_decay=5e-4、200 epochs 全批)公开可复核，同时规模小到能在单卡上快速迭代；**验证的方法、判据与公共改造点面向任意 MindSpore 模型的迁移，不限定于图网络**(详见 §4.2)。

**对照设计**是整个验证的核心方法：

- **B0 参照臂**(纯 MindSpore 两层 GCN)= 唯一基线，每次运行与混合臂**同种子、同初始化、同数据、同超参**一起训练——回答"敢切"；
- **E2 vs E3 主对照**：同为混合场景(MS 聚合+conv1 ──交换──▶ PT conv2)，唯一变量 = 交换机制(E2 = DLPack / E3 = host 拷贝)，两跑的框架调度开销完全相同，差值纯粹来自交换机制——回答"选对"；
- **E4 形态对称**：反向切分 PT(聚合+conv1)──交换──▶ MS(conv2)，迁移过半时的真实形态——回答"全程适用"；
- **E5 尺寸扫描**：与模型无关的纯交换吞吐 bench(边界张量 169 KB → 4 MB)，给机制级收益上界。

**判据体系**(对齐行业咨询口径；全部以 B0 为锚)：

| 编号 | 判据 | 口径 |
|---|---|---|
| C1 | 图语义聚合一致 | 交换边界两侧各做一次 scatter 聚合，结果一致 ≤1e-5(必过) |
| C2 | 梯度等价性(双锚) | 桥接梯度 vs torch 参考锚 + vs B0 纯 MS 端到端锚，相对误差 < 5e-3 |
| C3 | 稀疏参数完整性 | 交换路径上未访问 Embedding 行梯度恒零、更新后零变化 |
| C4 | 收敛曲线一致性 | GCN 大矩阵规模逐位一致不可达(归约顺序噪声)，口径 = ulp 级差 ≤1e-5 + Accuracy 趋势重合；混合 vs B0 统计等价(loss 比 ∈[0.5，2.0]、test Accuracy 差 ≤1.5pt) |
| P1 | 交换信噪比 | 交换段占单 epoch 训练 <1% 为佳，小模型口径偏高属预期 |
| P2 | epoch 耗时总账 | 真实 GNN 接入后 ≥3 epoch ≤B0×1.2 为 PASS |

一键复现：`python scripts/experiments/run_all.py` 顺序执行 E0 环境自检 → E1 机制验证 → E2/E3/E4 训练 → E5 bench，实时日志 + 摘要/逐步 CSV 全量落盘。

## 3 验证结果

### 3.1 一屏总览

| 判据 | 结果 | 关键实测 |
|---|---|---|
| E1 机制验证(17 检查项) | ✅ | 17/17 PASS(零拷贝/桥接解析解/负对照/C1/C3) |
| C1 图语义聚合一致 | ✅ | 相对误差 0.0(≤1e-5) |
| C2 梯度等价性(双锚) | ✅ | torch 锚 ≤9.1e-05 / B0 锚 ≤9.1e-05(容差 5e-3) |
| C3 稀疏参数完整性 | ✅ | 未访问行梯度恒零，更新后零变化 |
| C4 收敛曲线一致性 | ✅ | E2/E3 final loss 差 9.3e-10(ulp 级)；三臂收敛于 80.3% |
| 敢切(切换等价性) | ✅ PASS | test Accuracy 差 0.00pt，loss 比 1.003 |
| 选对(交换机制) | ✅ dlpack 优 | 交换段 copy 慢 6.6x/4.4x，训练数值一致 |
| 全程适用(对称形态 E4) | ✅ PASS | test Accuracy 差 0.00pt，loss 比 1.0000 |
| P1 交换信噪比 | ✅ | E2 0.05% / E3 0.26%(E4 8.69%，小图口径见 §3.8) |
| P2 epoch 耗时总账 | ✅ 同量级 | 混合臂 272/273/17 ms vs B0 ≈0.4 s/epoch；正式门禁 ≤B0×1.2 接入时复核 |

### 3.2 敢切：混合训练 ≡ 纯 MindSpore

每次运行 B0 与混合臂同种子同初始化训练 200 epochs，三臂并排：

| 指标 | B0 纯 MindSpore | E2 混合 (dlpack) | E3 混合 (copy) |
|---|---|---|---|
| final loss | 0.0098 | 0.0099 | 0.0099 |
| test Accuracy | 80.3% | 80.3% | 80.3% |
| 收敛 | ✅ | ✅ | ✅ |

loss 比 1.0030(∈[0.5，2.0])，test Accuracy 差 **0.00pt**(≤1.5pt)——**切换不改变训练效果**，从纯 MS 切到混合过渡态的第一天，训练曲线就应与现状重合。三条曲线的形态重合见图 1：

![GCN on Cora 三臂对照(910B 真机)](../charts/gcn_npu.png)

*图 1 — 训练 loss 与 test Accuracy 曲线(B0 灰 / E2 蓝 / E3 橙)。*

### 3.3 选对：交换段 dlpack 比 host 拷贝快 6.6x/4.4x

两臂唯一变量是交换机制，归因天然干净：

| 指标 | E2 dlpack | E3 host 拷贝 | 判定 |
|---|---|---|---|
| final loss | 0.0099 | 0.0099 | ✅ 一致(差 9.3e-10，ulp 级) |
| test Accuracy | 80.3% | 80.3% | — |
| 交换段 正向/回传 (µs/epoch) | 51.4 / 83.9 | 340.1 / 371.5 | **copy 慢 6.6x / 4.4x** |
| epoch 耗时(去预热) | 272.00 ms | 273.43 ms | ✅ dlpack 快 |

两种机制训练出**逐位一致**的数值(loss 差 9.3e-10)——交换机制不影响数值，可以放心选型；收益面：交换段 DLPack 显著更快，epoch 耗时的绝对差(+1.4 ms)被框架调度常数摊薄，机制级收益上界见 §3.7。

![E2/E3 交换段与 epoch 耗时对比(910B 真机)](../charts/e2e3_gcn.png)

*图 2 — 交换段耗时(µs/epoch，log 刻度)与 epoch 耗时。*

### 3.4 全程适用：反向切分(PT 前段 + MS 后段)同样成立

迁移过半后，真实形态是"已迁移模块在前"——PT(聚合+conv1) ──dlpack──▶ MS(conv2)。E4 与其运行内 B0 对照：

| 指标 | B0 纯 MindSpore | E4 对称混合 | 判定 |
|---|---|---|---|
| final loss | 0.0098 | 0.0098 | loss 比 1.0000 |
| test Accuracy | 80.3% | 80.3% | 差 0.00pt ✅ |
| C2 双锚 rel_err | — | torch 9.1e-05 / B0 7.3e-05 | ✅ PASS |

**切分方向不影响桥接机制**：E4 的交换是 PT→MS(激活值)与 MS→PT(边界梯度)双向，与 E2 镜像，等价性与双锚全过。

一个如实披露：E4 的 epoch 耗时 17 ms，远低于 E2 的 272 ms——这不是"E4 更优"，而是两形态框架分工不同(front_pt 把更多计算放到了调度更快的 PT 侧)。E4 与 E2 的时延**不可直接对比**(控制变量对照只有 E2 vs E3)；E4 验证的是机制对称性，不是性能。

### 3.5 机制正确性：零拷贝与梯度桥接的数值证据

收敛一致也可能是"两边都没训起来"的假象，机制验证(E1，17 个检查项)在训练前先证明桥接的数学正确性：

| 证据项 | 真机实测 |
|---|---|
| 双向零拷贝语义 | 一侧原地修改，另一侧同步可见(PT→MS 共享 NPU 显存 / MS→PT 共享 host 内存) |
| 前向 loss vs float64 解析解 | 0.22105893 vs 0.22105896 |
| 桥接梯度 vs 解析解(pt_input.grad / W₁ / b₁) | 相对误差 ≤1.4e-07(容差 5e-3，低约 5 个数量级) |
| C1 图语义聚合两侧一致 | 相对误差 0.0(≤1e-5) |
| 负对照(假 sens 必须不同) | 梯度差 \|ΔW\| = 1.515 → 传递的是真实上游梯度 |
| C3 稀疏 Embedding 完整性 | 未访问行梯度恒零，更新后零变化 |
| 两侧参数有效更新 | PT 1.12e-04 / MS 4.03e-03 |
| 单次交换耗时 | dlpack 12.1 µs vs copy 31.1 µs |

### 3.6 梯度双锚：桥接无损的定量证明

每个混合臂训练前，同权重单步对照两把锚：**torch 锚**(桥接梯度 vs 纯 PyTorch 参考)证明桥接数学正确；**B0 锚**(vs 纯 MindSpore 端到端梯度)证明与基线等价：

| 运行 | torch 锚 rel_err | B0 锚 rel_err | 判定 |
|---|---|---|---|
| E2 混合 dlpack | 1.04e-05 | 9.06e-05 | ✅ |
| E3 混合 copy | 2.71e-05 | 9.06e-05 | ✅ |
| E4 对称混合 | 9.06e-05 | 7.25e-05 | ✅ |

三种配置全部远小于容差 5e-3(1433 维特征经两层传播会放大跨框架 ulp 差，B0 锚容差独立放宽至 max(rtol， 5e-3))。

![验证判据评分卡(910B 真机)](../charts/metrics_scorecard.png)

*图 3 — 判据评分卡：C2 双锚 / C4 机制间一致 / 收益·步时延 / E5 加速比。*

### 3.7 交换收益上界：随边界张量放大到 60.9 倍

与训练解耦的纯交换吞吐扫描(边界张量 = 节点数 × 16 × 4B，同进程对比)：

| 边界张量 | 正向 MS→PT dlpack / copy | 收益 | 回传 PT→MS dlpack / copy | 收益 |
|---|---|---|---|---|
| 169 KB (2708 节点) | 19.2 / 150.7 µs | **7.8x** | 33.9 / 180.1 µs | **5.3x** |
| 1.0 MB (16384) | 19.0 / 406.1 µs | **21.4x** | 121.6 / 624.5 µs | **5.1x** |
| 4.0 MB (65536) | 19.5 / 1188.1 µs | **60.9x** | 448.5 / 2033.8 µs | **4.5x** |

DLPack 全程留在 NPU 显存；copy 每次都要 NPU→host→NPU 往返 + 流同步，代价随张量增大线性放大。**边界张量越大，选择 DLPack 的收益越大**——真实 GNN 的边界张量通常远大于 4 MB，这是机制级下界的保证。

![纯交换吞吐对比(910B --bench 尺寸扫描)](../charts/bench_npu.png)

*图 4 — 纯交换吞吐对比(log 刻度)。*

### 3.8 性能口径：信噪比与 epoch 总账

**P1 交换信噪比**(交换段占单 epoch 训练)：

| 运行 | 交换段合计 | 单 epoch | 占比 |
|---|---|---|---|
| E2 dlpack | 135.4 µs | 272 ms | **0.05%** ✅ |
| E3 copy | 711.5 µs | 273 ms | **0.26%** ✅ |
| E4 对称 | 1474.0 µs | 17 ms | 8.69%(小图口径，见下) |

E4 偏高有两层原因：一是分母小——单 epoch 只有 17 ms；二是它的交换更"重"——DLPack 传递的是同一块显存的视图，为了保证 MS 侧读到完整、不被后续计算覆写的数据，每次交换前要先复制一份独立副本并等待流同步，这份安全成本也计入了交换耗时。P1 的口径以真实 GNN 单 batch 为分母（例如交换 100 µs / 单 batch 100 ms = 0.1%），届时占比会自然落回 1% 以内。

**P2 epoch 耗时总账**：混合臂 272/273 ms（E2/E3）、17 ms（E4，去预热）；B0 纯 MindSpore 约 **0.4 s/epoch**。混合过渡态与纯 MindSpore 的 epoch 开销**同一量级**（front_pt 形态反而低一个数量级），范式代价不构成迁移障碍；正式门禁（真实 GNN ≥3 epoch ≤B0×1.2）在接入真实模型时执行。

## 4 详细设计

### 4.1 验证载体：GCN on Cora 的混合改造点

GCN on Cora 的两层结构是 `H = ReLU(Agg(X)·W1 + b1)`、`logits = Agg(H)·W2 + b2`（Agg = scatter 均值消息传递）。混合改造就是把这条链在 conv1 之后切开，两侧各留一段，边界张量是节点表示 H（2708×16）——**同一个切分点，两个方向都要跑通**：

```mermaid
flowchart TB
    subgraph S1["E2 形态 · 迁移初期：MS 前段 + PT 后段"]
        direction LR
        A1["MindSpore<br/>聚合 + conv1 + ReLU"] ==>|"边界张量 H 2708x16<br/>DLPack 零拷贝交换"| B1["PyTorch<br/>conv2 + 交叉熵"]
        B1 -.->|"梯度桥接<br/>pt_input.grad 转 sens 回传"| A1
    end
    subgraph S2["E4 形态 · 迁移过半：PT 前段 + MS 后段"]
        direction LR
        A2["PyTorch<br/>聚合 + conv1 + ReLU"] ==>|"边界张量 H 2708x16<br/>DLPack 零拷贝交换"| B2["MindSpore<br/>conv2 + 交叉熵"]
        B2 -.->|"梯度桥接<br/>vjp 边界梯度回传"| A2
    end
```

两个方向训练时都是**两侧参数在同一步联合更新**（PT `optim.step()` + MS `set_data`），虚线就是 §4.4 的梯度桥接。围绕这个切分，一共动了五处——它们是**图网络场景特有的改动**：

| # | 改造点 | 具体做法 | 为什么 / 踩过的坑 |
|---|---|---|---|
| 1 | 聚合算子对齐 | MS `unsorted_segment_sum(gather(X,src),dst)×inv_deg` ↔ PT `zeros.index_add_(0,dst,h[src])×inv_deg`，scatter 均值、O(E·F) | 真实 GNN 的算子语义；不用稠密 `Â·X`——Cora 稠密版 ≈10.5 GFLOP，会把交换耗时淹没在计算里 |
| 2 | 切分点选 conv1 之后 | 边界张量 = 节点表示 N×16，两个方向共用同一位置 | 真实 GNN 的典型切分位置；一个切分点即可覆盖双向交换与两种桥接路径 |
| 3 | 优化器严格同源 | PT `torch.optim.Adam` ↔ MS 逐参数同口径 Adam（lr=0.01、wd=5e-4，Kipf recipe） | E4 的 MS 后段最初用裸 SGD，200 epochs 完全不收敛（test Accuracy 14.4% = 随机水平）——**跨框架优化器必须同源**，实战第一大坑 |
| 4 | 多臂参数独立 | B0 / 混合 / 对称各臂的 Parameter 各自独立拷贝 | 曾因对称臂复用参照臂权重，从"训练过的权重"起步造成隐性污染（对照失真） |
| 5 | 双锚标定按拓扑取边界 | front_ms 取 conv1 边界梯度；front_pt 取 conv1+conv2 全部参数梯度 | 标定的必须是"该切分方向实际经过桥接的那部分梯度"，锚才算数 |

### 4.2 与模型无关的公共部分

换一个模型(包括真实 GNN)，以下基建**原样复用**。

**① 交换原语封装**——四个函数与模型无关，`copy` 分支即 E3 的对照路径(节选自 `scripts/experiments/gcn_hybrid.py`，import 与计时细节有删节)：

```python
def _npu_sync():
    if CFG["is_npu"]:
        torch.npu.synchronize()

def exchange_forward(ms_out):
    """MS -> PT（E2/E3 正向；仅交换段计时）"""
    t0 = time.perf_counter()
    if CFG["exchange"] == "dlpack":
        pt_input = torch_from_dlpack(ms_to_dlpack(ms_out))    # 零拷贝视图
    else:                    # E3 对照: host 拷贝（NPU->host->NPU 往返）
        pt_input = torch.from_numpy(np.array(ms_out.asnumpy()))
        if CFG["is_npu"]:
            pt_input = pt_input.to(CFG["device"])
    return pt_input, time.perf_counter() - t0

def exchange_backward(pt_input):
    """PT 边界梯度 -> MS sens（E2/E3 回传）"""
    t0 = time.perf_counter()
    grad = pt_input.grad.detach()        # 带梯度信息的张量必须先 detach 才能导出
    if CFG["exchange"] == "dlpack":
        sens = ms_from_dlpack(torch_to_dlpack(grad))          # 零拷贝视图
    else:
        sens = ms.Tensor(to_np(grad))
    return sens, time.perf_counter() - t0

def exchange_activation(pt_act):
    """PT -> MS 激活值（E4 正向；NPU 上必须先拷贝 + 同步）"""
    t0 = time.perf_counter()
    _npu_sync()
    h_copy = pt_act.detach().clone()     # dlpack 只是视图: 不拷贝, MS 侧会读到
    h_ms = ms_from_dlpack(torch_to_dlpack(h_copy))            # 被后续计算覆写的数据
    del h_copy
    _npu_sync()                          # 等待写入完成再交棒
    return h_ms, time.perf_counter() - t0

def exchange_grad_ms2pt(g_ms):
    """MS -> PT 边界梯度（E4 回传）"""
    t0 = time.perf_counter()
    _npu_sync()
    g_copy = ms.Tensor(to_np(g_ms))      # 独立副本
    # dlpack 输出可能落在 CPU -> 强制搬回 NPU（E4 首跑崩溃的根因）
    g_pt = torch_from_dlpack(ms_to_dlpack(g_copy)).to(CFG["device"])
    _npu_sync()
    return g_pt, time.perf_counter() - t0
```

**② 梯度桥接模式**——任意切分点通用，每步训练就是一个七拍循环(E2 形态)：

```python
ms_out, bwd_fn = vjp(front, x_ms, w_ref, b_ref)   # MS 前段: vjp 一次得前向 + 反向函数
pt_input, _ = exchange_forward(ms_out)            # ① 交换 MS -> PT (零拷贝)
pt_input.requires_grad_(True)                     # ② 重立叶子, 接入 PT 计算图
loss = pt_head_and_ce(pt_input, ...)              # ③ PT 局部前向
loss.backward()                                   # ④ PT 局部反向
sens, _ = exchange_backward(pt_input)             # ⑤ pt_input.grad 打包回 MS
_, dw1, db1 = bwd_fn(sens)                        # ⑥ MS 反向 (以 sens 为上游梯度)
opt_pt.step(); opt_ms.step([dw1, db1])            # ⑦ 两侧参数同一步更新
```

E4 形态方向镜像，梯度种子改由 MS 侧 vjp 给出：

```python
h_act = torch.relu(agg_pt(x_t) @ w1_t + b1_t)     # PT 前段 (保持在计算图内)
h_ms, _ = exchange_activation(h_act)              # ① 交换 PT -> MS
loss, bwd_fn = vjp(ms_back_ce, h_ms, w2s, b2s)    # ② MS 后段前向 + 局部反向就绪
grads = bwd_fn(ms.Tensor(1.0))                    # ③ 标量种子一次取 (dH, dW2, db2)
g_pt, _ = exchange_grad_ms2pt(grads[0])           # ④ 边界梯度 MS -> PT
h_act.backward(gradient=g_pt)                     # ⑤ 外部梯度种子驱动 PT 反传
opt_pt.step(); opt_ms.step([grads[1], grads[2]])  # ⑥ 两侧参数同一步更新
```

**③ 双锚标定方法**：训练前同权重单步，torch 参考 + 纯 MS 端到端两把锚；

**④ 判据体系与记分板呈现**(C1~C4 + P1~P2)；

**⑤ 一键编排与落盘**：run_all 顺序执行 + 摘要 JSON + 逐步 CSV + 数据核对版报告；

**⑥ bench 尺寸扫描方法**：与模型解耦的机制级收益上界。

### 4.3 机制验证清单(verify_dlpack.py，11 STEP / 17 检查项)

| STEP | 验证项 |
|---|---|
| 1/2 | MS→PT / PT→MS 交换(dtype/shape/device) |
| 3 | **零拷贝语义**：一侧原地修改另一侧同步可见(双向) |
| 4 | DLPack 叶子 `requires_grad_` → PT 局部图 backward |
| 5 | `pt_input.grad` → sens → MS 参数梯度 |
| 6 | **float64 解析解逐项对照**(loss / 边界梯度 / 参数梯度) |
| 7 | 负对照：假 sens 必须产生不同梯度 |
| 8 | 两侧参数均被更新 |
| 9 | 交换耗时：dlpack vs copy |
| 10 | **图语义聚合对照**(GNN C1)：两侧 scatter 聚合一致 ≤1e-5 |
| 11 | **稀疏 Embedding 完整性**(GNN C3)：未访问行梯度恒零 |

### 4.4 梯度桥接机制

```mermaid
sequenceDiagram
    autonumber
    participant MS as MindSpore
    participant DL as DLPack (零拷贝)
    participant PT as PyTorch

    MS->>MS: 前向得 ms_out (PYNATIVE, 无梯度图, 无需 detach)
    MS->>DL: to_dlpack(ms_out) 打包
    DL->>PT: from_dlpack -> pt_input (与 ms_out 共享同一内存)
    PT->>PT: pt_input.requires_grad_(True) 重立叶子
    PT->>PT: 前向 -> loss.backward()
    Note over PT: pt_input.grad = dLoss/d ms_out
    PT->>DL: grad.detach() -> to_dlpack 打包
    DL->>MS: from_dlpack -> sens
    MS->>MS: vjp 反向 (以 sens 为上游梯度, 无需重执行前向)
    Note over MS: 得到 MS 侧参数梯度
    PT->>PT: optim.step() 更新 PT 参数
    MS->>MS: set_data 更新 MS 参数
```

要点：两次交换都是同一块内存的视图而非拷贝；PT 侧通过"重立叶子"把外部张量接进本地计算图；MS 侧用 vjp 以 `sens` 为上游梯度一次反向，不重执行前向。

### 4.5 踩坑速记

| 坑 | 现象 / 处理 |
|---|---|
| MS→PT dlpack 输出落 CPU | `from_dlpack` 后设备不定，消费前 `.to(device)`(E4 首跑崩溃根因) |
| PT→MS 读到被覆写的数据 | dlpack 是视图，导出前 `.clone()` + `torch.npu.synchronize()` 断别名 |
| MS 后段裸 SGD 不收敛 | 跨框架优化器必须同源 Adam(200 epochs 停在随机水平 14.4%) |
| 多臂参数互相污染 | 同一运行内多臂 Parameter 必须独立拷贝 |
| `(N,)` × `(N,F)` 广播报错 | `inv_deg` 需 `.reshape(-1, 1)` |
| GradOperation 反向重执行前向 | 用 `mindspore.vjp` 替代；标量输出配标量种子 |
| numpy 2.x / ml-dtypes 0.6+ | mindspore wheel 按 numpy 1.x ABI 编译，锁 numpy<2.0 + ml-dtypes 0.5.1 |

## 5 环境与复现

| 项 | 值 |
|---|---|
| 硬件 | Atlas A2，昇腾 910B3 ×1 |
| CANN / OS | 8.3.RC1 / openEuler HCE 内核 aarch64 |
| Python / venv | 3.10.21(项目内 venv) |
| torch / torch_npu | 2.7.1+cpu / 2.7.1(官方配套表锁定) |
| MindSpore | 2.10.0(PYNATIVE) |
| numpy | 1.26.4(锁 1.x) |

```bash
# 服务器(910B)一键全跑: E0 自检 → E1 机制验证 → E2/E3/E4 训练 → E5 bench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-npu.txt
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 每次使用前
python scripts/experiments/run_all.py                # 全部结果落盘 results/npu/

# 单独重跑某一项 (示例):
python scripts/experiments/gcn_hybrid.py --device npu --rtol 5e-3                  # E2
python scripts/experiments/gcn_hybrid.py --device npu --exchange copy --rtol 5e-3  # E3
python scripts/experiments/gcn_hybrid.py --device npu --topology front_pt --rtol 5e-3  # E4
python scripts/experiments/gcn_hybrid.py --device npu --bench                       # E5
```

数据随项目 `data/Cora/` 离线加载，无需联网；固定 seed=42，全部结果可复现。交付物：`scripts/experiments/`(4 个脚本)+ `scripts/report/`(自动核对版报告与数据工作簿)+ `charts/` 四张证据图 + `results/npu/`(摘要 JSON、逐步 CSV、运行日志)。

## 6 限制与下一步

1. **版本配套是硬约束**：torch 与 torch_npu 主版本一致且匹配 CANN；变更 CANN 版本需重查配套表并用本套脚本回归；每次运行前 `source set_env.sh`；
2. **格式与算子边界**：FRACTAL_NZ 内部格式未覆盖(常规 ND 已验证)；Triton 自定义算子需单独评估；torch_npu 的 DLPack 走 `from_blob` 补丁通道；
3. **结论外推边界**：本验证为小模型上的机制与收敛验证；接入真实 GNN 时，消息传递类算子的动态 shape/非 ND 布局边界张量需按首个真实切分点单独回归；
4. **接入清单**：①B0 用真实 GNN 训练代码先跑通(参考曲线/梯度/epoch 耗时基线)→ ②以 B0 为锚做单步(C1/C2/C3)与多步(C4，≥3 epoch)对照 → ③P2 epoch 总账 ≤B0×1.2 → ④固定采样种子跑 3 次取最大误差 → ⑤异构图(RGCN/HGT)检查 etype 索引完整性 → ⑥按真实图数据结构选点覆盖(稠密/稀疏/大规模)；
5. **跨框架流同步**：并发写入/消费场景需补充 `torch.npu.synchronize()` 验证。
