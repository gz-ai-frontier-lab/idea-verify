# DLPack 混合过渡态验证: MindSpore → PyTorch 渐进迁移 (零拷贝交换 + 梯度桥接)

**业务背景**:业界训练框架以 PyTorch 为主,生态更丰富;基于 MindSpore 实现的存量模型存在渐进迁移诉求,且迁移周期与模型演进并行、无法一次性重写——选择 DLPack 混合编程作为**迁移过渡态**:未迁移模块留在 MS,已迁移模块改用 PT,边界两侧零拷贝交换 + sens 梯度桥接联合训练。以**真实开源模型 GCN on Cora 为载体**(图网络代表例子,不限定迁移对象本身)在 **昇腾 910B(Atlas A2)** 上验证过渡态的"敢切(等价)、选对(机制)、形态通用(对称)",交付 docs/DLPack混合训练实践报告.md。

**工作流(一条主线,NPU 全跑 → 拷回 → Windows 出图)**:

```
910B 服务器(全部实验在这一步完成)            Windows(仅图表生成, 非验证对象)
├── python scripts/experiments/run_all.py     ├── 整个 results/ 目录拷回项目根目录
│   (E0~E5 一条命令, tee 日志 + 自动 xlsx) ──▶ ├── python scripts\report\make_report.py
└── 产物: results/npu/ 摘要 + 逐步 CSV + 日志  └── 产物: charts/ 四图(交付报告 docs\DLPack混合训练实践报告.md 为手写维护)
```

**对照设计**(三要素):

- **敢切(切换等价性)**:每次运行同跑 **MS 参照臂 B0**(纯 MindSpore 两层 GCN = 唯一基线)与混合臂,判收敛 / test Accuracy / 训练曲线统计等价——混合态 vs 现状"训练效果一致";
- **选对(交换机制主对照 E2 vs E3)**:同为混合场景,唯一变量 = 交换机制(E2 = DLPack / E3 = host 拷贝)——混合前段的调度开销两跑完全相同,差值纯粹来自交换机制,归因天然干净;
- **形态通用(形态对称性 E4)**:PT前段+MS后段实测,证明切分方向不影响机制。

## 1. 目录结构

```
dlpack-verify/
├── scripts/
│   ├── experiments/            # 实验执行 (全部在 910B 上跑)
│   │   ├── run_all.py          #   一键全跑 E0~E5 (tee 时间戳日志, 末尾自动导 xlsx)
│   │   ├── check_env.py        #   E0 环境自检 (CPU/910B 通用)
│   │   ├── verify_dlpack.py    #   E1 机制验证 (11 STEP / 17 检查项, 含 GNN C1/C3)
│   │   └── gcn_hybrid.py       #   E2/E3/E4/E5 (GCN 唯一验证脚本: B0 + 混合臂 + bench)
│   └── report/                 # 交付物生成 (Windows)
│       ├── make_report.py      #   证据图表: 读 results/ 摘要 → 绘 charts/ 四图 (只出图)
│       └── export_xlsx.py      #   数据结果整合: 多 sheet Excel 工作簿 (时间戳命名)
├── data/Cora/                  # 数据集 (离线加载, 服务器免联网)
├── docs/                       # 文档
│   └── DLPack混合训练实践报告.md  # 最终交付报告 (手写维护, 博文风格; 数据源自 results/)
├── charts/                     # 报告引用的证据图 ×4 (make_report.py 生成)
├── results/                    # 运行产物 (脚本自动落盘, 按设备分目录)
│   ├── npu/                    #   910B: 摘要 JSON + 逐步 CSV + run_all_<时间戳>.log
│   │                           #        + smi_<阶段>_<时间戳>.csv (P1 采样存证)
│   │                           #        + results_<时间戳>.xlsx (多 sheet 数据工作簿)
│   └── cpu/                    #   本地调试产物
├── requirements-cpu.txt        # Windows CPU 环境依赖 (已实测锁定)
├── requirements-npu.txt        # 910B 环境依赖 (已按 CANN 8.3.RC1 锁定)
├── .gitignore
└── README.md                   # 本文件
```

## 2. 环境要求

| 项 | 910B(验证对象) | Windows(报告生成/冒烟, 非验证对象) |
|---|---|---|
| 硬件 | Atlas A2, 昇腾 910B3 ×1 | x64 CPU |
| CANN | **8.3.RC1**(version.cfg 确认) | — |
| Python | 3.10.21(项目内 venv) | 3.10.11 官方构建 |
| torch / torch_npu | 2.7.1 / 2.7.1(官方配套表锁定) | 2.5.1+cpu |
| MindSpore | 2.10.0 | 2.11.0(2.7.0 起恢复 Windows 包) |
| numpy | 1.26.4(**必须 <2.0**,见 §9 踩坑) | 1.26.4 |
| torch_geometric | 纯 Python 安装(无需编译扩展) | 同左 |

依赖清单:`requirements-npu.txt`(服务器)/ `requirements-cpu.txt`(Windows),版本已实测锁定,直接 `pip install -r` 即可。

## 3. 脚本一览(全部脚本集中在此)

| 脚本 | 作用 | 运行位置 |
|---|---|---|
| `scripts/experiments/run_all.py` | **一键全跑 E0~E5**(tee 时间戳日志,末尾自动导 xlsx,P1 npu-smi 逐卡采样) | **910B(主入口)** |
| `scripts/experiments/check_env.py` | E0 环境自检(torch_npu/NPU/CANN/双向 DLPack 冒烟;run_all 已含) | 910B(可单跑) |
| `scripts/experiments/verify_dlpack.py` | E1 机制验证(11 STEP / 17 检查项,含 GNN C1/C3) | 910B(单独重跑时) |
| `scripts/experiments/gcn_hybrid.py` | E2/E3/E4/E5 唯一训练验证脚本(B0 参照臂 + 混合臂 + bench) | 910B(单独重跑时) |
| `scripts/report/make_report.py` | **证据图表生成**(只出图):读 results/ → 绘 charts/ 四图 | **Windows** |
| `scripts/report/export_xlsx.py` | 多 sheet 数据工作簿(run_all 末尾自动调用,可独立运行) | 两者皆可 |

## 4. 执行流程

### 4.1 第一步:910B 服务器一键全跑(全部实验在这一步完成)

首次准备:

```bash
# 1. 整目录上传后, 建独立 venv (不动环境已有包):
python -m venv .venv && source .venv/bin/activate
# 2. 安装锁定版本依赖:
pip install -r requirements-npu.txt
# 3. 每次使用前激活 CANN 运行时 (容易漏!):
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

一键全跑(推荐,一条命令出全部结果与日志):

```bash
python scripts/experiments/run_all.py
# 本地快速冒烟(可选): python scripts/experiments/run_all.py --device cpu --epochs 20 --rtol 1e-3
```

顺序执行:E0 自检 → E1 机制验证 → E2/E3/E4 GCN 训练(每次运行同跑 B0 参照臂)→ E5 bench;实时回显并 tee 到 `results/npu/run_all_<时间戳>.log`(每阶段有横幅标注);任一阶段失败立即停止(E4* 为 NPU 调试项,FAIL 不阻塞后续阶段);全部通过后自动导出 `results_<时间戳>.xlsx` 多 sheet 数据工作簿。

**实验矩阵**(run_all 内容 = 下表全部;单独重跑某项时用对应命令,在项目根目录、已激活 venv 与 CANN 环境):

| 编号 | 实验 | 命令 | 变化量 | 回答的问题 |
|---|---|---|---|---|
| E0 | 环境自检 | `python scripts/experiments/check_env.py` | — | torch_npu/NPU/CANN 全 OK |
| E1 | 机制验证(11 STEP / 17 检查项,含 GNN C1/C3) | `python scripts/experiments/verify_dlpack.py --device npu --rtol 5e-3` | — | 零拷贝/桥接数学正确 |
| **E2** | **GCN 混合训练(dlpack)** | `python scripts/experiments/gcn_hybrid.py --device npu --rtol 5e-3` | —(基准) | 收敛/切换等价性/C2 双锚 |
| **E3** | **GCN 混合训练(copy 对照)** | `python scripts/experiments/gcn_hybrid.py --device npu --exchange copy --rtol 5e-3` | **仅交换方式: dlpack→host 拷贝** | **DLPack 交换机制的直接收益(主对照)** |
| **E4** | **GCN 形态对称性(PT前段+MS后段)** | `python scripts/experiments/gcn_hybrid.py --device npu --topology front_pt --rtol 5e-3` | **仅混合形态: MS前段→PT前段** | 对称形态机制成立(切分方向不影响) |
| E5 | 纯交换吞吐(尺寸扫描) | `python scripts/experiments/gcn_hybrid.py --device npu --bench`(默认节点 2708/16384/65536,`--bench-sizes` 自定) | 仅 bench 模式 + 边界张量尺寸档 | 收益随边界张量放大趋势(机制级上界) |

**固定参数**(Kipf 原始 recipe,所有实验一致,勿随意改动):

| 参数 | 值 | 说明 |
|---|---|---|
| `--epochs` | 200 | 训练轮数(全批) |
| `--hidden` | 16 | 隐层宽度(Kipf 设置) |
| `--lr` | 0.01 | Adam 学习率 |
| `--weight-decay` | 5e-4 | L2 正则(两侧 Adam 严格同源实现) |
| `--seed` | 42 | 同初始化/同数据划分 |
| `--rtol` | 5e-3(NPU)/ 1e-3(CPU) | C2 梯度双锚容差(B0 锚独立放宽至 max(rtol, 5e-3)) |

**数据落盘密度**:训练 loss / val_acc / test_acc 每 epoch 全量写逐步 CSV(200 点完整可审计);图表按**每 5 epoch** 取样绘制(约 40 点)。

**对照关系与归因原则(重要)**:

- **主对照 = E2 vs E3**:同为混合场景(MS conv1 + PT conv2、同切分点、同数据同种子),唯一变量是交换机制——这是"DLPack 本身价值"的直接证据;
- **归因原则**:混合前段的 MindSpore 调度开销在 E2/E3 两跑中完全相同,因此 E2 vs E3 的差值纯粹来自交换机制,归因天然干净;
- 混合训练的**绝对** epoch 耗时主体为 MS PYNATIVE 单算子调度常数,与交换无关,该常数随模型规模增大被摊薄,生产规模按占比评估(接入真实模型时回归);
- E4 的 epoch 耗时/交换段**不与 E2 直接对比**(两形态框架分工不同):front_pt 把聚合+conv1 放到 PT 侧,MS 调度常数占比大减,epoch 时延显著变短属正常;E4 交换发生在 PT→MS(激活)/MS→PT(梯度)双向,NPU 上导出前 clone + 流同步屏障是防别名污染的必要成本。E4 的验证目标是**机制对称性**,非性能对照。

### 4.2 第二步:拷回 Windows 一键出报告(最后一步)

```bash
# 服务器跑完后, 把整个 results/ 目录拷回 Windows 项目根目录, 然后:
.venv\Scripts\activate.bat
python scripts\report\make_report.py
```

产出:`charts/` 四张证据图(GCN 三臂曲线 / E2vE3 主对照 / 判据评分卡 / bench 尺寸扫描)。交付报告 docs\DLPack混合训练实践报告.md 为**手写维护**,make_report 只出图、不生成任何文档;数据工作簿 `results\<时间戳>.xlsx` 已由服务器端 run_all 末尾自动导出,如需补导可运行 `python scripts\report\export_xlsx.py`。

`make_report.py` 读取 `results/{npu|cpu}/` 下的 `verify_summary.json`、`gcn_summary_{dlpack|copy|dlpack_sym}.json`、`bench_summary.json` 与逐步训练 CSV,**数据优先级 npu > cpu**(有真机摘要时绝不使用 cpu 数据);缺项时对应图表以"待复测"占位,并在控制台列出补跑命令。

## 5. 判读要点(判据体系:正确性 C1~C4 / 性能 P1~P2,对齐行业咨询口径)

**基线**:B0 = MS 参照臂(纯 MindSpore 两层 GCN,每次运行同跑)为**唯一基线**;B1(纯 PyTorch)不做——那是迁移终点的实现工作,不在验证范围。

正确性(以 B0 为锚):

- **C1 图语义聚合一致(必过)**:交换边界两侧各做一次 scatter 聚合,结果一致 ≤1e-5(E1 STEP 10)——不只比数值,要证明聚合语义未被交换破坏;
- **C2 梯度等价性(双锚)**:解析解/torch 参考(M4 torch 锚)+ **B0 纯 MS 端到端**(M4 B0 锚),相对误差 < rtol(B0 锚容差独立放宽 max(rtol, 5e-3),GCN 数值敏感);
- **C3 稀疏参数完整性**:交换路径上未访问 Embedding 行梯度恒零、稀疏更新后零变化(E1 STEP 11);
- **C4 收敛曲线一致性(必须项)**:GCN 规模下逐位一致不可达——一致性口径 = **ulp 级 ≤1e-5 + Accuracy 曲线趋势重合**(浮点归约顺序噪声,非交换差异);混合 vs B0 为**统计等价**:loss 比 ∈[0.5,2.0]、test Accuracy 差 ≤1.5pt、曲线形态一致;过拟合检查同步进行;

性能:

- **P1 交换信噪比**:交换段占单 epoch 训练耗时比例(<1% 为佳);GCN on Cora 为小图全批训练,单 epoch 仅几~几百 ms,占比天然偏高,以真实 GNN 单 batch 口径评估;
- **P2 epoch 耗时总账**:接入真实 GNN 代码后 ≥3 epoch 耗时 ≤B0×1.2 为 PASS;本验证阶段按交换占比摊薄披露;

采样存证(不作为判据,报告中不呈现):run_all 在 E2/E3/E4 阶段自动并行采样 npu-smi 落盘 `results/npu/smi_<阶段>_<时间戳>.csv`(1 秒间隔,逐卡采样全部可见 card,原始输出截断存 raw 列);GCN 小图全批训练算子细碎,AI Core 利用率本征接近 0,不作判定;真实 GNN 接入时按需启用"AI Core 长时间近零 + HBM 非预期回落"检测。

收益证据(非门禁):

- E2 vs E3 机制选型对照:交换段收益、epoch 时延收益(bench 为机制级上界);
- **形态对称性**:E4(PT前段+MS后段)M4 双锚 + 收敛 + Accuracy 全通过即成立——切分方向不影响机制,两种形态都适用。

## 6. verify_dlpack.py 的验证项 ↔ 验收标准映射

| STEP | 验证项 | 对应验收标准 |
|---|---|---|
| 1 | MS→PT 交换 (dtype/shape/device) | 第一层-2 |
| 2 | PT→MS 交换 | 第一层-3 |
| 3 | **零拷贝语义** (一侧原地修改另一侧可见,双向) | 第一层-4(核心) |
| 4 | DLPack 叶子 `requires_grad_` → PT 局部图 backward | 第二层-6 |
| 5 | `pt_input.grad` → sens → `GradOperation(sens_param=True)` | 第二层-7 |
| 6 | **float64 解析解逐项对照**(loss、pt_input.grad、MS 参数梯度) | 第二层-7(数值≤容差,不止"非零") |
| 7 | 负对照:ones_like 假 sens 产生不同梯度 | 第二层-8 |
| 8 | 两侧参数均被更新 | 第二层-9 |
| 9 | 交换耗时:dlpack 路径 vs 拷贝路径 | NFR-1 |
| 10 | **图语义聚合对照**:交换边界两侧各做一次 scatter 聚合(unsorted_segment_sum / index_add_),结果一致 ≤1e-5 | GNN C1(必过) |
| 11 | **稀疏 Embedding 完整性**:交换路径上未访问行梯度恒零、稀疏更新后零变化 | GNN C3 |

## 7. 训练验证设计与冒烟基准(gcn_hybrid.py)

模型:Kipf & Welling 两层 GCN + PyG Planetoid **Cora**(2708 节点/10556 边/1433 维特征/7 分类,public split 140/500/1000;数据随项目 `data/` 离线加载)。切分点 = conv1 之后(边界 = 节点表示,真实 GNN 典型切分位置);B0 = 纯 MindSpore 两层 GCN(内置于每次运行);优化器两侧严格同源(Kipf recipe:Adam lr=0.01、weight_decay=5e-4、200 epochs 全批)。

判据:C4 收敛一致性用 **test Accuracy**(GCN 规模下逐位一致不可达,一致性口径 = ulp 级 ≤1e-5 + Accuracy 趋势重合——浮点归约顺序噪声,非交换差异);C2 梯度双锚 B0 锚容差独立放宽至 max(rtol, 5e-3)(1433 维特征两层传播会放大跨框架 ulp 差,torch 锚证明桥接无损)。

执行命令:见 §4.1 实验矩阵(E2/E3/E4/E5);数据集离线加载无需联网。

CPU 冒烟基准(200 epochs):混合 test_acc **80.3%** vs B0 **80.3%**(acc 差 0.00pt,≤1.5pt 判据 PASS)——复现 Kipf 论文公开水平(~81.5%);E2/E3 一致性 ulp 级 ≤5.5e-7。910B 真机图表由 `make_report.py` 生成(图 1~4),交付报告手写维护。

机制级补充:PYNATIVE GradOperation 反向会重执行一次前向(每步 2 次 MS 前向,固有行为)。已实装 `mindspore.vjp` 模式消除冗余前向(注意官方用法是 `vjp(fn, *inputs)`;标量输出可用标量种子 `bwd_fn(ms.Tensor(1.0))` 一次得全部输入梯度)。

## 8. 910B (Atlas A2) 环境说明与已知风险

### 8.1 版本配套

1. 服务器查 CANN:`cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg`
2. 查官方配套表:github.com/Ascend/pytorch 仓库 `COMPATIBILITY.md`(实测环境 CANN 8.3.RC1 → torch_npu 2.8.0/2.7.1/2.6.0.post3/2.1.0.post17,本验证锁定 torch 2.7.1 + torch_npu 2.7.1)
3. torch 与 torch_npu **主版本必须一致**;变更 CANN 版本用本套脚本回归

### 8.2 关键行为差异(CPU → NPU)

| 项 | CPU 本机 | 910B 上 |
|---|---|---|
| 设备 | `--device cpu` | `--device npu`(自动 `import torch_npu`) |
| MindSpore | `device_target="CPU"` | `device_target="Ascend"`(脚本自适应) |
| 原生 torch DLPack | 可用 | **A2 标记 Not Supported,必须走 torch_npu 补丁** |
| sens 传递 | CPU 内存共享 | NPU 显存共享;导出报错时回退 `.cpu().numpy()` 中转 |
| 流同步 | 单流,无感 | 数值偶发不一致时加 `torch.npu.synchronize()` |

### 8.3 已知风险与无害告警

1. **torch_npu 的 DLPack 走 `from_blob` 补丁**(310 NZ TODO);A2 常规 ND 格式已验证,FRACTAL_NZ 格式需先转 ND。
2. **torch 导出带梯度张量必须 `.detach()`**;MindSpore 侧 PYNATIVE 无梯度上下文时直接导出即可。
3. **NNAL/ATB 已知 issue**(Ascend/pytorch#99):vLLM/ATB 场景限定,本验证不受影响。
4. **无害告警**(脚本已内置精确过滤):torch_npu `owner does not match`(CANN root 安装 vs 普通用户)、numpy `smallest subnormal is zero`(ARM 容器关闭 denormal,与验证数值量级差 30 倍)。
5. **真机结论**(2026-11-23):机制验证 17/17 PASS;PT→MS 共享 **NPU device 显存**(torch_npu 补丁通道),MS→PT 共享 host 内存(MS PYNATIVE 张量驻留 host,算子消费时才上 device)——两者均为真零拷贝。
6. **行业图网络模型(GNN)接入**:消息传递类算子(scatter/gather/变长邻接结构)的边界张量可能为动态 shape/非常规布局,按首个真实切分点单独回归;图网络的采样 batch 可调,等价性/对称性判据可直接复用。

## 9. 常见坑速查

| 现象 | 原因 | 处理 |
|---|---|---|
| 提示符显示 venv 名但 import 全报缺包 | 幽灵激活(激活的 venv 已删除),回落到 base | 关终端重开重新激活;`where python` 确认路径 |
| `pip install -r` 报 GBK 解码错误 | requirements 含非 ASCII 注释 | 保持 requirements 纯 ASCII(本仓库已是) |
| `torch.utils.dlpack` 对 NPU 张量报错 | 忘了 `import torch_npu` | 任何 torch 调用前先 import |
| torch 导出报 `requires gradient` | 带 grad 张量直接导出 | `.detach()` 后导出 |
| MS 梯度"对但不对" | sens 用了默认 ones | STEP 7 负对照抓这个 |
| 数值偶发不一致(910B) | 跨框架流未同步 | 消费前 `torch.npu.synchronize()` |
| dlpack 交换的张量落在 CPU(910B) | MS→PT dlpack 输出设备不定 | `from_dlpack(...)` 后 `.to(device)`(本仓库已内置) |
| pip 找不到 mindspore Windows 包 | 2.4~2.6 无 Windows wheel | 选 2.11.0(或 ≥2.7)+ py3.10 |
| `ms.jit` 包装 Cell 后权重不更新 | jit 编译图固化编译时权重值 | 函数式前端 + 权重作参数传入(本仓库已如此实现) |
| `mindspore.vjp` 多参数输入报缺参 | vjp(fn) 未注册 inputs | 官方用法:`vjp(fn, x, w, b)` 返回 (out, bwd_fn),bwd_fn 只收种子向量 |
| 标量 loss 的 vjp | 标量输出需标量种子 | `bwd_fn(ms.Tensor(1.0))` 一次得全部输入梯度(CPU/NPU 均可用) |
| MS Parameter 用 torch Tensor 索引报错 | 跨框架索引混用 | MS 侧用 `ops.gather`,PT 侧用 `tensor[idx]`,不要混 |
| npu-smi `-i 0` 报 Invalid card id | 910B 服务器 card id 可能不从 0 起 | 采样器已自动发现 card 并逐卡采样 |
