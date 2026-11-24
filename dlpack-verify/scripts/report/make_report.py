# -*- coding: utf-8 -*-
"""
make_report.py — 证据图表生成 (只出图)
========================================

流程:
    1. 读取 results/{npu|cpu}/ 下的机器可读摘要
       (verify_summary.json / gcn_summary_{dlpack,copy[,dlpack_sym]}.json /
        bench_summary.json) 与逐步训练 CSV (gcn_log_*.csv)
    2. 用实测数据绘制 charts/ 四张证据图
       (GCN 三臂对照曲线 / E2 vs E3 交换机制主对照 / 判据评分卡 / 纯交换吞吐尺寸扫描)

说明:
    - 交付报告 docs/DLPack混合训练实践报告.md 为手写维护, 本脚本不生成/不覆盖任何文档;
    - 数据优先级: npu > cpu;
    - 数据工作簿(xlsx)由 scripts/report/export_xlsx.py 生成 (run_all 末尾已自动调用)。

用法: python scripts/report/make_report.py
(实验本身在 910B 上运行 scripts/experiments/run_all.py 一键完成)
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHARTS_DIR = os.path.join(PROJECT, "charts")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # 中文
plt.rcParams["axes.unicode_minus"] = False

C_B, C_COPY = "#1f77b4", "#ff7f0e"
C_OK, C_BAD = "#2ca02c", "#d62728"
C_GRAY = "#7f7f7f"

# ---- 数据设备 (main() 按摘要存在性确定 npu/cpu) ----
DEV = "npu"

# ---- 图表数据源 ----
GCN = {"e2": None, "e3": None, "e4": None}   # norm_gcn 视图, main() 填充
BENCH = []   # bench 尺寸行 (nodes/boundary/四项耗时); main() 用 bench_summary 覆盖


def bench_label(r):
    """bench 行标签: v2 = boundary (nodes=n); v1 兜底 = batch=n。"""
    if r.get("boundary"):
        return (f"{r['boundary']} (nodes={r['nodes']})"
                if r.get("nodes") else str(r["boundary"]))
    return f"batch={r.get('batch', '?')}"


def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def load_summaries(dev):
    d = os.path.join(PROJECT, "results", dev)
    j = lambda name: load_json(os.path.join(d, name))
    return {
        "verify": j("verify_summary.json"),
        "gcn_e2": j("gcn_summary_dlpack.json"),
        "gcn_e3": j("gcn_summary_copy.json"),
        "gcn_e4": j("gcn_summary_dlpack_sym.json"),
        "bench": j("bench_summary.json"),
    }


def norm_gcn(s):
    """GCN 摘要 -> 统一视图 (E2/E3/E4 通用)。"""
    if not s:
        return None
    r = s.get("results", {})

    def g(k):
        v = r.get(k)
        return float(v) if isinstance(v, (int, float)) else None

    cal = s.get("m4_calibration", {}) or {}
    ref = s.get("ms_ref") or {}
    eq = s.get("eq") or {}

    def rg(k):
        v = ref.get(k)
        return float(v) if isinstance(v, (int, float)) else None

    def ng(d, k):
        v = d.get(k)
        return float(v) if isinstance(v, (int, float)) else None

    return {
        "exchange": s.get("exchange"), "topology": s.get("topology", "front_ms"),
        "final_loss": g("final_loss"), "final_val_acc": g("final_val_acc"),
        "final_test_acc": g("final_test_acc"), "lat_ms": g("lat_ms"),
        "exf_us": g("exchange_fwd_us"), "exb_us": g("exchange_bwd_us"),
        "ratio_pct": g("exchange_ratio_pct"),
        "m4_err": (float(cal["rel_err"]) if isinstance(cal.get("rel_err"), (int, float)) else None),
        "m4_ms_err": (float(cal["ms_rel_err"]) if isinstance(cal.get("ms_rel_err"), (int, float)) else None),
        "m4_pass": cal.get("pass"),
        "converged": r.get("converged"),
        "b0_loss": rg("final_loss"), "b0_test_acc": rg("final_test_acc"),
        "b0_converged": ref.get("converged"),
        "eq_ratio": ng(eq, "final_ratio"), "eq_gap": ng(eq, "acc_gap_pt"),
        "eq_pass": eq.get("pass"),
    }


def norm_bench(b):
    """bench 摘要 -> 尺寸行列表 (兼容 v1 单尺寸扁平格式); 缺摘要返回空列表。"""
    if not b:
        return []
    if isinstance(b.get("sizes"), list) and b["sizes"]:
        return b["sizes"]
    return [{"boundary": "基准尺寸", "nodes": "",
             "fwd_dlpack_us": b["fwd_dlpack_us"], "fwd_copy_us": b["fwd_copy_us"],
             "bwd_dlpack_us": b["bwd_dlpack_us"], "bwd_copy_us": b["bwd_copy_us"]}]


PLOT_STRIDE = 5   # 曲线绘图取样间隔(步); 原始逐步数据仍每步全量落盘 CSV


def decimate(xs, *arrays):
    """按 PLOT_STRIDE 取样绘图数据(保留首末点); 点数 <=50 时原样返回。"""
    n = len(xs)
    if n <= 50:
        return (list(xs), *[list(a) for a in arrays])
    idx = list(range(0, n, PLOT_STRIDE))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return ([xs[i] for i in idx], *[[a[i] for i in idx] for a in arrays])


def load_csv_curve_gcn(path):
    """GCN 逐步 CSV (epoch, train_loss, val_acc, test_acc, ref_loss,
    ref_val_acc, ref_test_acc, t_ms) -> dict; 缺列返回 None。"""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "train_loss" not in rows[0]:
        return None

    def col(name):
        out = []
        for row in rows:
            v = row.get(name, "")
            out.append(float(v) if v not in (None, "") else None)
        return out

    return {"epoch": [int(r["epoch"]) for r in rows], "train_loss": col("train_loss"),
            "test_acc": col("test_acc"), "ref_loss": col("ref_loss"),
            "ref_test_acc": col("ref_test_acc")}


# ---------------------------------------------------------------------------
# 图 1: MS 参照 + E2 + E3 训练/验证曲线 (收敛一致性 + 泛化/过拟合 + 切换等价性证据)
# ---------------------------------------------------------------------------
def fig_gcn():
    csv_dir = os.path.join(PROJECT, "results", DEV)
    csv_path = os.path.join(csv_dir, "gcn_log_dlpack.csv")
    csv_c = os.path.join(csv_dir, "gcn_log_copy.csv")
    d = load_csv_curve_gcn(csv_path) if os.path.exists(csv_path) else None
    c = load_csv_curve_gcn(csv_c) if os.path.exists(csv_c) else None
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=150,
                             gridspec_kw={"height_ratios": [1.4, 1.2]})
    ax = axes[0]
    if d:
        rx, ry = decimate(d["epoch"], d["train_loss"])
        ax.plot(rx, ry, "-", color=C_B, lw=1.6, label="E2 混合 dlpack 训练 loss",
                zorder=3)
    if c:
        rx, ry = decimate(c["epoch"], c["ref_loss"])
        ax.plot(rx, ry, "-", color=C_GRAY, lw=3.2, alpha=0.45,
                label="纯MindSpore 基线 B0 训练 loss", zorder=2)
        rx, ry = decimate(c["epoch"], c["train_loss"])
        ax.plot(rx, ry, "--", color=C_COPY, lw=1.6, label="E3 混合 copy 训练 loss",
                zorder=4)
    ax.set_ylabel("训练 loss (交叉熵)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("GCN on Cora (真实开源模型, PyG Planetoid) — 三臂对照", fontsize=10)
    ax2 = axes[1]
    if c:
        rx, ry = decimate(c["epoch"], c["ref_test_acc"])
        ax2.plot(rx, ry, "-", color=C_GRAY, lw=2.2, label="B0 纯MindSpore 基线")
    if d:
        rx, ry = decimate(d["epoch"], d["test_acc"])
        ax2.plot(rx, ry, "-", color=C_B, label="E2 混合 dlpack")
    if c:
        rx, ry = decimate(c["epoch"], c["test_acc"])
        ax2.plot(rx, ry, "--", color=C_COPY, label="E3 混合 copy")
    ax2.set_ylabel("test Accuracy")
    ax2.set_ylim(0.5, 0.9)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_xlabel("epoch")
    ax2.set_title("test Accuracy (C4 判据: GCN 规模下 ulp 级一致 ≤1e-5, 趋势重合)",
                  fontsize=9)
    if not d and not c:
        ax.text(0.5, 0.5, "GCN 曲线待复测\nrun_all 重跑后生成",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="gray")
    fig.suptitle("真实开源模型 GCN on Cora — 混合过渡态 vs 基线 (910B 真机)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p_gcn = os.path.join(CHARTS_DIR, "gcn_npu.png")
    fig.savefig(p_gcn)
    plt.close(fig)
    return p_gcn


# ---------------------------------------------------------------------------
# 图 2: E2 vs E3 主对照 — 交换段耗时 (log) + epoch 耗时 (唯一变量=交换机制)
# ---------------------------------------------------------------------------
def fig_e2e3():
    e2, e3 = GCN.get("e2"), GCN.get("e3")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=150)
    ax = axes[0]
    have_ex = all(isinstance((e or {}).get(k), (int, float))
                  for e in (e2, e3) for k in ("exf_us", "exb_us"))
    if have_ex:
        vals_d = [e2["exf_us"], e2["exb_us"]]
        vals_c = [e3["exf_us"], e3["exb_us"]]
        xs = np.arange(2)
        w = 0.36
        b1 = ax.bar(xs - w / 2, vals_d, w, color=C_B, label="E2 混合 dlpack")
        b2 = ax.bar(xs + w / 2, vals_c, w, color=C_COPY, label="E3 混合 copy")
        ax.bar_label(b1, fmt="%.1f", fontsize=8)
        ax.bar_label(b2, fmt="%.1f", fontsize=8)
        for i, (d, c) in enumerate(zip(vals_d, vals_c)):
            ax.text(i, max(c, d) * 2.2, f"copy 慢 {c / max(d, 1e-9):.1f}x",
                    ha="center", fontsize=9, color=C_BAD, fontweight="bold")
        allv = vals_d + vals_c
        ax.set_yscale("log")
        ax.set_ylim(min(allv) * 0.5, max(allv) * 9)
        ax.set_xticks(xs)
        ax.set_xticklabels(["正向 MS→PT", "回传 PT→MS"], fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "E2/E3 交换段待复测", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
    ax.set_ylabel("单 epoch 交换段耗时 (µs, log 刻度)")
    ax.set_title("交换段耗时 (µs/epoch)", fontsize=10)
    ax2 = axes[1]
    l2, l3 = (e2 or {}).get("lat_ms"), (e3 or {}).get("lat_ms")
    if isinstance(l2, (int, float)) and isinstance(l3, (int, float)):
        bars = ax2.bar(["E2 混合 dlpack", "E3 混合 copy"], [l2, l3],
                       color=[C_B, C_COPY], width=0.5)
        ax2.bar_label(bars, fmt="%.1f ms", fontsize=9, fontweight="bold")
        ax2.set_ylim(0, max(l2, l3) * 1.25)
        ax2.grid(alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "epoch 耗时待复测", transform=ax2.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
    ax2.set_ylabel("epoch 耗时 (ms, 去预热)")
    ax2.set_title("epoch 耗时 (主体为框架调度常数, 两跑相同)", fontsize=10)
    fig.suptitle("交换机制主对照 E2 vs E3 — 同为混合场景, 唯一变量 = 交换机制 (910B 真机)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(CHARTS_DIR, "e2e3_gcn.png")
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# 图 3: 验证判据评分卡 (C2-M4 / C4 机制间一致 / 收益·步时延 / E5 加速比)
# ---------------------------------------------------------------------------
def fig_scorecard():
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), dpi=150)
    fig.subplots_adjust(left=0.07, right=0.945, top=0.9, bottom=0.17,
                        hspace=0.5, wspace=0.26)
    fig.suptitle("验证判据评分卡 (910B 真机)", fontsize=12, y=0.965)
    names = ["GCN on Cora"]
    xs = np.arange(len(names))

    def set_ticks(ax):
        ax.set_xticks(xs)
        ax.set_xticklabels(names, fontsize=8)

    def na_mark(ax, i, y):
        ax.text(i, y, "待复测", ha="center", va="center", fontsize=9,
                color="gray")

    # M4: 边界梯度相对误差 (log, 阈值 rtol)
    ax = axes[0][0]
    vals = [(GCN["e2"] or {}).get("m4_err")]
    rtol = (GCN["e2"] or {}).get("rtol") or 5e-3
    for i, v in enumerate(vals):
        if not isinstance(v, (int, float)):
            na_mark(ax, i, 2e-4)
            continue
        hgt = max(v, 5e-7)
        ax.bar(i, hgt, width=0.5, color=C_OK)
        ax.text(i, hgt * 1.6, "0.0(逐位一致)" if v == 0 else f"{v:.2e}",
                ha="center", fontsize=9)
    ax.axhline(rtol, color=C_BAD, ls="--", lw=1)
    ax.text(len(names) - 0.55, rtol * 1.4, f"rtol={rtol:g}", fontsize=8,
            color=C_BAD)
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 3e-2)
    set_ticks(ax)
    ax.set_ylabel("最大相对误差 (log)")
    ax.set_title("C2-M4 边界数值 (torch 锚, B0 锚见报告 §3.6)", fontsize=10)
    ax.grid(alpha=0.3, axis="y", which="both")

    # C4: E3/E2 final loss 比 (期望 1.000)
    ax = axes[0][1]
    e2, e3 = GCN["e2"], GCN["e3"]
    vals = [e3["final_loss"] / e2["final_loss"]
            if e2 and e3 and e2.get("final_loss") and e3.get("final_loss") else None]
    for i, v in enumerate(vals):
        if not isinstance(v, (int, float)):
            na_mark(ax, i, 0.9955)
            continue
        ax.bar(i, v, width=0.5, color=C_OK)
        ax.text(i, v + 0.0002, f"{v:.6f}", ha="center", fontsize=9)
    ax.axhspan(0.995, 1.005, color=C_OK, alpha=0.13)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_ylim(0.99, 1.01)
    set_ticks(ax)
    ax.set_ylabel("E3/E2 final loss 比")
    ax.set_title("C4 机制间一致 (E3/E2 final loss ≈ 1.0)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # 收益: E3/E2 步时延比 (>1 = dlpack 更快)
    ax = axes[1][0]
    vals = []
    pairs_ms = []
    e2, e3 = GCN["e2"], GCN["e3"]
    if e2 and e3 and e2.get("lat_ms") and e3.get("lat_ms"):
        vals = [e3["lat_ms"] / e2["lat_ms"]]
        pairs_ms = [(e2["lat_ms"], e3["lat_ms"])]
    else:
        vals = [None]
        pairs_ms = [None]
    top = max([v for v in vals if isinstance(v, (int, float))] or [1.5]) * 1.3
    for i, v in enumerate(vals):
        if not isinstance(v, (int, float)):
            na_mark(ax, i, top * 0.5)
            continue
        ax.bar(i, v, width=0.5, color=C_OK)
        ax.text(i, v + top * 0.03, f"{v:.2f}", ha="center", fontsize=10,
                fontweight="bold")
        ms = pairs_ms[i]
        ax.text(i, top * 0.06, f"{ms[0]:.2f} vs {ms[1]:.2f} ms", ha="center",
                fontsize=8, color="white", fontweight="bold")
    ax.axhspan(1.0, top, color=C_OK, alpha=0.10)
    ax.axhline(1.0, color=C_BAD, ls="--", lw=1)
    ax.set_ylim(0, top)
    set_ticks(ax)
    ax.set_ylabel("E3/E2 步时延比 (>1 = dlpack 更快)")
    ax.set_title("收益·步时延 (E3/E2 比, >1 = dlpack 更优)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # E5: bench 加速比 (copy/dlpack, 按边界张量尺寸扫描)
    ax = axes[1][1]
    if not BENCH:
        ax.text(0.5, 0.5, "bench 待复测\nrun_all E5 重跑后生成",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="gray")
    else:
        labels = [bench_label(r) for r in BENCH]
        fwd = [r["fwd_copy_us"] / max(r["fwd_dlpack_us"], 1e-9) for r in BENCH]
        bwd = [r["bwd_copy_us"] / max(r["bwd_dlpack_us"], 1e-9) for r in BENCH]
        xs5 = np.arange(len(labels))
        w5 = 0.36
        b1 = ax.bar(xs5 - w5 / 2, fwd, w5, color=C_OK, label="正向 MS→PT")
        b2 = ax.bar(xs5 + w5 / 2, bwd, w5, color="#74c476", label="回传 PT→MS")
        ax.bar_label(b1, fmt="%.1fx", fontsize=8, fontweight="bold")
        ax.bar_label(b2, fmt="%.1fx", fontsize=8, fontweight="bold")
        ax.axhline(1.0, color=C_BAD, ls="--", lw=1)
        ax.set_ylim(0, max(fwd + bwd) * 1.3)
        ax.set_xticks(xs5)
        ax.set_xticklabels(labels, fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylabel("copy/dlpack 耗时比 (>1 = dlpack 更快)")
        ax.set_title("E5 纯交换加速比 (--bench 尺寸扫描, 机制级上界)", fontsize=10)
        ax.grid(alpha=0.3, axis="y")

    fig.text(0.01, 0.008,
             "说明: 混合训练绝对 epoch 耗时的主体为 MindSpore PYNATIVE 单算子调度常数, "
             "该常数在 E2/E3 两种交换机制下完全相同 —— 主对照差值纯粹来自交换机制。\n"
             "该绝对开销随模型规模增大被摊薄 (E4 实测见报告 §3.8), 生产规模按占比评估。",
             fontsize=7.5, va="bottom",
             bbox=dict(boxstyle="round", fc="#fff7e6", alpha=0.9))
    p = os.path.join(CHARTS_DIR, "metrics_scorecard.png")
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# 图 4: 纯交换吞吐对比 (E5 --bench 尺寸扫描)
# ---------------------------------------------------------------------------
def fig_bench():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, squeeze=False)
    groups = [bench_label(r) for r in BENCH]
    keys = [("fwd_dlpack_us", "fwd_copy_us", "正向 MS→PT 交换 (单次)"),
            ("bwd_dlpack_us", "bwd_copy_us", "回传 PT→MS 交换 (单次)")]
    for k, (dk, ck, title) in enumerate(keys):
        ax = axes[0][k]
        if not BENCH:
            ax.text(0.5, 0.5, "bench 待复测\nrun_all E5 重跑后生成",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="gray")
            continue
        dl = [r[dk] for r in BENCH]
        cp = [r[ck] for r in BENCH]
        xs = np.arange(len(BENCH))
        w = 0.32
        b1 = ax.bar(xs - w / 2, dl, w, color=C_B, label="dlpack (零拷贝, 留在 NPU 显存)")
        b2 = ax.bar(xs + w / 2, cp, w, color=C_COPY, label="copy (NPU→host→NPU 往返)")
        ax.set_yscale("log")
        ax.set_ylim(min(min(cp + dl) * 0.5, 5), max(cp + dl) * 12)
        ax.bar_label(b1, fmt="%.1f", fontsize=8)
        ax.bar_label(b2, fmt="%.1f", fontsize=8)
        for i, (d, c) in enumerate(zip(dl, cp)):
            ax.text(i, max(c, d) * 2.2, f"copy 慢 {c / max(d, 1e-9):.1f}x",
                    ha="center", fontsize=9, color=C_BAD, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(groups, fontsize=8)
        ax.set_ylabel("单次交换耗时 (µs, log 刻度)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y", which="both")
    fig.suptitle("纯交换吞吐对比 — 边界张量尺寸扫描 (E5 --bench, 910B 真机, 同进程)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(CHARTS_DIR, "bench_npu.png")
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    global DEV
    # ---- 数据设备选择: npu 优先, 无 npu 摘要时回退 cpu ----
    dev = "npu"
    s = load_summaries(dev)
    if not s["verify"]:
        dev = "cpu"
        s = load_summaries(dev)
    if not s["verify"]:
        print("[FATAL] results/ 下没有任何 verify_summary.json, 请先运行 "
              "scripts/experiments/verify_dlpack.py")
        return 2
    DEV = dev

    # ---- GCN 摘要装载 (E2/E3/E4) + bench ----
    globals()["GCN"] = {"e2": norm_gcn(s["gcn_e2"]), "e3": norm_gcn(s["gcn_e3"]),
                        "e4": norm_gcn(s["gcn_e4"])}
    BENCH[:] = norm_bench(s["bench"])

    # ---- 缺失项与补跑命令提示 ----
    missing = []
    if not (s["gcn_e2"] and s["gcn_e3"]):
        missing.append(("E2/E3 GCN 主对照对",
                        "python scripts/experiments/gcn_hybrid.py --device npu --rtol 5e-3  &  "
                        "python scripts/experiments/gcn_hybrid.py --device npu --exchange copy --rtol 5e-3"))
    if not s["gcn_e4"]:
        missing.append(("E4 GCN 形态对称性 (PT前段+MS后段)",
                        "python scripts/experiments/gcn_hybrid.py --device npu --topology front_pt --rtol 5e-3"))
    if not s["bench"]:
        missing.append(("E5 纯交换吞吐(尺寸扫描)",
                        "python scripts/experiments/gcn_hybrid.py --device npu --bench"))

    # ---- 绘图 (四张证据图) ----
    os.makedirs(CHARTS_DIR, exist_ok=True)
    charts = [fig_gcn(), fig_e2e3(), fig_scorecard(), fig_bench()]
    print("=" * 78)
    print(f"数据来源: results/{dev}/")
    for p in charts:
        print(f"图表已生成: {p}")
    if missing:
        print("-" * 78)
        print("以下数据缺失, 对应图表以\"待复测\"占位:")
        for name, cmd in missing:
            print(f"  [{name}]")
            print(f"    {cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
