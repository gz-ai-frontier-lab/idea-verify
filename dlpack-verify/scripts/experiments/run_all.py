# -*- coding: utf-8 -*-
"""
run_all.py — 一键全跑: 顺序执行 E0~E5 全部验证阶段, 输出实时回显并 tee 到
results/{dev}/run_all_<时间戳>.log (日志中每个阶段有明确横幅标注做了什么)。
任一阶段失败立即停止并以非 0 退出; 全部通过后导出时间戳版多 sheet 数据工作簿。

用法 (服务器上已激活 venv 与 CANN 环境, 项目根目录):
    python scripts/run_all.py                          # 全量默认 (npu, 300 步)
    python scripts/run_all.py --device cpu --steps 20  # 本地快速冒烟
"""
import argparse
import os
import re
import subprocess
import sys
import threading
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")   # 管道输出统一 utf-8, 防 GBK 乱码
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIMENTS = os.path.join(PROJECT, "scripts", "experiments")
REPORT = os.path.join(PROJECT, "scripts", "report")
PY = sys.executable


def start_smi_sampler(dev, tag, out_dir, interval=1.0):
    """P1 回退检测采样 (仅 NPU): 后台采集 npu-smi 的 AI Core/HBM 利用率到 CSV。

    npu-smi 各版本参数与 card 编号差异大 (910B 服务器 card id 可能不从 0 起),
    且多卡/容器场景下**首个能解析出数值的卡不一定是训练在用的卡** (v3 曾因此
    锁定空闲卡, 全程采到 AI Core=0), v4 策略:
      1) 跑无参 `npu-smi info` 从表格解析全部 card id (失败回退 [0, 1, 40, 41]);
      2) 每个采样周期对**所有** card 各采一次, 命令按 usages/common/board 顺序
         试到首个能解析出占用率的并记住该卡的可用命令 (逐卡记忆, 不全局锁定);
      3) 解析只认占用率口径: 行含 aicore/ai core/hbm 且排除 capacity/total/
         count 行, 数值 >100 视为容量/内存值丢弃, "已用/容量" 双数值行按比例
         换算百分比; 每卡首个成功输出把原始文本落盘 raw 列 (截断), 供人工核对。
    返回 (stop_event, csv_path); cpu 环境返回 None。"""
    if dev != "npu":
        return None
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"smi_{tag}_{ts}.csv")
    stop = threading.Event()
    state = {"cards": None, "type_of": {}}   # card -> 已探明的 -t 类型
    raw_logged = set()                       # 已落盘过 raw 输出的 card

    def run(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")

    def discover_cards(text):
        ids = re.findall(r"^\|\s*(\d+)\s+\S", text, re.M)
        seen, ordered = set(), []
        for i in ids:
            if i not in seen:
                seen.add(i)
                ordered.append(i)
        return ordered or ["0", "1", "40", "41"]

    def pick_pct(line):
        """行内取占用率: >100 的数字视为容量/内存值丢弃; "已用/容量" 双数值行
        (如 HBM `32768 65536`) 按比例换算为百分比。"""
        nums = [float(x) for x in re.findall(r"\d+\.?\d*", line)]
        if len(nums) == 2 and nums[1] > nums[0] > 0:
            return round(nums[0] / nums[1] * 100, 1)
        ok = [x for x in nums if x <= 100]
        return ok[-1] if ok else None

    def parse(text):
        aicore = hbm = None
        for line in text.splitlines():
            low = line.lower()
            if not any(c.isdigit() for c in line):
                continue
            if "capacity" in low or "total" in low or "count" in low:
                continue
            if aicore is None and ("aicore" in low or "ai core" in low):
                aicore = pick_pct(line)
            elif hbm is None and "hbm" in low:
                hbm = pick_pct(line)
        return aicore, hbm

    def loop():
        with open(path, "w", encoding="utf-8") as f:
            f.write("unix_ts,card,cmd,aicore_pct,hbm_pct,ok,raw\n")
            while not stop.is_set():
                if state["cards"] is None:
                    try:
                        text = run(["npu-smi", "info"])
                        state["cards"] = discover_cards(text)
                        f.write(f"{time.time()},,'npu-smi info',,,,1,"
                                f"\"cards discovered: {','.join(state['cards'])}\"\n")
                    except Exception as e:  # noqa: BLE001
                        state["cards"] = ["0", "1", "40", "41"]
                        f.write(f"{time.time()},,,,0,"
                                f"\"card discovery failed: {type(e).__name__}: {e}\"\n")
                for card in state["cards"]:
                    typ = state["type_of"].get(card)
                    got = ("", "", "", 0, "")
                    for t in ([typ] if typ else ["usages", "common", "board"]):
                        cmd = ["npu-smi", "info", "-t", t, "-i", card]
                        try:
                            out = run(cmd).strip()
                        except Exception as e:  # noqa: BLE001
                            out = f"{type(e).__name__}: {e}"
                        a, h = parse(out)
                        cmd_str = " ".join(cmd).replace('"', "'")
                        if a is not None or h is not None:
                            state["type_of"][card] = t
                            got = (cmd_str, "" if a is None else a,
                                   "" if h is None else h, 1, out)
                            break
                        got = (cmd_str, "", "", 0, out)
                    cmd_str, a, h, ok, out = got
                    raw = ""
                    if card not in raw_logged:
                        raw_logged.add(card)
                        raw = '"' + out.replace('"', "'")[:1000] + '"'
                    f.write(f"{time.time()},{card},'{cmd_str}',{a},{h},{ok},{raw}\n")
                f.flush()
                stop.wait(interval)

    threading.Thread(target=loop, daemon=True).start()
    return stop, path


def build_stages(a):
    """阶段表: (编号, 说明, 命令参数列表)。参数按各脚本的接口透传。"""
    common = ["--device", a.device]
    gcn = (common + ["--epochs", str(a.epochs), "--hidden", str(a.hidden),
                     "--lr", str(a.lr), "--weight-decay", str(a.weight_decay),
                     "--rtol", str(a.rtol), "--seed", str(a.seed)])
    return [
        ("E0", "环境自检 (torch_npu/NPU/CANN/双向 DLPack 冒烟)",
         [os.path.join(EXPERIMENTS, "check_env.py")]),
        ("E1", "机制验证: 11 STEP / 17 检查项 (零拷贝/桥接/解析解/负对照/GNN C1 C3)",
         [os.path.join(EXPERIMENTS, "verify_dlpack.py"), "--device", a.device,
          "--rtol", str(a.rtol)]),
        ("E2", "GCN 混合训练 dlpack + MS 参照臂 (收敛/切换等价性/C2 双锚, 逐步 CSV)",
         [os.path.join(EXPERIMENTS, "gcn_hybrid.py")] + gcn),
        ("E3", "GCN 混合训练 copy 对照 + MS 参照臂 (唯一变量=交换机制)",
         [os.path.join(EXPERIMENTS, "gcn_hybrid.py")] + gcn
         + ["--exchange", "copy"]),
        ("E4*", "GCN 形态对称性: PT前段+MS后段 + MS 参照臂 (迁移过半形态) [NPU 调试项, FAIL 不阻塞]",
         [os.path.join(EXPERIMENTS, "gcn_hybrid.py")] + gcn
         + ["--topology", "front_pt"]),
        ("E5", "GCN 纯交换吞吐尺寸扫描 (边界张量 173KB/1MB/4MB)",
         [os.path.join(EXPERIMENTS, "gcn_hybrid.py")]
         + ["--device", a.device, "--hidden", str(a.hidden), "--bench",
            "--bench-sizes", a.bench_sizes]),
    ]


def main():
    ap = argparse.ArgumentParser(
        description="DLPack 混合过渡态验证: 一键顺序执行 E0~E5 并 tee 日志")
    ap.add_argument("--device", default="npu", choices=["npu", "cpu"])
    ap.add_argument("--epochs", type=int, default=200, help="GCN 训练轮数")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--rtol", default="5e-3", help="C2 双锚容差 (默认 NPU 口径 5e-3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bench-sizes", default="2708,16384,65536")
    ap.add_argument("--bench-n", type=int, default=50)
    a = ap.parse_args()

    dev = "cpu" if a.device == "cpu" else "npu"
    out_dir = os.path.join(PROJECT, "results", dev)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir,
                            f"run_all_{time.strftime('%Y%m%d_%H%M%S')}.log")
    stages = build_stages(a)

    print("=" * 78)
    print(f"DLPack 混合过渡态验证 — 一键全跑 ({len(stages)} 个阶段)")
    print(f"device={a.device}  epochs={a.epochs}  hidden={a.hidden}  lr={a.lr}  "
          f"wd={a.weight_decay}  rtol={a.rtol}")
    print(f"日志: {log_path}")
    print("=" * 78)

    t_all = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        def tee(line):
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()

        for i, (tag, desc, cmd) in enumerate(stages, 1):
            tee("\n" + "#" * 78 + "\n")
            tee(f"### [{i}/{len(stages)}] {tag} — {desc}\n")
            tee(f"### cmd: {PY} {cmd[0]} {' '.join(cmd[1:])}\n")
            tee("#" * 78 + "\n")
            sampler = start_smi_sampler(a.device, tag.rstrip("*"), out_dir) \
                if tag.rstrip("*") in ("E2", "E3", "E4") else None
            if sampler:
                tee(f"### [P1] npu-smi 回退检测采样已启动: {sampler[1]}\n")
            t0 = time.perf_counter()
            # 子进程继承 PYTHONIOENCODING=utf-8 (本模块顶部已设), 管道按 utf-8 读
            proc = subprocess.Popen([PY] + cmd, cwd=PROJECT,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8",
                                    errors="replace", bufsize=1)
            for line in proc.stdout:
                tee(line)
            rc = proc.wait()
            if sampler:
                sampler[0].set()
                tee(f"### [P1] 采样已停止, 数据: {sampler[1]}\n")
            dt = time.perf_counter() - t0
            tee(f"### [{tag}] 退出码 {rc}  耗时 {dt:.1f} s\n")
            if rc != 0:
                if tag.startswith("E4*"):
                    tee(f"### [WARN] {tag} 判定未过 (NPU 调试项), 继续执行后续阶段。\n")
                else:
                    tee("\n" + "!" * 78 + "\n")
                    tee(f"!!! 一键全跑在阶段 {tag} 失败 (退出码 {rc}), 后续阶段未执行。\n")
                    tee(f"!!! 日志: {log_path}\n")
                    return rc if rc else 1

        tee("\n" + "#" * 78 + "\n")
        tee(f"### 全部 {len(stages)} 个阶段通过, 总耗时 "
            f"{(time.perf_counter() - t_all) / 60:.1f} 分钟\n")
        tee("#" * 78 + "\n")
        try:
            sys.path.insert(0, REPORT)
            from export_xlsx import export_xlsx
            xlsx = export_xlsx(dev)
            tee(f"### 数据结果工作簿: {xlsx}\n")
        except ImportError:
            tee("### [提示] 未安装 openpyxl, 跳过 Excel 工作簿导出 "
                "(pip install openpyxl)\n")
        tee(f"### 完整日志: {log_path}\n")
        tee("### 下一步: 把 results/ 目录拷回 Windows, 运行 "
            "python scripts/report/make_report.py 生成最终验证报告\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
