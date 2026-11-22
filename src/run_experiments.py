"""
run_experiments.py
------------------
Chạy thực nghiệm đầy đủ (STEP 4 + STEP 6 + STEP 7) trên dữ liệu thật đã xử lý:
- 03 kịch bản: NORMAL, FLASH_SALE (+30%), SURGE (+50%)
- 04 phương pháp: No Transfer, Greedy, KLA (5 seeds), KLA-LS (5 seeds)
- Tất cả thuật toán CHỈ tối ưu trên nhu cầu dự báo (demand_forecast.csv),
  sau đó đánh giá công bằng trên 07 ngày kiểm tra thực tế (demand_actual_test.csv).

Đầu ra tự động sinh tại results/:
- baseline_summary.csv   (No Transfer & Greedy)
- kla_summary.csv        (KLA qua các seeds)
- kla_ls_summary.csv     (KLA-LS qua các seeds)
- summary.csv            (Bảng tổng hợp đầy đủ mọi lần chạy theo chuẩn STEP 7.4)
- transfers_greedy.csv   (Lệnh điều chuyển chi tiết của Greedy)
- transfers_kla.csv      (Lệnh điều chuyển chi tiết của KLA - seed tốt nhất)
- transfers_kla_ls.csv   (Lệnh điều chuyển chi tiết của KLA-LS - seed tốt nhất)
- kla_convergence.csv    (Lịch sử hội tụ KLA vs KLA-LS theo vòng lặp)
- experiment_log.md      (Nhật ký chạy đầy đủ cấu hình, seed, thời gian)
- compare_cost.png       (Biểu đồ cột so sánh tổng chi phí theo phương pháp & kịch bản)
- compare_service.png    (Biểu đồ lượng thiếu hàng & Fill Rate theo kịch bản)
- convergence_kla.png    (Biểu đồ hội tụ KLA vs KLA-LS)
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from greedy import solve_greedy
from kla import solve_kla
from local_search import solve_kla_ls
from simulator import (
    ProblemInstance,
    compute_objective_from_demand,
    load_problem_instance,
    no_transfer_plan,
    simulate_7_days,
    transfers_to_dataframe,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

SCENARIOS = ["NORMAL", "FLASH_SALE", "SURGE"]
SEEDS = [42, 43, 44, 45, 46]
KLA_POP_SIZE = 15
KLA_MAX_ITER = 35
KLA_LS_EVERY = 10
METHODS_ORDER = ["No Transfer", "Greedy", "KLA", "KLA-LS"]


def _build_record(
    instance: ProblemInstance,
    method: str,
    seed: int,
    transfers: np.ndarray,
    runtime_sec: float,
) -> Dict[str, object]:
    """Đánh giá phương án trên cả nhu cầu dự báo (fitness) và 7 ngày kiểm tra thực tế."""
    sim_actual = simulate_7_days(transfers, instance, use_actual=True)
    eval_forecast = compute_objective_from_demand(transfers, instance, instance.forecast_demand)
    m = sim_actual["metrics"]

    return {
        "scenario": instance.scenario_name,
        "method": method,
        "seed": seed,
        "total_cost": round(float(m["total_cost"]), 4),
        "transfer_cost": round(float(m["transfer_cost"]), 4),
        "shortage_cost": round(float(m["shortage_cost"]), 4),
        "holding_cost": round(float(m["holding_cost"]), 4),
        "unmet_demand": round(float(m["unmet_demand"]), 4),
        "leftover_inventory": round(float(m["leftover_inventory"]), 4),
        "fill_rate": None if m["fill_rate"] is None else round(float(m["fill_rate"]), 4),
        "total_units_moved": int(m["total_units_moved"]),
        "forecast_objective": round(float(eval_forecast["total_cost"]), 4),
        "runtime_sec": round(float(runtime_sec), 6),
        "feasible": bool(sim_actual["feasible"]),
    }


def plot_compare_cost(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Vẽ biểu đồ cột so sánh tổng chi phí trung bình theo kịch bản và phương pháp."""
    agg = (
        summary_df.groupby(["scenario", "method"], as_index=False)["total_cost"]
        .mean()
        .pivot(index="scenario", columns="method", values="total_cost")
    )
    scen_order = [s for s in SCENARIOS if s in agg.index]
    meth_order = [m for m in METHODS_ORDER if m in agg.columns]
    agg = agg.loc[scen_order, meth_order]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    agg.plot(kind="bar", ax=ax, width=0.72)
    ax.set_title("So sánh tổng chi phí (USD) trên tập kiểm tra 7 ngày theo kịch bản", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Kịch bản nhu cầu")
    ax.set_ylabel("Tổng chi phí trung bình (USD) — càng thấp càng tốt")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8, padding=2)
    ax.legend(title="Phương pháp")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_compare_service(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Vẽ biểu đồ song song: Lượng thiếu hàng (đơn vị) và Fill Rate (%) theo kịch bản."""
    agg_unmet = (
        summary_df.groupby(["scenario", "method"], as_index=False)["unmet_demand"]
        .mean()
        .pivot(index="scenario", columns="method", values="unmet_demand")
    )
    agg_fr = (
        summary_df.groupby(["scenario", "method"], as_index=False)["fill_rate"]
        .mean()
        .pivot(index="scenario", columns="method", values="fill_rate")
    )
    scen_order = [s for s in SCENARIOS if s in agg_unmet.index]
    meth_order = [m for m in METHODS_ORDER if m in agg_unmet.columns]
    agg_unmet = agg_unmet.loc[scen_order, meth_order]
    agg_fr = agg_fr.loc[scen_order, meth_order]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    agg_unmet.plot(kind="bar", ax=axes[0], width=0.76)
    axes[0].set_title("Tổng lượng thiếu hàng (đơn vị) — càng thấp càng tốt", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Kịch bản")
    axes[0].set_ylabel("Lượng thiếu hàng trung bình (đơn vị)")
    axes[0].set_ylim(0, agg_unmet.to_numpy().max() * 1.15)
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].grid(axis="y", alpha=0.3)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.1f", fontsize=7.2, padding=2)
    axes[0].legend(title="Phương pháp", fontsize=8, loc="upper left")

    agg_fr.plot(kind="bar", ax=axes[1], width=0.76)
    axes[1].set_title("Tỷ lệ đáp ứng nhu cầu Fill Rate (%) — càng cao càng tốt", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Kịch bản")
    axes[1].set_ylabel("Fill Rate trung bình (%)")
    axes[1].set_ylim(0, 128)
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].grid(axis="y", alpha=0.3)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f%%", fontsize=6.8, padding=3, rotation=25)
    axes[1].legend(title="Phương pháp", fontsize=8, loc="upper center", ncol=4, framealpha=0.9)

    fig.suptitle("Đánh giá mức độ phục vụ trên tập kiểm tra 7 ngày thực tế", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_convergence(conv_df: pd.DataFrame, out_path: Path) -> None:
    """Vẽ đường hội tụ hàm mục tiêu dự báo F(X) của KLA và KLA-LS qua các vòng lặp."""
    if conv_df.empty:
        return
    scenarios_present = [s for s in SCENARIOS if s in conv_df["scenario"].unique()]
    fig, axes = plt.subplots(1, len(scenarios_present), figsize=(5 * len(scenarios_present), 4.2), sharey=False)
    if len(scenarios_present) == 1:
        axes = [axes]

    for ax, scen in zip(axes, scenarios_present):
        sub = conv_df[conv_df["scenario"] == scen]
        for method, color, style in [("KLA", "#1f77b4", "--"), ("KLA-LS", "#d62728", "-")]:
            msub = sub[sub["method"] == method]
            if msub.empty:
                continue
            mean_curve = msub.groupby("iteration")["forecast_objective"].mean()
            ax.plot(mean_curve.index, mean_curve.values, label=method, color=color, linestyle=style, linewidth=2.0)
        ax.set_title(f"Kịch bản: {scen}", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Vòng lặp (Iteration)")
        ax.set_ylabel("Hàm mục tiêu dự báo F(X) (USD)")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Quá trình hội tụ trung bình (5 seeds): KLA vs KLA–LS", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    required_files = [
        DATA_DIR / "inventory_scenario.csv",
        DATA_DIR / "transfer_cost.csv",
        DATA_DIR / "scenarios.csv",
        DATA_DIR / "demand_actual_test.csv",
    ]
    for rf in required_files:
        if not rf.exists() or rf.stat().st_size == 0:
            raise FileNotFoundError(
                f"Thiếu file dữ liệu đầu vào: {rf}\n"
                "Hãy chạy 'py src/prepare_data.py' sau khi đã đặt sales_train_validation.csv vào data/raw/."
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_records: List[Dict[str, object]] = []
    greedy_transfer_dfs: List[pd.DataFrame] = []
    kla_transfer_dfs: List[pd.DataFrame] = []
    kla_ls_transfer_dfs: List[pd.DataFrame] = []
    conv_rows: List[Dict[str, object]] = []
    log_lines: List[str] = [
        "# Nhật ký Chạy Thực nghiệm — Team Chuyển Kho (HUIT DSC 2026)",
        "",
        f"- **Thời gian chạy:** `{started_at}`",
        f"- **Kịch bản:** `{SCENARIOS}`",
        f"- **Cấu hình KLA:** `pop_size={KLA_POP_SIZE}, max_iter={KLA_MAX_ITER}, seeds={SEEDS}`",
        f"- **Cấu hình KLA-LS:** `pop_size={KLA_POP_SIZE}, max_iter={KLA_MAX_ITER}, ls_every={KLA_LS_EVERY}, seeds={SEEDS}`",
        "",
        "---",
    ]

    for scen in SCENARIOS:
        print(f"\n[RUN] Đang chạy kịch bản: {scen} ...")
        instance = load_problem_instance(DATA_DIR, scen)

        # 1. No Transfer
        t0 = time.perf_counter()
        x_none = no_transfer_plan(instance)
        rt_none = time.perf_counter() - t0
        rec_none = _build_record(instance, "No Transfer", 0, x_none, rt_none)
        all_records.append(rec_none)
        print(f"  - No Transfer : cost={rec_none['total_cost']:.2f} USD | unmet={rec_none['unmet_demand']:.2f} | FR={rec_none['fill_rate']:.2f}%")

        # 2. Greedy
        x_greedy, info_greedy = solve_greedy(instance)
        rec_greedy = _build_record(instance, "Greedy", 0, x_greedy, float(info_greedy["runtime_sec"]))
        all_records.append(rec_greedy)
        greedy_transfer_dfs.append(transfers_to_dataframe(x_greedy, instance, "Greedy"))
        print(f"  - Greedy      : cost={rec_greedy['total_cost']:.2f} USD | unmet={rec_greedy['unmet_demand']:.2f} | FR={rec_greedy['fill_rate']:.2f}%")

        # 3. KLA (5 seeds)
        best_kla_f = float("inf")
        best_kla_x = None
        for seed in SEEDS:
            x_kla, info_kla = solve_kla(instance, pop_size=KLA_POP_SIZE, max_iter=KLA_MAX_ITER, seed=seed)
            rec_kla = _build_record(instance, "KLA", seed, x_kla, float(info_kla["runtime_sec"]))
            all_records.append(rec_kla)
            if float(info_kla["forecast_objective"]) < best_kla_f:
                best_kla_f = float(info_kla["forecast_objective"])
                best_kla_x = x_kla.copy()
            for it_idx, f_val in enumerate(info_kla["convergence_history"]):
                conv_rows.append(
                    {"scenario": scen, "method": "KLA", "seed": seed, "iteration": it_idx, "forecast_objective": round(float(f_val), 4)}
                )
        if best_kla_x is not None:
            kla_transfer_dfs.append(transfers_to_dataframe(best_kla_x, instance, "KLA"))

        # 4. KLA-LS (5 seeds)
        best_klals_f = float("inf")
        best_klals_x = None
        for seed in SEEDS:
            x_klals, info_klals = solve_kla_ls(
                instance,
                pop_size=KLA_POP_SIZE,
                max_iter=KLA_MAX_ITER,
                seed=seed,
                ls_every=KLA_LS_EVERY,
            )
            rec_klals = _build_record(instance, "KLA-LS", seed, x_klals, float(info_klals["runtime_sec"]))
            all_records.append(rec_klals)
            if float(info_klals["forecast_objective"]) < best_klals_f:
                best_klals_f = float(info_klals["forecast_objective"])
                best_klals_x = x_klals.copy()
            for it_idx, f_val in enumerate(info_klals["convergence_history"]):
                conv_rows.append(
                    {"scenario": scen, "method": "KLA-LS", "seed": seed, "iteration": it_idx, "forecast_objective": round(float(f_val), 4)}
                )
        if best_klals_x is not None:
            kla_ls_transfer_dfs.append(transfers_to_dataframe(best_klals_x, instance, "KLA-LS"))

        scen_df = pd.DataFrame([r for r in all_records if r["scenario"] == scen])
        kla_mean = scen_df[scen_df["method"] == "KLA"]["total_cost"].mean()
        klals_mean = scen_df[scen_df["method"] == "KLA-LS"]["total_cost"].mean()
        print(f"  - KLA (5s)    : mean_cost={kla_mean:.2f} USD")
        print(f"  - KLA-LS (5s) : mean_cost={klals_mean:.2f} USD")

    # Lưu tất cả các file CSV kết quả
    summary_df = pd.DataFrame(all_records)
    summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    baseline_df = summary_df[summary_df["method"].isin(["No Transfer", "Greedy"])]
    baseline_df.to_csv(RESULTS_DIR / "baseline_summary.csv", index=False)

    summary_df[summary_df["method"] == "KLA"].to_csv(RESULTS_DIR / "kla_summary.csv", index=False)
    summary_df[summary_df["method"] == "KLA-LS"].to_csv(RESULTS_DIR / "kla_ls_summary.csv", index=False)

    pd.concat(greedy_transfer_dfs, ignore_index=True).to_csv(RESULTS_DIR / "transfers_greedy.csv", index=False)
    pd.concat(kla_transfer_dfs, ignore_index=True).to_csv(RESULTS_DIR / "transfers_kla.csv", index=False)
    pd.concat(kla_ls_transfer_dfs, ignore_index=True).to_csv(RESULTS_DIR / "transfers_kla_ls.csv", index=False)

    conv_df = pd.DataFrame(conv_rows)
    conv_df.to_csv(RESULTS_DIR / "kla_convergence.csv", index=False)

    # Vẽ biểu đồ so sánh và hội tụ
    plot_compare_cost(summary_df, RESULTS_DIR / "compare_cost.png")
    plot_compare_service(summary_df, RESULTS_DIR / "compare_service.png")
    plot_convergence(conv_df, RESULTS_DIR / "convergence_kla.png")

    # Tổng hợp bảng trung bình & độ lệch chuẩn vào log
    agg_table = (
        summary_df.groupby(["scenario", "method"], sort=False)
        .agg(
            mean_total_cost=("total_cost", "mean"),
            std_total_cost=("total_cost", "std"),
            mean_unmet=("unmet_demand", "mean"),
            mean_fill_rate=("fill_rate", "mean"),
            mean_forecast_obj=("forecast_objective", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
            all_feasible=("feasible", "all"),
        )
        .reset_index()
    )
    log_lines.append("\n## Bảng tổng hợp kết quả trung bình trên tập test 7 ngày\n")
    log_lines.append("```text")
    log_lines.append(agg_table.to_string(index=False))
    log_lines.append("```")
    log_lines.append("\n## Chi tiết tất cả các lần chạy\n")
    log_lines.append("```text")
    log_lines.append(summary_df.to_string(index=False))
    log_lines.append("```")
    (RESULTS_DIR / "experiment_log.md").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print("\n[DONE] Đã hoàn tất thực nghiệm và lưu toàn bộ CSV, log, biểu đồ vào thư mục results/.")
    print("[NEXT] Khởi chạy demo Streamlit: py -m streamlit run app/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
