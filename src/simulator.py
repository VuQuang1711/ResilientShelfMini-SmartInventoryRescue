"""
simulator.py
------------
Bộ mô phỏng tồn kho 7 ngày và kiểm tra tính khả thi cho bài toán điều chuyển
đa cửa hàng (STEP 4).

Công thức thống nhất cho mọi thuật toán (No Transfer, Greedy, KLA, KLA–LS):
  - Biến quyết định: x[i, j, k] nguyên >= 0 (đơn vị SKU k chuyển từ i -> j đầu kỳ).
  - Tồn đầu kỳ sau điều chuyển:
        I_after[i, k] = I_before[i, k] + sum_j x[j, i, k] - sum_j x[i, j, k]
  - Hàm mục tiêu (USD):
        F(X) = C_move(X) + 10.0 * Shortage(X) + 0.5 * Leftover(X)
  - Tỷ lệ đáp ứng nhu cầu:
        FillRate = 100 * (TotalDemand - UnmetDemand) / TotalDemand (%)
        Nếu TotalDemand == 0 -> trả về None (không tự động gán 100%).

Giả định MVP: Toàn bộ hàng chuyển đi được thực hiện ngay trước ngày đầu tiên
của chu kỳ 7 ngày (lead time = 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SHORTAGE_PENALTY_USD = 10.0
LEFTOVER_COST_USD = 0.5


@dataclass
class ProblemInstance:
    """Biểu diễn đầy đủ một kịch bản bài toán điều chuyển 4 cửa hàng × 3 SKU."""

    scenario_name: str
    stores: List[str]
    items: List[str]
    # Mảng 2D kích thước (S, K): S = số cửa hàng, K = số sản phẩm
    initial_inventory: np.ndarray   # int, shape (S, K)
    safety_target: np.ndarray       # int, shape (S, K)
    max_surplus_send: np.ndarray    # int, shape (S, K)
    capacity: np.ndarray            # int, shape (S, K)
    forecast_demand: np.ndarray     # float, shape (S, K) — dùng khi tối ưu
    actual_demand_daily: Optional[np.ndarray] = None  # float, shape (S, K, 7) — chỉ dùng kiểm tra
    actual_demand_total: Optional[np.ndarray] = None  # float, shape (S, K) — chỉ dùng kiểm tra
    # Ma trận chi phí điều chuyển kích thước (S, S)
    transfer_cost: np.ndarray = None  # float, shape (S, S)
    shortage_penalty: float = SHORTAGE_PENALTY_USD
    leftover_cost: float = LEFTOVER_COST_USD

    @property
    def n_stores(self) -> int:
        return len(self.stores)

    @property
    def n_items(self) -> int:
        return len(self.items)


def load_problem_instance(data_dir: Path, scenario_name: str = "NORMAL") -> ProblemInstance:
    """Nạp dữ liệu kịch bản từ thư mục data/ mà KHÔNG gây rò rỉ dữ liệu test."""
    inv_df = pd.read_csv(data_dir / "inventory_scenario.csv")
    cost_df = pd.read_csv(data_dir / "transfer_cost.csv")
    scen_df = pd.read_csv(data_dir / "scenarios.csv")
    actual_df = pd.read_csv(data_dir / "demand_actual_test.csv")

    stores = list(dict.fromkeys(inv_df["store_id"].tolist()))
    items = list(dict.fromkeys(inv_df["item_id"].tolist()))
    s_idx = {s: i for i, s in enumerate(stores)}
    k_idx = {k: j for j, k in enumerate(items)}
    S, K = len(stores), len(items)

    initial_inventory = np.zeros((S, K), dtype=int)
    safety_target = np.zeros((S, K), dtype=int)
    max_surplus_send = np.zeros((S, K), dtype=int)
    capacity = np.zeros((S, K), dtype=int)
    forecast_demand = np.zeros((S, K), dtype=float)

    for _, row in inv_df.iterrows():
        i, k = s_idx[row["store_id"]], k_idx[row["item_id"]]
        initial_inventory[i, k] = int(row["initial_inventory"])
        safety_target[i, k] = int(row["safety_target"])
        max_surplus_send[i, k] = int(row["max_surplus_send"])
        capacity[i, k] = int(row["capacity"])
        forecast_demand[i, k] = float(row["forecast_7d"])

    day_cols = [c for c in actual_df.columns if c.startswith("d_")]
    actual_demand_daily = np.zeros((S, K, len(day_cols)), dtype=float)
    actual_demand_total = np.zeros((S, K), dtype=float)
    for _, row in actual_df.iterrows():
        i, k = s_idx[row["store_id"]], k_idx[row["item_id"]]
        daily_vals = row[day_cols].to_numpy(dtype=float)
        actual_demand_daily[i, k, :] = daily_vals
        actual_demand_total[i, k] = float(row["actual_7d"])

    # Áp dụng hệ số kịch bản (chỉ lên cặp store-item được chỉ định trong scenarios.csv)
    scen_rows = scen_df[scen_df["scenario"] == scenario_name]
    if scen_rows.empty:
        raise ValueError(f"Không tìm thấy kịch bản '{scenario_name}' trong scenarios.csv")

    for _, srow in scen_rows.iterrows():
        mult = float(srow["multiplier"])
        t_store = str(srow["store_id"])
        t_item = str(srow["item_id"])
        if t_store == "ALL" and t_item == "ALL":
            forecast_demand *= mult
            actual_demand_daily *= mult
            actual_demand_total *= mult
        else:
            if t_store in s_idx and t_item in k_idx:
                i, k = s_idx[t_store], k_idx[t_item]
                forecast_demand[i, k] *= mult
                actual_demand_daily[i, k, :] *= mult
                actual_demand_total[i, k] *= mult

    transfer_cost = np.zeros((S, S), dtype=float)
    for _, row in cost_df.iterrows():
        if row["from_store"] in s_idx and row["to_store"] in s_idx:
            i, j = s_idx[row["from_store"]], s_idx[row["to_store"]]
            transfer_cost[i, j] = float(row["unit_cost_usd"])

    return ProblemInstance(
        scenario_name=scenario_name,
        stores=stores,
        items=items,
        initial_inventory=initial_inventory,
        safety_target=safety_target,
        max_surplus_send=max_surplus_send,
        capacity=capacity,
        forecast_demand=forecast_demand,
        actual_demand_daily=actual_demand_daily,
        actual_demand_total=actual_demand_total,
        transfer_cost=transfer_cost,
    )


def no_transfer_plan(instance: ProblemInstance) -> np.ndarray:
    """Phương án đối chứng No Transfer (không điều chuyển, toàn bộ bằng 0)."""
    return np.zeros((instance.n_stores, instance.n_stores, instance.n_items), dtype=int)


def compute_post_transfer_inventory(
    initial_inventory: np.ndarray,
    transfers: np.ndarray,
) -> np.ndarray:
    """Tính tồn kho đầu kỳ sau điều chuyển: I_after = I_before + inflow - outflow."""
    inflow = transfers.sum(axis=0)   # shape (S, K)
    outflow = transfers.sum(axis=1)  # shape (S, K)
    return initial_inventory + inflow - outflow


def check_feasible(
    transfers: np.ndarray,
    instance: ProblemInstance,
) -> Tuple[bool, List[str]]:
    """
    Kiểm tra tất cả ràng buộc khả thi của phương án điều chuyển X (shape S × S × K):
    1. Không chuyển âm và phải là số nguyên.
    2. Không tự chuyển cho chính mình (x[i, i, k] == 0).
    3. Không xuất vượt quá lượng hàng dư an toàn: sum_j x[i, j, k] <= max_surplus_send[i, k].
    4. Tồn sau điều chuyển không âm và không vượt sức chứa capacity[i, k].
    5. Bảo toàn tổng hàng hóa toàn mạng lưới theo từng sản phẩm.
    """
    reasons: List[str] = []
    expected_shape = (instance.n_stores, instance.n_stores, instance.n_items)
    if transfers.shape != expected_shape:
        return False, [f"Sai kích thước mảng transfers: {transfers.shape} != {expected_shape}"]

    # 1. Nguyên không âm
    if not np.all(np.isfinite(transfers)):
        reasons.append("Tồn tại giá trị NaN hoặc vô cực.")
        return False, reasons
    if np.any(transfers < 0):
        reasons.append("Tồn tại lượng điều chuyển âm.")
    if not np.all(np.equal(np.mod(transfers, 1), 0)):
        reasons.append("Tồn tại lượng điều chuyển không phải số nguyên.")

    x = np.asarray(transfers, dtype=int)

    # 2. Không chuyển cho chính mình
    for i in range(instance.n_stores):
        if np.any(x[i, i, :] != 0):
            reasons.append(f"Cửa hàng {instance.stores[i]} tự chuyển hàng cho chính mình.")

    # 3. Không xuất quá hàng dư an toàn
    outflow = x.sum(axis=1)  # (S, K)
    over_send = outflow > instance.max_surplus_send
    if np.any(over_send):
        idx = np.argwhere(over_send)
        for i, k in idx[:5]:
            reasons.append(
                f"Xuất vượt ngưỡng dư an toàn tại ({instance.stores[i]}, {instance.items[k]}): "
                f"xuất {outflow[i, k]} > giới hạn {instance.max_surplus_send[i, k]}."
            )

    # 4. Sức chứa và tồn không âm
    inv_after = compute_post_transfer_inventory(instance.initial_inventory, x)
    if np.any(inv_after < 0):
        reasons.append("Tồn kho sau điều chuyển bị âm.")
    over_cap = inv_after > instance.capacity
    if np.any(over_cap):
        idx = np.argwhere(over_cap)
        for i, k in idx[:5]:
            reasons.append(
                f"Vượt sức chứa tại ({instance.stores[i]}, {instance.items[k]}): "
                f"tồn sau {inv_after[i, k]} > sức chứa {instance.capacity[i, k]}."
            )

    # 5. Bảo toàn tổng tồn kho từng sản phẩm
    if not np.array_equal(inv_after.sum(axis=0), instance.initial_inventory.sum(axis=0)):
        reasons.append("Vi phạm định luật bảo toàn hàng hóa (tổng tồn trước != tổng tồn sau).")

    return len(reasons) == 0, reasons


def compute_objective_from_demand(
    transfers: np.ndarray,
    instance: ProblemInstance,
    demand_total: np.ndarray,
) -> Dict[str, Optional[float]]:
    """
    Tính nhanh hàm mục tiêu F(X) và các thành phần chi phí trên một ma trận nhu cầu
    7 ngày (S × K). Dùng chung cho cả bước tối ưu (forecast) và đánh giá (test).
    """
    x = np.asarray(transfers, dtype=int)
    inv_after = compute_post_transfer_inventory(instance.initial_inventory, x).astype(float)

    # Chi phí điều chuyển: sum_{i,j,k} c[i,j] * x[i,j,k]
    move_cost = float(np.sum(x.sum(axis=2) * instance.transfer_cost))

    shortage_mat = np.maximum(0.0, demand_total - inv_after)
    leftover_mat = np.maximum(0.0, inv_after - demand_total)

    total_demand = float(np.sum(demand_total))
    unmet_demand = float(np.sum(shortage_mat))
    leftover_inv = float(np.sum(leftover_mat))

    shortage_cost = instance.shortage_penalty * unmet_demand
    holding_cost = instance.leftover_cost * leftover_inv
    total_cost = move_cost + shortage_cost + holding_cost

    if total_demand <= 0.0:
        fill_rate: Optional[float] = None
    else:
        fill_rate = 100.0 * (total_demand - unmet_demand) / total_demand

    return {
        "total_cost": total_cost,
        "transfer_cost": move_cost,
        "shortage_cost": shortage_cost,
        "holding_cost": holding_cost,
        "total_demand": total_demand,
        "unmet_demand": unmet_demand,
        "leftover_inventory": leftover_inv,
        "fill_rate": fill_rate,
        "total_units_moved": int(np.sum(x)),
    }


def simulate_7_days(
    transfers: np.ndarray,
    instance: ProblemInstance,
    use_actual: bool = True,
) -> Dict[str, object]:
    """
    Mô phỏng chi tiết từng ngày trong chu kỳ 7 ngày:
    - Áp dụng điều chuyển đầu kỳ (trước ngày 1).
    - Mỗi ngày t = 1..7: bán min(tồn hiện tại, nhu cầu ngày t), ghi nhận thiếu hàng
      và tồn cuối ngày.
    - Trả về đầy đủ KPI tổng hợp, bảng chi tiết theo (cửa hàng, sản phẩm) và lịch sử ngày.
    """
    feasible, reasons = check_feasible(transfers, instance)
    x = np.asarray(transfers, dtype=int)
    inv_after = compute_post_transfer_inventory(instance.initial_inventory, x).astype(float)

    if use_actual and instance.actual_demand_daily is not None:
        daily_demand = instance.actual_demand_daily.astype(float)
    else:
        # Nếu dùng nhu cầu dự báo 7 ngày, phân bổ đều theo 7 ngày
        daily_demand = np.repeat(instance.forecast_demand[:, :, None] / 7.0, 7, axis=2)

    S, K, T = daily_demand.shape
    inv_curr = inv_after.copy()
    sold_daily = np.zeros((S, K, T), dtype=float)
    unmet_daily = np.zeros((S, K, T), dtype=float)
    end_inv_daily = np.zeros((S, K, T), dtype=float)

    for t in range(T):
        d_t = daily_demand[:, :, t]
        sold_t = np.minimum(inv_curr, d_t)
        unmet_t = d_t - sold_t
        inv_curr = inv_curr - sold_t

        sold_daily[:, :, t] = sold_t
        unmet_daily[:, :, t] = unmet_t
        end_inv_daily[:, :, t] = inv_curr

    demand_total = daily_demand.sum(axis=2)
    metrics = compute_objective_from_demand(x, instance, demand_total)

    # Kiểm chứng nhất quán giữa mô phỏng từng ngày và công thức tổng hợp 7 ngày
    assert np.isclose(unmet_daily.sum(), metrics["unmet_demand"], atol=1e-6)
    assert np.isclose(inv_curr.sum(), metrics["leftover_inventory"], atol=1e-6)

    # Bảng chi tiết theo từng cửa hàng × sản phẩm
    inflow = x.sum(axis=0)
    outflow = x.sum(axis=1)
    rows = []
    for i, store in enumerate(instance.stores):
        for k, item in enumerate(instance.items):
            d_val = float(demand_total[i, k])
            u_val = float(unmet_daily[i, k, :].sum())
            fr_ik = None if d_val <= 0 else 100.0 * (d_val - u_val) / d_val
            rows.append(
                {
                    "store_id": store,
                    "item_id": item,
                    "initial_inventory": int(instance.initial_inventory[i, k]),
                    "safety_target": int(instance.safety_target[i, k]),
                    "capacity": int(instance.capacity[i, k]),
                    "inflow": int(inflow[i, k]),
                    "outflow": int(outflow[i, k]),
                    "inventory_after": int(inv_after[i, k]),
                    "forecast_7d": round(float(instance.forecast_demand[i, k]), 2),
                    "demand_evaluated": round(d_val, 2),
                    "sold_7d": round(float(sold_daily[i, k, :].sum()), 2),
                    "unmet_demand": round(u_val, 2),
                    "leftover_inventory": round(float(inv_curr[i, k]), 2),
                    "fill_rate_pct": None if fr_ik is None else round(fr_ik, 2),
                }
            )

    return {
        "feasible": feasible,
        "infeasible_reasons": reasons,
        "metrics": metrics,
        "store_item_table": pd.DataFrame(rows),
        "daily_end_inventory": end_inv_daily,
        "daily_unmet": unmet_daily,
    }


def transfers_to_dataframe(
    transfers: np.ndarray,
    instance: ProblemInstance,
    method_name: str = "",
) -> pd.DataFrame:
    """Chuyển mảng 3D transfers (S × S × K) thành bảng danh sách các lệnh > 0."""
    rows = []
    x = np.asarray(transfers, dtype=int)
    for k, item in enumerate(instance.items):
        for i, s_from in enumerate(instance.stores):
            for j, s_to in enumerate(instance.stores):
                qty = int(x[i, j, k])
                if qty > 0:
                    unit_cost = float(instance.transfer_cost[i, j])
                    rows.append(
                        {
                            "scenario": instance.scenario_name,
                            "method": method_name,
                            "item_id": item,
                            "from_store": s_from,
                            "to_store": s_to,
                            "quantity": qty,
                            "unit_cost_usd": unit_cost,
                            "transfer_cost_usd": round(qty * unit_cost, 4),
                        }
                    )
    cols = [
        "scenario",
        "method",
        "item_id",
        "from_store",
        "to_store",
        "quantity",
        "unit_cost_usd",
        "transfer_cost_usd",
    ]
    return pd.DataFrame(rows, columns=cols)
