"""
greedy.py
---------
Thuật toán đối chứng Greedy (tham lam) cho bài toán điều chuyển tồn kho 1 kỳ (STEP 4).

Nguyên tắc hoạt động (chỉ sử dụng nhu cầu dự báo forecast_demand, KHÔNG nhìn tập test):
- Xử lý độc lập cho từng sản phẩm k:
  1. Xác định lượng hàng dư an toàn của từng cửa hàng gửi:
       surplus[i] = max_surplus_send[i, k] = max(0, I_before[i, k] - safety_target[i, k])
  2. Xác định lượng hàng thiếu theo nhu cầu dự báo tại cửa hàng nhận:
       deficit[j] = max(0, forecast_demand[j, k] - I_before[j, k])
  3. Sắp xếp tất cả các cặp (i -> j, i != j) theo chi phí điều chuyển đơn vị tăng dần.
  4. Với mỗi cặp (i -> j) có chi phí nhỏ nhất còn lại:
       - Tính số đơn vị khả thi: qty = min(surplus[i], int(deficit[j]), rem_cap[j])
       - Kiểm tra lợi ích biên (chỉ chuyển nếu việc chuyển làm giảm hàm mục tiêu F(X)).
       - Cập nhật lệnh chuyển, lượng dư còn lại, lượng thiếu còn lại, sức chứa còn lại.
  5. Dừng khi không còn lệnh điều chuyển khả thi có lợi.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np

from simulator import (
    ProblemInstance,
    check_feasible,
    compute_objective_from_demand,
)


def solve_greedy(instance: ProblemInstance) -> Tuple[np.ndarray, Dict[str, object]]:
    """Chạy thuật toán Greedy trên nhu cầu dự báo của instance."""
    t0 = time.perf_counter()
    S = instance.n_stores
    K = instance.n_items

    transfers = np.zeros((S, S, K), dtype=int)

    # Sắp xếp trước mọi cặp (i, j) với i != j theo chi phí đơn vị tăng dần
    pairs: List[Tuple[float, int, int]] = []
    for i in range(S):
        for j in range(S):
            if i != j:
                pairs.append((float(instance.transfer_cost[i, j]), i, j))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))

    for k in range(K):
        surplus = instance.max_surplus_send[:, k].astype(int).copy()
        rem_cap = np.maximum(0, instance.capacity[:, k] - instance.initial_inventory[:, k]).astype(int)
        # Lượng thiếu theo dự báo (giữ dạng float để xét chính xác phần lẻ nếu có từ kịch bản +30%/+50%)
        deficit_float = np.maximum(0.0, instance.forecast_demand[:, k] - instance.initial_inventory[:, k])

        # Bước 1: Chuyển phần nguyên lượng thiếu trước theo thứ tự chi phí tăng dần
        for unit_cost, i, j in pairs:
            if surplus[i] <= 0 or rem_cap[j] <= 0 or deficit_float[j] <= 0:
                continue

            # Lợi ích biên cho 1 đơn vị nguyên chuyển từ i (đang dư > safety_target) sang j (đang thiếu):
            # giảm shortage_penalty + giảm leftover_cost tại i vs tốn unit_cost
            int_need = int(math.floor(deficit_float[j]))
            if int_need > 0 and unit_cost < (instance.shortage_penalty + instance.leftover_cost):
                qty = min(int(surplus[i]), int_need, int(rem_cap[j]))
                if qty > 0:
                    transfers[i, j, k] += qty
                    surplus[i] -= qty
                    rem_cap[j] -= qty
                    deficit_float[j] -= qty

            # Xét thêm 1 đơn vị nếu còn phần thiếu lẻ (0 < frac < 1) và lợi ích biên dương
            frac = deficit_float[j]
            if 0.0 < frac < 1.0 and surplus[i] > 0 and rem_cap[j] > 0:
                # Chuyển thêm 1 đơn vị: tiết kiệm frac * shortage_penalty + leftover_cost (tại i)
                # nhưng phát sinh (1 - frac) * leftover_cost (tại j) + unit_cost
                marginal_saving = (
                    frac * instance.shortage_penalty
                    + instance.leftover_cost
                    - (1.0 - frac) * instance.leftover_cost
                )
                if unit_cost < marginal_saving:
                    transfers[i, j, k] += 1
                    surplus[i] -= 1
                    rem_cap[j] -= 1
                    deficit_float[j] = 0.0

    elapsed = time.perf_counter() - t0
    feasible, reasons = check_feasible(transfers, instance)
    if not feasible:
        raise RuntimeError(f"Greedy tạo nghiệm không khả thi: {reasons}")

    forecast_eval = compute_objective_from_demand(transfers, instance, instance.forecast_demand)
    info = {
        "method": "Greedy",
        "runtime_sec": elapsed,
        "feasible": feasible,
        "forecast_objective": forecast_eval["total_cost"],
    }
    return transfers, info
