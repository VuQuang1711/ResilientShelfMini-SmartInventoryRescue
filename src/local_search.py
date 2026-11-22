"""
local_search.py
---------------
Thuật toán Tìm kiếm cục bộ (Local Search) và thuật toán lai KLA–LS (STEP 6).

Các phép toán lân cận được cài đặt cho từng sản phẩm k (chỉ chấp nhận nghiệm
khả thi có hàm mục tiêu dự báo F(X) tốt hơn chặt):
1. Điều chỉnh số lượng (±1, ±2, ±5 đơn vị) trên một tuyến điều chuyển (i -> j).
2. Đổi cửa hàng gửi (Swap Sender): chuyển 1, 2, 5 đơn vị hoặc toàn bộ lượng từ
   tuyến (i -> j) sang tuyến (i' -> j) nếu cửa hàng i' còn dư an toàn và tiết kiệm chi phí.
3. Đổi cửa hàng nhận (Swap Receiver): chuyển hướng 1, 2, 5 đơn vị từ (i -> j)
   sang (i -> j') nếu cửa hàng j' đang thiếu nhiều hơn hoặc có lợi hơn.

`solve_kla_ls` kết hợp KLA với Local Search:
- Gọi Local Search tinh chỉnh nghiệm tốt nhất định kỳ sau mỗi `ls_every` vòng KLA
  và tại cuối vòng chạy, đưa nghiệm cải thiện ngược lại vào quần thể (memetic KLA–LS).
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from kla import (
    _off_diagonal_indices,
    build_sku_bounds,
    decode_vector_to_sku_matrix,
    encode_sku_matrix_to_vector,
    evaluate_sku_objective,
    kla_optimize_continuous,
)
from simulator import (
    ProblemInstance,
    check_feasible,
    compute_objective_from_demand,
)


def _is_sku_matrix_feasible(
    x_mat: np.ndarray,
    initial_inv_k: np.ndarray,
    max_surplus_k: np.ndarray,
    capacity_k: np.ndarray,
) -> bool:
    if np.any(x_mat < 0):
        return False
    if np.any(np.diag(x_mat) != 0):
        return False
    outflow = x_mat.sum(axis=1)
    if np.any(outflow > max_surplus_k):
        return False
    inv_after = initial_inv_k + x_mat.sum(axis=0) - outflow
    if np.any(inv_after < 0) or np.any(inv_after > capacity_k):
        return False
    return True


def local_search_sku(
    start_mat: np.ndarray,
    initial_inv_k: np.ndarray,
    max_surplus_k: np.ndarray,
    capacity_k: np.ndarray,
    forecast_k: np.ndarray,
    transfer_cost: np.ndarray,
    shortage_penalty: float,
    leftover_cost: float,
    max_rounds: int = 30,
    steps: Tuple[int, ...] = (5, 2, 1),
) -> Tuple[np.ndarray, float, int]:
    """
    Tìm kiếm cục bộ leo đồi (best/first-improvement) trên ma trận chuyển S × S của 1 SKU.
    Trả về: (best_mat, best_obj, n_improvements).
    """
    S = initial_inv_k.shape[0]
    best_mat = np.asarray(start_mat, dtype=int).copy()
    best_obj = evaluate_sku_objective(
        best_mat, initial_inv_k, forecast_k, transfer_cost, shortage_penalty, leftover_cost
    )
    improvements = 0

    for _ in range(max_rounds):
        improved_this_round = False

        # --- Lân cận 1: Tăng/giảm lượng chuyển trên từng cặp (i -> j) ---
        for i in range(S):
            if max_surplus_k[i] <= 0:
                continue
            for j in range(S):
                if i == j:
                    continue
                for delta in steps:
                    for sign in (-1, 1):
                        new_val = best_mat[i, j] + sign * delta
                        if new_val < 0:
                            continue
                        cand = best_mat.copy()
                        cand[i, j] = new_val
                        if not _is_sku_matrix_feasible(cand, initial_inv_k, max_surplus_k, capacity_k):
                            continue
                        cand_obj = evaluate_sku_objective(
                            cand, initial_inv_k, forecast_k, transfer_cost, shortage_penalty, leftover_cost
                        )
                        if cand_obj < best_obj - 1e-9:
                            best_mat = cand
                            best_obj = cand_obj
                            improvements += 1
                            improved_this_round = True

        # --- Lân cận 2: Đổi nơi gửi (i -> j) sang (i' -> j) ---
        for i in range(S):
            for j in range(S):
                if i == j or best_mat[i, j] <= 0:
                    continue
                for i_alt in range(S):
                    if i_alt == i or i_alt == j or max_surplus_k[i_alt] <= 0:
                        continue
                    cur_qty = int(best_mat[i, j])
                    move_candidates = sorted({min(cur_qty, s) for s in steps} | {cur_qty}, reverse=True)
                    for delta in move_candidates:
                        if delta <= 0 or best_mat[i, j] < delta:
                            continue
                        cand = best_mat.copy()
                        cand[i, j] -= delta
                        cand[i_alt, j] += delta
                        if not _is_sku_matrix_feasible(cand, initial_inv_k, max_surplus_k, capacity_k):
                            continue
                        cand_obj = evaluate_sku_objective(
                            cand, initial_inv_k, forecast_k, transfer_cost, shortage_penalty, leftover_cost
                        )
                        if cand_obj < best_obj - 1e-9:
                            best_mat = cand
                            best_obj = cand_obj
                            improvements += 1
                            improved_this_round = True
                            break

        # --- Lân cận 3: Đổi nơi nhận (i -> j) sang (i -> j') ---
        for i in range(S):
            for j in range(S):
                if i == j or best_mat[i, j] <= 0:
                    continue
                for j_alt in range(S):
                    if j_alt == i or j_alt == j:
                        continue
                    cur_qty = int(best_mat[i, j])
                    move_candidates = sorted({min(cur_qty, s) for s in steps} | {cur_qty}, reverse=True)
                    for delta in move_candidates:
                        if delta <= 0 or best_mat[i, j] < delta:
                            continue
                        cand = best_mat.copy()
                        cand[i, j] -= delta
                        cand[i, j_alt] += delta
                        if not _is_sku_matrix_feasible(cand, initial_inv_k, max_surplus_k, capacity_k):
                            continue
                        cand_obj = evaluate_sku_objective(
                            cand, initial_inv_k, forecast_k, transfer_cost, shortage_penalty, leftover_cost
                        )
                        if cand_obj < best_obj - 1e-9:
                            best_mat = cand
                            best_obj = cand_obj
                            improvements += 1
                            improved_this_round = True
                            break

        if not improved_this_round:
            break

    return best_mat, best_obj, improvements


def improve_transfers_local_search(
    start_transfers: np.ndarray,
    instance: ProblemInstance,
    max_rounds: int = 30,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Áp dụng Local Search lên toàn bộ phương án điều chuyển 3D (S × S × K)."""
    t0 = time.perf_counter()
    S, K = instance.n_stores, instance.n_items
    improved = np.zeros((S, S, K), dtype=int)
    total_improvements = 0

    for k in range(K):
        mat_k, _, n_imp = local_search_sku(
            start_mat=start_transfers[:, :, k],
            initial_inv_k=instance.initial_inventory[:, k],
            max_surplus_k=instance.max_surplus_send[:, k],
            capacity_k=instance.capacity[:, k],
            forecast_k=instance.forecast_demand[:, k],
            transfer_cost=instance.transfer_cost,
            shortage_penalty=instance.shortage_penalty,
            leftover_cost=instance.leftover_cost,
            max_rounds=max_rounds,
        )
        improved[:, :, k] = mat_k
        total_improvements += n_imp

    elapsed = time.perf_counter() - t0
    feasible, reasons = check_feasible(improved, instance)
    if not feasible:
        raise RuntimeError(f"Local Search tạo nghiệm không khả thi: {reasons}")

    eval_res = compute_objective_from_demand(improved, instance, instance.forecast_demand)
    return improved, {
        "runtime_sec": elapsed,
        "feasible": feasible,
        "n_improvements": total_improvements,
        "forecast_objective": eval_res["total_cost"],
    }


def solve_kla_ls(
    instance: ProblemInstance,
    pop_size: int = 15,
    max_iter: int = 35,
    seed: int = 42,
    ls_every: int = 10,
    ls_max_rounds: int = 25,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Thuật toán lai KLA–LS (Kirchhoff's Law Algorithm + Local Search):
    - Chạy KLA tìm kiếm toàn cục với cùng seed/pop_size/max_iter.
    - Sau mỗi `ls_every` vòng lặp và ở vòng cuối cùng, gọi Local Search tinh chỉnh
      nghiệm tốt nhất hiện tại và phản hồi lại vào quần thể KLA.
    """
    t0 = time.perf_counter()
    S, K = instance.n_stores, instance.n_items
    pairs = _off_diagonal_indices(S)
    transfers = np.zeros((S, S, K), dtype=int)
    total_history = np.zeros(max_iter + 1, dtype=float)
    total_ls_improvements = 0

    for k in range(K):
        lb, ub = build_sku_bounds(instance, k, pairs)
        init_k = instance.initial_inventory[:, k]
        surp_k = instance.max_surplus_send[:, k]
        cap_k = instance.capacity[:, k]
        fore_k = instance.forecast_demand[:, k]

        def obj_k(vec: np.ndarray) -> float:
            mat = decode_vector_to_sku_matrix(
                vec, S, pairs, init_k, surp_k, cap_k, instance.transfer_cost
            )
            return evaluate_sku_objective(
                mat, init_k, fore_k, instance.transfer_cost,
                instance.shortage_penalty, instance.leftover_cost,
            )

        ls_counter = [0]

        def ls_hook(it: int, best_vec: np.ndarray, best_f: float, _rng: np.random.Generator) -> Tuple[np.ndarray, float]:
            if (it % ls_every != 0) and (it != max_iter):
                return best_vec, best_f
            cur_mat = decode_vector_to_sku_matrix(
                best_vec, S, pairs, init_k, surp_k, cap_k, instance.transfer_cost
            )
            imp_mat, imp_f, n_imp = local_search_sku(
                start_mat=cur_mat,
                initial_inv_k=init_k,
                max_surplus_k=surp_k,
                capacity_k=cap_k,
                forecast_k=fore_k,
                transfer_cost=instance.transfer_cost,
                shortage_penalty=instance.shortage_penalty,
                leftover_cost=instance.leftover_cost,
                max_rounds=ls_max_rounds,
            )
            ls_counter[0] += n_imp
            if imp_f < best_f:
                return encode_sku_matrix_to_vector(imp_mat, pairs), imp_f
            return best_vec, best_f

        init_pop = [np.zeros(len(pairs), dtype=float)]
        best_vec, _, hist_k = kla_optimize_continuous(
            objective_fn=obj_k,
            lb=lb,
            ub=ub,
            pop_size=pop_size,
            max_iter=max_iter,
            seed=seed * 100 + k,
            init_solutions=init_pop,
            post_iter_hook=ls_hook,
        )
        transfers[:, :, k] = decode_vector_to_sku_matrix(
            best_vec, S, pairs, init_k, surp_k, cap_k, instance.transfer_cost
        )
        total_history += np.asarray(hist_k, dtype=float)
        total_ls_improvements += ls_counter[0]

    elapsed = time.perf_counter() - t0
    feasible, reasons = check_feasible(transfers, instance)
    if not feasible:
        raise RuntimeError(f"KLA-LS tạo nghiệm không khả thi: {reasons}")

    forecast_eval = compute_objective_from_demand(transfers, instance, instance.forecast_demand)
    info = {
        "method": "KLA-LS",
        "seed": seed,
        "pop_size": pop_size,
        "max_iter": max_iter,
        "ls_every": ls_every,
        "ls_improvements": total_ls_improvements,
        "runtime_sec": elapsed,
        "feasible": feasible,
        "forecast_objective": forecast_eval["total_cost"],
        "convergence_history": total_history.tolist(),
    }
    return transfers, info
