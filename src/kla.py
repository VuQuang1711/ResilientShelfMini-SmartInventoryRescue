"""
kla.py
------
Cài đặt thuật toán Kirchhoff's Law Algorithm (KLA) theo bài báo gốc:
  Ghasemi, M., Khodadadi, N., Trojovský, P., Li, L., Mansor, Z., Abualigah, L.,
  Alharbi, A. H., & El-Kenawy, E.-S. M. (2025).
  "Kirchhoff's law algorithm (KLA): a novel physics-inspired non-parametric
  metaheuristic algorithm for optimization problems."
  Artificial Intelligence Review, 58, 325.
  DOI: https://doi.org/10.1007/s10462-025-11289-5

Gồm 3 thành phần rõ ràng (STEP 6):
1. `kla_optimize_continuous`: Lõi KLA liên tục không tham số (chỉ cần pop_size, max_iter)
   với kiểm thử hàm chuẩn liên tục (`verify_kla_on_continuous_benchmark`).
2. `decode_and_repair_sku` / `repair_transfers`: Chuyển vector vị trí liên tục thành
   ma trận điều chuyển nguyên không âm và sửa nghiệm thỏa mãn tuyệt đối:
   - không chuyển âm, không tự chuyển (x[i,i,k]=0), triệt tiêu chuyển hai chiều ngược nhau,
   - không xuất quá lượng dư an toàn max_surplus_send,
   - không nhận vượt quá sức chứa capacity.
3. `solve_kla`: Tối ưu bài toán điều chuyển đa cửa hàng bằng KLA thuần (không Local Search).
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from simulator import (
    ProblemInstance,
    check_feasible,
    compute_objective_from_demand,
)

EPS = 1e-12


def kla_optimize_continuous(
    objective_fn: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    pop_size: int = 15,
    max_iter: int = 35,
    seed: int = 42,
    init_solutions: Optional[List[np.ndarray]] = None,
    post_iter_hook: Optional[
        Callable[[int, np.ndarray, float, np.random.Generator], Tuple[np.ndarray, float]]
    ] = None,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Lõi tối ưu hóa liên tục Kirchhoff's Law Algorithm (KLA — tham khảo Ghasemi et al., 2025).

    Quy trình mỗi vòng lặp cho từng cá thể X_i:
    - Chọn ngẫu nhiên 3 cá thể khác biệt j, a, b != i trong quần thể.
    - Tính biến hướng dòng điện (charge-flow direction):
        q   = (f_i - f_j + eps) / (|f_i - f_j| + eps)
        Q_a = (f_i - f_a + eps) / (|f_i - f_a| + eps)
        Q_b = (f_i - f_b + eps) / (|f_i - f_b| + eps)
    - Tính hệ số tỷ lệ cường độ (phiên bản chuẩn hóa ổn định số học trong (0, 1]
      nhằm tránh bùng nổ bước nhảy khi chênh lệch |f_j| và |f_i| lớn):
        q_1   = ((min(|f_i|, |f_j|) + eps) / (max(|f_i|, |f_j|) + eps))^(2 * r_q)
        Q_a,1 = ((min(|f_i|, |f_a|) + eps) / (max(|f_i|, |f_a|) + eps))^(2 * r_a)
        Q_b,1 = ((min(|f_i|, |f_b|) + eps) / (max(|f_i|, |f_b|) + eps))^(2 * r_b)
    - Thành phần tương tác (dạng dòng Kirchhoff):
        S_1 = (q_1 / q) * (X_j - X_i)
        S_2 = (Q_a,1 / Q_a) * (X_a - X_i)
        S_3 = (Q_b,1 / Q_b) * (X_b - X_i)
    - Cập nhật vị trí (trọng số ngẫu nhiên r1, r2, r3 ~ U(0, 1)^dim):
        X_new = clip(X_i + r1 * S_1 + r2 * S_2 + r3 * S_3, lb, ub)
    - Giữ nghiệm tốt hơn (greedy selection).
    """
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    dim = lb.shape[0]

    if pop_size < 4:
        raise ValueError("KLA yêu cầu pop_size >= 4 để chọn được 3 cá thể khác biệt (j, a, b).")

    # Nếu không gian tìm kiếm suy biến (ub == lb ở mọi chiều), trả về ngay
    if np.all(ub <= lb + EPS):
        x0 = lb.copy()
        f0 = float(objective_fn(x0))
        return x0, f0, [f0] * (max_iter + 1)

    # Khởi tạo quần thể đều trong [lb, ub]
    pop = rng.uniform(lb, ub, size=(pop_size, dim))
    if init_solutions:
        for idx, sol in enumerate(init_solutions[:pop_size]):
            pop[idx] = np.clip(np.asarray(sol, dtype=float), lb, ub)

    fitness = np.array([float(objective_fn(ind)) for ind in pop], dtype=float)
    best_idx = int(np.argmin(fitness))
    best_x = pop[best_idx].copy()
    best_f = float(fitness[best_idx])
    history: List[float] = [best_f]

    for it in range(1, max_iter + 1):
        for i in range(pop_size):
            # Chọn 3 cá thể phân biệt j, a, b != i
            others = [idx for idx in range(pop_size) if idx != i]
            j, a, b = rng.choice(others, size=3, replace=False)

            f_i, f_j, f_a, f_b = fitness[i], fitness[j], fitness[a], fitness[b]

            # Hướng dòng tương tác theo định luật Kirchhoff (chặt chẽ tránh chia 0):
            # q > 0 khi đối tác tốt hơn i (hút về), q < 0 khi đối tác kém hơn i (đẩy ra)
            q = (f_i - f_j + EPS) / (abs(f_i - f_j) + EPS)
            q_a = (f_i - f_a + EPS) / (abs(f_i - f_a) + EPS)
            q_b = (f_i - f_b + EPS) / (abs(f_i - f_b) + EPS)

            # Tỷ lệ cường độ theo độ thích nghi (chuẩn hóa trong (0, 1] khi so sánh cặp)
            r_q, r_a, r_b = rng.random(3)
            q1 = ((min(abs(f_i), abs(f_j)) + EPS) / (max(abs(f_i), abs(f_j)) + EPS)) ** (2.0 * r_q)
            qa1 = ((min(abs(f_i), abs(f_a)) + EPS) / (max(abs(f_i), abs(f_a)) + EPS)) ** (2.0 * r_a)
            qb1 = ((min(abs(f_i), abs(f_b)) + EPS) / (max(abs(f_i), abs(f_b)) + EPS)) ** (2.0 * r_b)

            s1 = (q1 / q) * (pop[j] - pop[i])
            s2 = (qa1 / q_a) * (pop[a] - pop[i])
            s3 = (qb1 / q_b) * (pop[b] - pop[i])

            # Trọng số ngẫu nhiên cho 3 nhánh tương tác Kirchhoff
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            r3 = rng.random(dim)

            x_new = pop[i] + r1 * s1 + r2 * s2 + r3 * s3
            x_new = np.clip(x_new, lb, ub)
            f_new = float(objective_fn(x_new))

            if f_new < fitness[i]:
                pop[i] = x_new
                fitness[i] = f_new
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()

        # Hook tùy chọn (dùng cho KLA-LS khi chèn Local Search lên nghiệm tốt nhất)
        if post_iter_hook is not None:
            cand_x, cand_f = post_iter_hook(it, best_x, best_f, rng)
            if cand_f < best_f:
                best_x = np.clip(cand_x, lb, ub)
                best_f = float(cand_f)
                # Thay cá thể kém nhất trong quần thể bằng nghiệm đã cải thiện
                worst_idx = int(np.argmax(fitness))
                pop[worst_idx] = best_x.copy()
                fitness[worst_idx] = best_f

        history.append(best_f)

    return best_x, best_f, history


def verify_kla_on_continuous_benchmark(seed: int = 42) -> Dict[str, float]:
    """
    Kiểm chứng KLA trên hàm chuẩn liên tục Sphere 5 chiều: f(x) = sum(x_d^2),
    miền [-10, 10]^5, cực tiểu lý thuyết f(0) = 0 (STEP 6.2).
    """
    dim = 5
    lb = np.full(dim, -10.0)
    ub = np.full(dim, 10.0)
    best_x, best_f, history = kla_optimize_continuous(
        objective_fn=lambda x: float(np.sum(x ** 2)),
        lb=lb,
        ub=ub,
        pop_size=20,
        max_iter=80,
        seed=seed,
    )
    return {
        "initial_best_f": float(history[0]),
        "final_best_f": float(best_f),
        "best_norm": float(np.linalg.norm(best_x)),
    }


def _off_diagonal_indices(n_stores: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n_stores) for j in range(n_stores) if i != j]


def repair_sku_matrix(
    raw_mat: np.ndarray,
    initial_inv_k: np.ndarray,
    max_surplus_k: np.ndarray,
    capacity_k: np.ndarray,
    transfer_cost: np.ndarray,
) -> np.ndarray:
    """
    Sửa một ma trận điều chuyển S × S của 1 sản phẩm để đảm bảo khả thi tuyệt đối:
    1. Làm tròn về số nguyên >= 0, đặt đường chéo = 0.
    2. Triệt tiêu chuyển hai chiều ngược nhau giữa cùng một cặp (i, j).
    3. Giới hạn tổng xuất tại mỗi cửa hàng i <= max_surplus_k[i].
    4. Giới hạn tồn sau điều chuyển tại mỗi cửa hàng j <= capacity_k[j].
    """
    S = initial_inv_k.shape[0]
    x = np.maximum(0, np.rint(raw_mat)).astype(int)
    np.fill_diagonal(x, 0)

    # Triệt tiêu dòng đối nghịch (i->j và j->i cùng > 0)
    for i in range(S):
        for j in range(i + 1, S):
            both = min(x[i, j], x[j, i])
            if both > 0:
                x[i, j] -= both
                x[j, i] -= both

    # Ràng buộc xuất không vượt quá hàng dư an toàn
    for i in range(S):
        cap_send = int(max_surplus_k[i])
        out_sum = int(x[i, :].sum())
        if out_sum > cap_send:
            if cap_send <= 0:
                x[i, :] = 0
            else:
                # Thu gọn tỷ lệ rồi cắt phần dư ưu tiên tuyến chi phí cao nhất trước
                scaled = np.floor(x[i, :] * (cap_send / out_sum)).astype(int)
                x[i, :] = scaled
                excess = int(x[i, :].sum()) - cap_send
                if excess > 0:
                    order = np.argsort(-transfer_cost[i, :])
                    for j in order:
                        if j == i or x[i, j] <= 0:
                            continue
                        cut = min(int(x[i, j]), excess)
                        x[i, j] -= cut
                        excess -= cut
                        if excess <= 0:
                            break

    # Ràng buộc sức chứa cửa hàng nhận: initial_inv + inflow - outflow <= capacity
    for j in range(S):
        inflow = int(x[:, j].sum())
        outflow = int(x[j, :].sum())
        inv_after = int(initial_inv_k[j]) + inflow - outflow
        excess = inv_after - int(capacity_k[j])
        if excess > 0:
            # Cắt bớt hàng nhập ưu tiên từ tuyến đắt nhất trước
            order = np.argsort(-transfer_cost[:, j])
            for i in order:
                if i == j or x[i, j] <= 0:
                    continue
                cut = min(int(x[i, j]), excess)
                x[i, j] -= cut
                excess -= cut
                if excess <= 0:
                    break

    return x


def decode_vector_to_sku_matrix(
    vec: np.ndarray,
    n_stores: int,
    pairs: List[Tuple[int, int]],
    initial_inv_k: np.ndarray,
    max_surplus_k: np.ndarray,
    capacity_k: np.ndarray,
    transfer_cost: np.ndarray,
) -> np.ndarray:
    mat = np.zeros((n_stores, n_stores), dtype=float)
    for idx, (i, j) in enumerate(pairs):
        mat[i, j] = vec[idx]
    return repair_sku_matrix(mat, initial_inv_k, max_surplus_k, capacity_k, transfer_cost)


def encode_sku_matrix_to_vector(mat: np.ndarray, pairs: List[Tuple[int, int]]) -> np.ndarray:
    return np.array([float(mat[i, j]) for (i, j) in pairs], dtype=float)


def evaluate_sku_objective(
    x_mat: np.ndarray,
    initial_inv_k: np.ndarray,
    forecast_k: np.ndarray,
    transfer_cost: np.ndarray,
    shortage_penalty: float,
    leftover_cost: float,
) -> float:
    inv_after = initial_inv_k + x_mat.sum(axis=0) - x_mat.sum(axis=1)
    move_cost = float(np.sum(x_mat * transfer_cost))
    shortage = float(np.sum(np.maximum(0.0, forecast_k - inv_after)))
    leftover = float(np.sum(np.maximum(0.0, inv_after - forecast_k)))
    return move_cost + shortage_penalty * shortage + leftover_cost * leftover


def build_sku_bounds(
    instance: ProblemInstance,
    k: int,
    pairs: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Xây dựng cận dưới/cận trên tự nhiên cho từng cung (i -> j) của sản phẩm k."""
    lb = np.zeros(len(pairs), dtype=float)
    ub = np.zeros(len(pairs), dtype=float)
    for idx, (i, j) in enumerate(pairs):
        max_send = int(instance.max_surplus_send[i, k])
        # Sức chứa khả dụng tối đa tại j (kể cả trường hợp j tự xuất hết phần dư)
        max_recv = max(0, int(instance.capacity[j, k] - instance.initial_inventory[j, k]) + int(instance.max_surplus_send[j, k]))
        ub[idx] = float(max(0, min(max_send, max_recv)))
    return lb, ub


def solve_kla(
    instance: ProblemInstance,
    pop_size: int = 15,
    max_iter: int = 35,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Chạy thuật toán KLA thuần (không Local Search) trên nhu cầu dự báo.
    Do các sản phẩm độc lập trong phiên bản MVP, tối ưu từng sản phẩm k theo
    không gian 12 chiều (4 × 3 cặp cửa hàng có hướng) rồi ghép thành ma trận 3D.
    """
    t0 = time.perf_counter()
    S, K = instance.n_stores, instance.n_items
    pairs = _off_diagonal_indices(S)
    transfers = np.zeros((S, S, K), dtype=int)
    total_history = np.zeros(max_iter + 1, dtype=float)

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

        # Gieo 1 cá thể khởi tạo bằng phương án rỗng (x=0) để KLA luôn không tệ hơn No Transfer
        init_pop = [np.zeros(len(pairs), dtype=float)]

        best_vec, _, hist_k = kla_optimize_continuous(
            objective_fn=obj_k,
            lb=lb,
            ub=ub,
            pop_size=pop_size,
            max_iter=max_iter,
            seed=seed * 100 + k,
            init_solutions=init_pop,
        )
        transfers[:, :, k] = decode_vector_to_sku_matrix(
            best_vec, S, pairs, init_k, surp_k, cap_k, instance.transfer_cost
        )
        total_history += np.asarray(hist_k, dtype=float)

    elapsed = time.perf_counter() - t0
    feasible, reasons = check_feasible(transfers, instance)
    if not feasible:
        raise RuntimeError(f"KLA tạo nghiệm không khả thi: {reasons}")

    forecast_eval = compute_objective_from_demand(transfers, instance, instance.forecast_demand)
    info = {
        "method": "KLA",
        "seed": seed,
        "pop_size": pop_size,
        "max_iter": max_iter,
        "runtime_sec": elapsed,
        "feasible": feasible,
        "forecast_objective": forecast_eval["total_cost"],
        "convergence_history": total_history.tolist(),
    }
    return transfers, info
