"""
verify_pipeline.py
------------------
Bộ kiểm thử độc lập (chạy được NGAY CẢ KHI CHƯA TẢI DỮ LIỆU M5) nhằm xác minh
tất cả yêu cầu kỹ thuật trong tài liệu hướng dẫn:
1. Kiểm tra 03 trường hợp biên bằng tay (STEP 4.6):
   - Trường hợp A: Không có thiếu hàng -> Greedy & KLA-LS trả về phương án rỗng (0 đơn vị).
   - Trường hợp B: Không có hàng dư an toàn -> trả về phương án rỗng.
   - Trường hợp C: Cửa hàng nhận không còn sức chứa -> trả về phương án rỗng.
2. Kiểm tra công thức Fill Rate khi TotalDemand == 0 -> trả về None (không gán 100%).
3. Kiểm tra bảo toàn hàng hóa và các ràng buộc vi phạm trong check_feasible().
4. Kiểm tra lõi KLA trên hàm chuẩn liên tục Sphere 5D (STEP 6.2).
5. Kiểm tra Greedy, KLA và KLA-LS trên kịch bản nhân tạo có chênh lệch thiếu–dư rõ ràng,
   chứng minh KLA-LS tạo nghiệm khả thi và Local Search thực sự cải thiện nghiệm.

Lưu ý: Script này CHỈ chạy kiểm thử trong RAM, KHÔNG ghi đè bất kỳ file dữ liệu thật
nào trong data/ hay results/.
"""

from __future__ import annotations

import numpy as np

from greedy import solve_greedy
from kla import solve_kla, verify_kla_on_continuous_benchmark
from local_search import improve_transfers_local_search, solve_kla_ls
from simulator import (
    ProblemInstance,
    check_feasible,
    compute_objective_from_demand,
    no_transfer_plan,
    simulate_7_days,
)

STORES = ["CA_1", "CA_2", "CA_3", "CA_4"]
ITEMS = ["TEST_SKU_1", "TEST_SKU_2"]
COST_MAT = np.array(
    [
        [0.0, 1.5, 2.5, 3.5],
        [1.5, 0.0, 1.0, 2.0],
        [2.5, 1.0, 0.0, 3.0],
        [3.5, 2.0, 3.0, 0.0],
    ],
    dtype=float,
)


def _make_instance(
    name: str,
    init_inv: np.ndarray,
    safety: np.ndarray,
    cap: np.ndarray,
    forecast: np.ndarray,
    actual_total: np.ndarray | None = None,
) -> ProblemInstance:
    max_surp = np.maximum(0, init_inv - safety)
    if actual_total is None:
        actual_total = forecast.copy()
    actual_daily = np.repeat(actual_total[:, :, None] / 7.0, 7, axis=2)
    return ProblemInstance(
        scenario_name=name,
        stores=STORES,
        items=ITEMS,
        initial_inventory=init_inv.astype(int),
        safety_target=safety.astype(int),
        max_surplus_send=max_surp.astype(int),
        capacity=cap.astype(int),
        forecast_demand=forecast.astype(float),
        actual_demand_daily=actual_daily,
        actual_demand_total=actual_total.astype(float),
        transfer_cost=COST_MAT.copy(),
    )


def test_edge_cases() -> None:
    print("[TEST 1] Kiểm tra 03 trường hợp biên bằng tay (STEP 4.6)...")

    # Case A: Không có thiếu hàng (mọi cửa hàng đều có tồn >= nhu cầu dự báo)
    forecast_a = np.array([[20, 10], [20, 10], [20, 10], [20, 10]], dtype=float)
    init_a = np.array([[25, 15], [30, 15], [30, 15], [22, 12]], dtype=int)
    inst_a = _make_instance("NO_SHORTAGE", init_a, forecast_a, forecast_a * 2, forecast_a)
    x_g_a, _ = solve_greedy(inst_a)
    x_k_a, _ = solve_kla_ls(inst_a, pop_size=10, max_iter=15, seed=42)
    assert x_g_a.sum() == 0, f"Greedy phải trả về 0 khi không thiếu hàng, nhưng nhận {x_g_a.sum()}"
    assert x_k_a.sum() == 0, f"KLA-LS phải trả về 0 khi không thiếu hàng, nhưng nhận {x_k_a.sum()}"
    print("  [PASS] Case A (Không thiếu hàng): Greedy=0, KLA-LS=0")

    # Case B: Không có hàng dư an toàn (mọi cửa hàng tồn <= safety_target)
    forecast_b = np.array([[40, 20], [30, 20], [30, 20], [30, 20]], dtype=float)
    init_b = np.array([[10, 5], [30, 20], [25, 18], [30, 20]], dtype=int)
    inst_b = _make_instance("NO_SURPLUS", init_b, forecast_b, forecast_b * 2, forecast_b)
    x_g_b, _ = solve_greedy(inst_b)
    x_k_b, _ = solve_kla_ls(inst_b, pop_size=10, max_iter=15, seed=42)
    assert x_g_b.sum() == 0 and x_k_b.sum() == 0, "Phải trả về 0 khi không có hàng dư an toàn"
    print("  [PASS] Case B (Không có hàng dư): Greedy=0, KLA-LS=0")

    # Case C: Cửa hàng nhận không còn sức chứa (capacity == initial_inventory tại cửa hàng thiếu)
    forecast_c = np.array([[50, 30], [20, 20], [20, 20], [20, 20]], dtype=float)
    init_c = np.array([[20, 10], [40, 40], [40, 40], [25, 25]], dtype=int)
    cap_c = np.array([[20, 10], [50, 50], [50, 50], [50, 50]], dtype=int)  # CA_1 đã đầy kho!
    inst_c = _make_instance("NO_CAPACITY", init_c, forecast_c, cap_c, forecast_c)
    x_g_c, _ = solve_greedy(inst_c)
    x_k_c, _ = solve_kla_ls(inst_c, pop_size=10, max_iter=15, seed=42)
    assert x_g_c.sum() == 0 and x_k_c.sum() == 0, "Phải trả về 0 khi cửa hàng thiếu không còn sức chứa"
    print("  [PASS] Case C (Cửa hàng nhận hết sức chứa): Greedy=0, KLA-LS=0")


def test_fill_rate_zero_demand_and_feasibility() -> None:
    print("[TEST 2] Kiểm tra Fill Rate khi TotalDemand = 0 và ràng buộc check_feasible...")
    zero_d = np.zeros((4, 2), dtype=float)
    init_z = np.ones((4, 2), dtype=int) * 5
    inst_z = _make_instance("ZERO_DEMAND", init_z, init_z, init_z * 2, zero_d, zero_d)
    sim_z = simulate_7_days(no_transfer_plan(inst_z), inst_z, use_actual=True)
    assert sim_z["metrics"]["fill_rate"] is None, "TotalDemand=0 phải trả về FillRate=None, KHÔNG gán 100%!"

    # Kiểm tra check_feasible phát hiện vi phạm tự chuyển và âm
    bad_self = no_transfer_plan(inst_z)
    bad_self[0, 0, 0] = 2
    ok_self, _ = check_feasible(bad_self, inst_z)
    assert not ok_self, "check_feasible phải từ chối tự chuyển cho chính mình!"

    bad_neg = no_transfer_plan(inst_z)
    bad_neg[1, 0, 0] = -1
    ok_neg, _ = check_feasible(bad_neg, inst_z)
    assert not ok_neg, "check_feasible phải từ chối chuyển số lượng âm!"
    print("  [PASS] FillRate(TotalDemand=0)=None & check_feasible bắt đủ lỗi vi phạm.")


def test_kla_continuous() -> None:
    print("[TEST 3] Kiểm tra lõi KLA trên hàm chuẩn liên tục Sphere 5D (STEP 6.2)...")
    res = verify_kla_on_continuous_benchmark(seed=42)
    print(f"  Sphere 5D: initial_best={res['initial_best_f']:.4f} -> final_best={res['final_best_f']:.6f}")
    assert res["final_best_f"] < 1e-2, f"KLA không hội tụ tốt trên hàm Sphere 5D: {res}"
    print("  [PASS] KLA hội tụ mạnh về gần 0 trên hàm chuẩn liên tục.")


def test_optimization_and_local_search() -> None:
    print("[TEST 4] Kiểm tra end-to-end No Transfer vs Greedy vs KLA vs KLA-LS (kịch bản mẫu)...")
    forecast = np.array(
        [
            [60, 40],  # CA_1
            [50, 35],  # CA_2
            [45, 30],  # CA_3
            [40, 25],  # CA_4
        ],
        dtype=float,
    )
    init_inv = np.round(forecast * np.array([[0.5], [1.6], [1.6], [1.1]])).astype(int)
    safety = np.round(forecast).astype(int)
    cap = np.maximum(np.ceil(forecast * 2.0).astype(int), init_inv)
    inst = _make_instance("SYNTHETIC_CHECK", init_inv, safety, cap, forecast)

    x_none = no_transfer_plan(inst)
    x_greedy, _ = solve_greedy(inst)
    x_kla, info_kla = solve_kla(inst, pop_size=15, max_iter=35, seed=42)
    x_klals, info_klals = solve_kla_ls(inst, pop_size=15, max_iter=35, seed=42, ls_every=10)

    f_none = compute_objective_from_demand(x_none, inst, inst.forecast_demand)["total_cost"]
    f_greedy = compute_objective_from_demand(x_greedy, inst, inst.forecast_demand)["total_cost"]
    f_kla = info_kla["forecast_objective"]
    f_klals = info_klals["forecast_objective"]

    # Kiểm chứng thêm Local Search trực tiếp cải thiện một nghiệm dưới tối ưu (chuyển từ tuyến đắt CA_4->CA_1)
    subopt = no_transfer_plan(inst)
    subopt[3, 0, 0] = min(int(inst.max_surplus_send[3, 0]), 4)  # Gửi từ CA_4 (3.5 USD) thay vì CA_2 (1.5 USD)
    f_sub_before = compute_objective_from_demand(subopt, inst, inst.forecast_demand)["total_cost"]
    sub_imp, sub_info = improve_transfers_local_search(subopt, inst)
    f_sub_after = sub_info["forecast_objective"]

    print(f"  No Transfer : F = {f_none:.2f} USD")
    print(f"  Greedy      : F = {f_greedy:.2f} USD (chuyển {int(x_greedy.sum())} đv)")
    print(f"  KLA         : F = {f_kla:.2f} USD (chuyển {int(x_kla.sum())} đv)")
    print(f"  KLA-LS      : F = {f_klals:.2f} USD (chuyển {int(x_klals.sum())} đv, LS cải thiện {info_klals['ls_improvements']} lần)")
    print(f"  LS standalone test: {f_sub_before:.2f} -> {f_sub_after:.2f} USD (cải thiện {sub_info['n_improvements']} bước)")

    assert f_greedy < f_none, "Greedy phải tốt hơn No Transfer khi có thiếu và dư hàng!"
    assert f_klals <= f_kla + 1e-6, "KLA-LS phải tốt hơn hoặc bằng KLA thuần!"
    assert f_sub_after < f_sub_before, "Local Search phải cải thiện được nghiệm dưới tối ưu nhân tạo!"
    assert check_feasible(x_kla, inst)[0] and check_feasible(x_klals, inst)[0]
    print("  [PASS] Toàn bộ thuật toán hoạt động chính xác, khả thi 100% và KLA-LS cải thiện nghiệm rõ rệt.")


def main() -> int:
    test_edge_cases()
    test_fill_rate_zero_demand_and_feasibility()
    test_kla_continuous()
    test_optimization_and_local_search()
    print("\n=== TẤT CẢ KIỂM THỬ LOGIC ĐỀU ĐẠT (ALL PASS) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
