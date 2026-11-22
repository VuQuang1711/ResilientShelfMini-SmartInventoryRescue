# Tài liệu Giải thích Mã nguồn Thư mục `src/` bằng Mã giả (Toàn bộ Lớp và Hàm)

- **Cuộc thi:** HUIT Data Science Challenge 2026 — **Team Chuyển Kho**
- **Đề tài:** Tối ưu điều chuyển tồn kho đa cửa hàng dưới nhu cầu bất định bằng thuật toán lai Kirchhoff’s Law Algorithm và Tìm kiếm cục bộ (`KLA–LS`)
- **Phạm vi:** Toàn bộ **07 tệp** trong thư mục `src/`, gồm **01 lớp dữ liệu** (`ProblemInstance` với 2 thuộc tính) và **35 hàm cấp module** (cùng các hàm nội bộ `obj_k`, `ls_hook`) — phủ **42/42 thành phần (100%)**.

---

## Bảng tổng quan 07 tệp mã nguồn trong `src/`

| Chương | Tệp nguồn | Quy mô | Vai trò chính trong pipeline |
| :--- | :--- | :--- | :--- |
| **Chương 1** | `src/prepare_data.py` | 4 hàm | STEP 1–3: Lọc dữ liệu M5 (4 cửa hàng × 3 SKU × 49 ngày), EDA, dự báo 7 ngày và sinh kịch bản. |
| **Chương 2** | `src/simulator.py` | 1 lớp + 7 hàm | STEP 4: Đóng gói kịch bản, kiểm tra ràng buộc khả thi, tính hàm mục tiêu $F(X)$ và mô phỏng 7 ngày. |
| **Chương 3** | `src/greedy.py` | 1 hàm | STEP 4: Thuật toán tham lam đối chứng (`Greedy`) ưu tiên tuyến điều chuyển chi phí nhỏ nhất. |
| **Chương 4** | `src/kla.py` | 9 hàm (`+ obj_k`) | STEP 6: Lõi KLA liên tục, toán tử mã hóa/giải mã/sửa nghiệm khả thi và giải KLA thuần. |
| **Chương 5** | `src/local_search.py` | 4 hàm (`+ hook`) | STEP 6: Tìm kiếm cục bộ 3 lân cận (chỉnh lượng, đổi nơi gửi, đổi nơi nhận) và thuật toán lai `KLA–LS`. |
| **Chương 6** | `src/run_experiments.py` | 5 hàm | STEP 7: Chạy thực nghiệm 3 kịch bản × 4 phương pháp × 5 seeds, xuất CSV, log Markdown và 3 biểu đồ. |
| **Chương 7** | `src/verify_pipeline.py` | 6 hàm | Kiểm thử tự động trong RAM: 3 trường hợp biên, Fill Rate khi cầu = 0, KLA Sphere 5D và `KLA–LS`. |

### Luồng chạy tổng thể của hệ thống (Pipeline)

```text
BƯỚC 1: Chạy verify_pipeline.py   --> Kiểm tra toàn bộ logic thuật toán trên dữ liệu nhân tạo trong RAM.
BƯỚC 2: Chạy prepare_data.py      --> Cắt dữ liệu M5 thật -> EDA -> Dự báo (42 ngày đầu) -> Sinh kịch bản.
BƯỚC 3: Chạy run_experiments.py   --> Gọi simulator + greedy + kla + local_search trên 3 kịch bản (5 seeds).
                                      Xuất CSV, nhật ký experiment_log.md và biểu đồ PNG vào thư mục results/.
BƯỚC 4: Chạy app/app.py           --> Hiển thị kết quả từ results/ hoặc gọi trực tiếp các bộ giải ở chế độ Live.
```

---

## CHƯƠNG 1. `src/prepare_data.py` — Chuẩn bị dữ liệu, EDA & Kịch bản

*Thực hiện trọn vẹn STEP 1, STEP 2 và STEP 3. Hằng số khóa phạm vi: 4 cửa hàng (`CA_1..CA_4`), 3 sản phẩm, 49 ngày cuối (42 lịch sử / 7 test), tỷ lệ tồn đầu kỳ (`50%`, `160%`, `160%`, `110%`), ngưỡng an toàn `100%`, sức chứa `>= 200%`, ma trận chi phí đối xứng `1.0–3.5 USD/đơn vị`.*

### 1.1 `step1_select_m5_data(raw_csv=RAW_CSV)`
- **Mục đích:** Đọc tiết kiệm RAM file doanh số gốc M5, lọc 4 cửa hàng `CA_1..CA_4`, chọn 3 sản phẩm bán chạy xuất hiện đủ ở cả 4 cửa hàng với doanh số dương, và lưu ra `data/m5_selected.csv`.
- **Đầu vào:** `raw_csv`: đường dẫn file `data/raw/sales_train_validation.csv`.
- **Đầu ra:** `(out_df, chosen_items, recent_days)`: bảng 12 dòng × 49 ngày, danh sách 3 mã SKU được chọn, và danh sách 49 tên cột ngày.

```text
HÀM step1_select_m5_data(raw_csv):
    NẾU raw_csv không tồn tại HOẶC dung lượng == 0 THÌ
        BÁO LỖI FileNotFoundError (hướng dẫn tải M5 từ Kaggle)

    // Đọc dòng tiêu đề trước để tiết kiệm RAM
    header_cols ← đọc 0 dòng đầu của raw_csv
    day_cols    ← lọc các cột bắt đầu bằng "d_" trong header_cols
    NẾU số lượng(day_cols) < 49 THÌ BÁO LỖI ValueError

    recent_days ← 49 cột ngày cuối cùng của day_cols
    raw ← đọc raw_csv CHỈ với các cột ["item_id", "store_id"] + recent_days
    raw ← lọc các dòng có store_id thuộc {"CA_1", "CA_2", "CA_3", "CA_4"}

    history_days    ← 42 ngày đầu của recent_days
    last7_hist_days ← 7 ngày cuối của history_days
    raw["last49_sum"]     ← tổng doanh số theo hàng trên recent_days
    raw["last7_hist_sum"] ← tổng doanh số theo hàng trên last7_hist_days

    // Lọc ứng viên: xuất hiện đủ ở 4 cửa hàng và mỗi chuỗi đều bán > 0
    store_counts   ← số cửa hàng khác nhau của mỗi item_id
    min_49_by_item ← giá trị nhỏ nhất của last49_sum theo mỗi item_id
    min_7h_by_item ← giá trị nhỏ nhất của last7_hist_sum theo mỗi item_id

    valid_ids ← các item_id có (store_counts == 4) VÀ (min_49_by_item > 0) VÀ (min_7h_by_item > 0)
    NẾU số lượng(valid_ids) < 3 THÌ
        valid_ids ← nới lỏng chỉ cần (store_counts == 4) VÀ (min_49_by_item > 0)
    NẾU số lượng(valid_ids) < 3 THÌ BÁO LỖI ValueError

    // Chọn Top 3 sản phẩm có tổng doanh số 49 ngày lớn nhất trên 4 cửa hàng
    candidates   ← lọc raw theo valid_ids
    scores       ← tổng last49_sum theo từng item_id
    chosen_items ← 3 item_id đầu tiên khi sắp xếp scores giảm dần

    selected ← lọc candidates theo chosen_items, sắp xếp cố định theo (chosen_items, STORES)
    out_df   ← chọn các cột ["item_id", "store_id"] + recent_days
    GHI out_df ra file "data/m5_selected.csv"
    TRẢ VỀ (out_df, chosen_items, recent_days)
```

### 1.2 `step2_eda_and_forecast(selected, chosen_items, recent_days)`
- **Mục đích:** Kiểm tra chất lượng dữ liệu (không xóa ngày doanh số = 0), tách cứng 42 ngày lịch sử / 7 ngày kiểm tra (chống data leakage), tính dự báo 7 ngày và vẽ 3 biểu đồ EDA.
- **Đầu vào:** `selected` (DataFrame 12 chuỗi), `chosen_items` (3 SKU), `recent_days` (49 cột ngày).
- **Đầu ra:** `(forecast_df, actual_df)`: bảng dự báo 7 ngày và bảng nhu cầu thực 7 ngày test; đồng thời lưu `eda_01..03.png` và `results/eda_report.md`.

```text
HÀM step2_eda_and_forecast(selected, chosen_items, recent_days):
    history_days ← 42 ngày đầu của recent_days
    test_days    ← 7 ngày cuối của recent_days

    // 1. Kiểm tra chất lượng dữ liệu trên toàn bộ 12 × 49 = 588 quan sát
    Đếm n_missing (NaN), n_negative (< 0), n_zeros (== 0) và tỷ lệ zero_pct
    (Giữ nguyên các ngày = 0, không xóa, không sửa)

    // 2. Dự báo CHỈ dùng dữ liệu lịch sử (cấm nhìn test_days)
    last7_hist ← 7 ngày cuối của history_days      // tuần lịch sử ngay trước test
    prev7_hist ← 7 ngày từ vị trí -14 đến -8       // tuần liền trước đó (lag 2w đối chứng)

    forecast_rows ← rỗng; actual_rows ← rỗng
    LẶP VỚI MỖI dòng (item_id, store_id) trong selected:
        daily_mean        ← trung bình(doanh số trên last7_hist)
        forecast_7d       ← 7 × daily_mean
        seasonal_naive_7d ← tổng(doanh số trên prev7_hist)
        THÊM {item_id, store_id, daily_mean, forecast_7d, seasonal_naive_7d} vào forecast_rows

        test_vals ← doanh số từng ngày trong test_days
        THÊM {item_id, store_id, test_vals..., actual_7d = tổng(test_vals)} vào actual_rows

    GHI forecast_df ra "data/demand_forecast.csv"
    GHI actual_df   ra "data/demand_actual_test.csv"

    // 3. Tính MAE tham khảo EDA (không dùng để điều chỉnh tham số)
    mae_ma7    ← trung bình(|forecast_7d - actual_7d|)
    mae_snaive ← trung bình(|seasonal_naive_7d - actual_7d|)

    // 4. Vẽ 3 biểu đồ EDA bằng Matplotlib
    Vẽ eda_01.png: 3 đồ thị đường biến động 49 ngày (có đường đứt ngăn cách History / Test)
    Vẽ eda_02.png: biểu đồ cột tổng doanh số 42 ngày lịch sử theo cửa hàng × sản phẩm
    Vẽ eda_03.png: biểu đồ cột so sánh nhu cầu dự báo 7 ngày giữa 4 cửa hàng

    // 5. Ghi báo cáo kiểm tra dữ liệu & bảng so sánh
    GHI báo cáo Markdown ra "results/eda_report.md"
    TRẢ VỀ (forecast_df, actual_df)
```

### 1.3 `step3_create_scenarios(forecast_df, chosen_items)`
- **Mục đích:** Sinh dữ liệu mô phỏng tồn kho đầu kỳ, ngưỡng an toàn, sức chứa, ma trận chi phí điều chuyển và định nghĩa 3 kịch bản nhu cầu (`NORMAL`, `FLASH_SALE`, `SURGE`).
- **Đầu vào:** `forecast_df` (bảng dự báo 7 ngày), `chosen_items` (danh sách 3 SKU đã chọn).
- **Đầu ra:** `(inv_df, cost_df, scenarios_df)` và lưu tương ứng vào `data/inventory_scenario.csv`, `data/transfer_cost.csv`, `data/scenarios.csv`.

```text
HÀM step3_create_scenarios(forecast_df, chosen_items):
    // 1. Tạo tồn kho đầu kỳ, ngưỡng an toàn và sức chứa CHỈ từ forecast_7d
    inv_rows ← rỗng
    LẶP VỚI MỖI dòng (item_id, store_id, f7) trong forecast_df:
        init_inv      ← làm_tròn_nguyên(INITIAL_INV_RATIO[store_id] × f7)
                        // CA_1: 50%, CA_2: 160%, CA_3: 160%, CA_4: 110%
        safety_target ← làm_tròn_nguyên(1.00 × f7)
        max_surplus   ← max(0, init_inv - safety_target)
        capacity      ← max(làm_tròn_lên(2.00 × f7), init_inv)
        THÊM {item_id, store_id, f7, init_inv, safety_target, max_surplus, capacity} vào inv_rows
    GHI inv_df ra "data/inventory_scenario.csv"

    // 2. Tạo ma trận chi phí điều chuyển đối xứng (4 × 4 = 16 dòng)
    cost_rows ← rỗng
    LẶP s_from TRONG STORES:
        LẶP s_to TRONG STORES:
            NẾU s_from == s_to THÌ c ← 0.0
            NGƯỢC LẠI c ← tra cứu TRANSFER_COST_MATRIX đối xứng (1.0 .. 3.5 USD)
            THÊM {s_from, s_to, c} vào cost_rows
    GHI cost_df ra "data/transfer_cost.csv"

    // 3. Định nghĩa 3 kịch bản nhu cầu (sốc cục bộ tại CA_1 và SKU bán chạy nhất)
    target_store ← "CA_1"
    target_item  ← chosen_items[0]
    scenarios_rows ← [
        (NORMAL,     ALL,         ALL,          1.0),
        (FLASH_SALE, target_item, target_store, 1.3),
        (SURGE,      target_item, target_store, 1.5)
    ]
    GHI scenarios_df ra "data/scenarios.csv"
    TRẢ VỀ (inv_df, cost_df, scenarios_df)
```

### 1.4 `main()`
- **Mục đích:** Hàm điểm vào (entry point) thực thi liên hoàn STEP 1, STEP 2 và STEP 3.
- **Đầu vào:** Không có tham số.
- **Đầu ra:** Mã thoát `0` khi hoàn tất thành công.

```text
HÀM main():
    (selected, chosen_items, recent_days) ← step1_select_m5_data()
    (forecast_df, _)                      ← step2_eda_and_forecast(selected, chosen_items, recent_days)
    step3_create_scenarios(forecast_df, chosen_items)
    IN thông báo hoàn tất STEP 1–3
    TRẢ VỀ 0
```

---

## CHƯƠNG 2. `src/simulator.py` — Bộ mô phỏng 7 ngày & Kiểm tra khả thi

*Thực hiện STEP 4: Định nghĩa bài toán chuẩn hóa dùng chung cho tất cả thuật toán (`No Transfer`, `Greedy`, `KLA`, `KLA–LS`). Giả định MVP: điều chuyển ngay đầu kỳ (`lead time = 0`); hệ số phạt thiếu hàng `= 10.0 USD/đv`, chi phí tồn dư `= 0.5 USD/đv`.*

### 2.1 `class ProblemInstance` (thuộc tính: `n_stores`, `n_items`)
- **Mục đích:** Cấu trúc dữ liệu (`@dataclass`) lưu trữ trọn vẹn một kịch bản bài toán điều chuyển đa cửa hàng.
- **Đầu vào:** Tên kịch bản, danh sách cửa hàng $S$, danh sách SKU $K$, các ma trận $(S \times K)$ tồn đầu kỳ / ngưỡng an toàn / dư tối đa / sức chứa / dự báo / thực tế 7 ngày, ma trận chi phí $(S \times S)$, và hệ số phạt.
- **Đầu ra:** Đối tượng `ProblemInstance`; thuộc tính `n_stores` trả về `len(stores)`, `n_items` trả về `len(items)`.

```text
LỚP ProblemInstance:
    TRƯỜNG DỮ LIỆU:
        scenario_name, stores (S), items (K)
        initial_inventory[S, K], safety_target[S, K], max_surplus_send[S, K], capacity[S, K]
        forecast_demand[S, K]           // Dùng khi tối ưu
        actual_demand_daily[S, K, 7]    // Chỉ dùng kiểm tra
        actual_demand_total[S, K]       // Chỉ dùng kiểm tra
        transfer_cost[S, S]
        shortage_penalty = 10.0, leftover_cost = 0.5

    THUỘC TÍNH n_stores: TRẢ VỀ số lượng(stores)
    THUỘC TÍNH n_items : TRẢ VỀ số lượng(items)
```

### 2.2 `load_problem_instance(data_dir, scenario_name='NORMAL')`
- **Mục đích:** Đọc dữ liệu từ các file CSV trong thư mục `data/` và xây dựng đối tượng `ProblemInstance` cho kịch bản được yêu cầu.
- **Đầu vào:** `data_dir` (`Path` thư mục `data/`), `scenario_name` (`'NORMAL' | 'FLASH_SALE' | 'SURGE'`).
- **Đầu ra:** Đối tượng `ProblemInstance` đã áp dụng hệ số nhân kịch bản.

```text
HÀM load_problem_instance(data_dir, scenario_name):
    ĐỌC inventory_scenario.csv, transfer_cost.csv, scenarios.csv, demand_actual_test.csv
    Xây dựng ánh xạ chỉ số s_idx[store] = 0..S-1 và k_idx[item] = 0..K-1
    Khởi tạo các mảng 0 kích thước (S, K) và (S, K, 7)

    LẶP VỚI MỖI dòng trong inventory_scenario.csv:
        Điền initial_inventory[i, k], safety_target[i, k], max_surplus_send[i, k],
             capacity[i, k], forecast_demand[i, k]

    LẶP VỚI MỖI dòng trong demand_actual_test.csv:
        Điền actual_demand_daily[i, k, 0..6] và actual_demand_total[i, k]

    Lọc các dòng ứng với scenario_name trong scenarios.csv (nếu rỗng -> BÁO LỖI ValueError)
    LẶP VỚI MỖI quy tắc (store_id, item_id, mult) của kịch bản:
        NẾU store_id == "ALL" VÀ item_id == "ALL" THÌ
            Nhân toàn bộ forecast_demand, actual_demand_daily, actual_demand_total với mult
        NGƯỢC LẠI
            Chỉ nhân đúng phần tử [i, k] tương ứng với mult

    Điền ma trận chi phí transfer_cost[i, j] từ transfer_cost.csv
    TRẢ VỀ đối tượng ProblemInstance
```

### 2.3 `no_transfer_plan(instance)`
- **Mục đích:** Sinh phương án đối chứng `No Transfer` (không điều chuyển bất kỳ đơn vị hàng nào).
- **Đầu vào:** `instance` (`ProblemInstance`).
- **Đầu ra:** Mảng số nguyên 3D toàn `0` kích thước `(S, S, K)`.

```text
HÀM no_transfer_plan(instance):
    TRẢ VỀ mảng số nguyên 0 có kích thước (instance.n_stores, instance.n_stores, instance.n_items)
```

### 2.4 `compute_post_transfer_inventory(initial_inventory, transfers)`
- **Mục đích:** Tính lượng tồn kho đầu kỳ tại mỗi cửa hàng–sản phẩm ngay sau khi thực hiện các lệnh điều chuyển.
- **Đầu vào:** `initial_inventory[S, K]` và `transfers[S, S, K]` (với `transfers[i, j, k]` là lượng SKU $k$ chuyển từ $i$ sang $j$).
- **Đầu ra:** Mảng `inv_after[S, K]` = tồn trước + tổng nhận − tổng gửi.

```text
HÀM compute_post_transfer_inventory(initial_inventory, transfers):
    inflow[j, k]  ← TỔNG theo chiều gửi i của transfers[i, j, k]    // axis=0
    outflow[i, k] ← TỔNG theo chiều nhận j của transfers[i, j, k]   // axis=1
    TRẢ VỀ initial_inventory + inflow - outflow
```

### 2.5 `check_feasible(transfers, instance)`
- **Mục đích:** Kiểm tra nghiêm ngặt toàn bộ 5 nhóm ràng buộc khả thi của một phương án điều chuyển $X$.
- **Đầu vào:** `transfers[S, S, K]` và `instance`.
- **Đầu ra:** `(feasible: bool, reasons: List[str])` — `True` nếu thỏa mãn mọi ràng buộc, ngược lại `False` kèm danh sách lỗi.

```text
HÀM check_feasible(transfers, instance):
    reasons ← danh sách rỗng
    NẾU kích thước transfers != (S, S, K) THÌ TRẢ VỀ (False, ["Sai kích thước"])

    // Ràng buộc 1: Hữu hạn, không âm và là số nguyên
    NẾU có phần tử NaN hoặc vô cực THÌ TRẢ VỀ (False, ["Giá trị không hữu hạn"])
    NẾU có phần tử < 0 THÌ thêm lỗi "Chuyển âm" vào reasons
    NẾU có phần tử không phải số nguyên THÌ thêm lỗi "Không nguyên" vào reasons
    Ép kiểu x ← số nguyên(transfers)

    // Ràng buộc 2: Không tự chuyển cho chính mình
    LẶP i từ 0 đến S-1:
        NẾU tồn tại k sao cho x[i, i, k] != 0 THÌ thêm lỗi "Tự chuyển" vào reasons

    // Ràng buộc 3: Không xuất vượt quá hàng dư an toàn
    outflow[i, k] ← tổng_j(x[i, j, k])
    NẾU tồn tại (i, k) có outflow[i, k] > max_surplus_send[i, k] THÌ thêm lỗi vào reasons

    // Ràng buộc 4: Tồn sau điều chuyển không âm và không vượt sức chứa
    inv_after ← compute_post_transfer_inventory(initial_inventory, x)
    NẾU tồn tại inv_after < 0 THÌ thêm lỗi "Tồn sau âm" vào reasons
    NẾU tồn tại inv_after[i, k] > capacity[i, k] THÌ thêm lỗi "Vượt sức chứa" vào reasons

    // Ràng buộc 5: Bảo toàn tổng tồn kho từng sản phẩm trên toàn mạng lưới
    NẾU tổng_i(inv_after[i, k]) != tổng_i(initial_inventory[i, k]) với SKU k nào đó THÌ
        thêm lỗi "Vi phạm bảo toàn hàng hóa" vào reasons

    TRẢ VỀ (reasons rỗng, reasons)
```

### 2.6 `compute_objective_from_demand(transfers, instance, demand_total)`
- **Mục đích:** Tính nhanh hàm mục tiêu tổng hợp $F(X)$ và toàn bộ thành phần KPI trên một ma trận nhu cầu 7 ngày (dùng chung cho bước tối ưu trên dự báo và đánh giá trên thực tế).
- **Đầu vào:** `transfers[S, S, K]`, `instance`, `demand_total[S, K]`.
- **Đầu ra:** Từ điển gồm `total_cost`, `transfer_cost`, `shortage_cost`, `holding_cost`, `total_demand`, `unmet_demand`, `leftover_inventory`, `fill_rate`, `total_units_moved`.

```text
HÀM compute_objective_from_demand(transfers, instance, demand_total):
    inv_after ← compute_post_transfer_inventory(instance.initial_inventory, transfers)

    // Chi phí vận chuyển: sum_{i,j,k} c[i,j] * x[i,j,k]
    move_cost ← TỔNG theo (i, j) của (tổng_k(transfers[i, j, k]) × transfer_cost[i, j])

    shortage_mat ← max(0, demand_total - inv_after)
    leftover_mat ← max(0, inv_after - demand_total)

    total_demand ← TỔNG(demand_total)
    unmet_demand ← TỔNG(shortage_mat)
    leftover_inv ← TỔNG(leftover_mat)

    shortage_cost ← instance.shortage_penalty × unmet_demand    // 10.0 USD/đv
    holding_cost  ← instance.leftover_cost × leftover_inv       // 0.5  USD/đv
    total_cost    ← move_cost + shortage_cost + holding_cost

    NẾU total_demand <= 0 THÌ
        fill_rate ← None   // Không tự động gán 100% khi cầu bằng 0
    NGƯỢC LẠI
        fill_rate ← 100.0 × (total_demand - unmet_demand) / total_demand

    TRẢ VỀ từ điển chứa các chi phí, unmet_demand, leftover_inv, fill_rate, tổng số đơn vị chuyển
```

### 2.7 `simulate_7_days(transfers, instance, use_actual=True)`
- **Mục đích:** Mô phỏng chi tiết hoạt động bán hàng từng ngày $t = 1..7$ sau khi áp dụng điều chuyển đầu kỳ, kiểm chứng khớp tuyệt đối với công thức tổng 7 ngày và tổng hợp bảng chi tiết theo từng cửa hàng × sản phẩm.
- **Đầu vào:** `transfers[S, S, K]`, `instance`, `use_actual` (`True`: dùng 7 ngày thực tế; `False`: chia đều dự báo cho 7 ngày).
- **Đầu ra:** Từ điển gồm `feasible`, `infeasible_reasons`, `metrics`, `store_item_table` (DataFrame 12 dòng), `daily_end_inventory`, `daily_unmet`.

```text
HÀM simulate_7_days(transfers, instance, use_actual):
    (feasible, reasons) ← check_feasible(transfers, instance)
    inv_after ← compute_post_transfer_inventory(initial_inventory, transfers)

    NẾU use_actual VÀ có actual_demand_daily THÌ
        daily_demand ← actual_demand_daily          // kích thước (S, K, 7)
    NGƯỢC LẠI
        daily_demand ← chia đều forecast_demand / 7.0 cho mỗi ngày t = 0..6

    inv_curr ← bản sao(inv_after)
    LẶP ngày t từ 0 đến 6:
        d_t      ← daily_demand[:, :, t]
        sold_t   ← min(inv_curr, d_t)               // Không bán vượt tồn kho
        unmet_t  ← d_t - sold_t                     // Lượng thiếu trong ngày t
        inv_curr ← inv_curr - sold_t                // Tồn kho cuối ngày t
        Lưu sold_t, unmet_t, inv_curr vào mảng lịch sử ngày

    demand_total ← tổng theo ngày của daily_demand
    metrics      ← compute_objective_from_demand(transfers, instance, demand_total)

    KIỂM CHỨNG (assert): tổng(unmet_daily) == metrics.unmet_demand
                         tổng(inv_curr)    == metrics.leftover_inventory

    Lập bảng store_item_table cho từng cặp (store, item): tồn đầu, nhập, xuất, tồn sau,
        dự báo, cầu đánh giá, đã bán, thiếu hàng, tồn cuối, fill_rate từng chuỗi
    TRẢ VỀ {feasible, reasons, metrics, store_item_table, daily_end_inventory, daily_unmet}
```

### 2.8 `transfers_to_dataframe(transfers, instance, method_name='')`
- **Mục đích:** Chuyển đổi mảng 3D `transfers` thành bảng danh sách các lệnh điều chuyển thực sự phát sinh (`quantity > 0`) để lưu CSV và hiển thị trên Streamlit.
- **Đầu vào:** `transfers[S, S, K]`, `instance`, `method_name`.
- **Đầu ra:** DataFrame 8 cột: `scenario`, `method`, `item_id`, `from_store`, `to_store`, `quantity`, `unit_cost_usd`, `transfer_cost_usd`.

```text
HÀM transfers_to_dataframe(transfers, instance, method_name):
    rows ← rỗng
    LẶP k từ 0 đến K-1:
        LẶP i từ 0 đến S-1:
            LẶP j từ 0 đến S-1:
                qty ← transfers[i, j, k]
                NẾU qty > 0 THÌ
                    unit_cost ← transfer_cost[i, j]
                    THÊM {scenario, method_name, items[k], stores[i], stores[j],
                          qty, unit_cost, qty × unit_cost} vào rows
    TRẢ VỀ DataFrame(rows) với đủ 8 cột chuẩn
```

---

## CHƯƠNG 3. `src/greedy.py` — Thuật toán Tham lam Đối chứng (Greedy)

*Thực hiện STEP 4: Thuật toán đối chứng cơ sở, chạy nhanh và minh bạch, chỉ sử dụng nhu cầu dự báo (`forecast_demand`), tuyệt đối không nhìn thấy tập kiểm tra 7 ngày.*

### 3.1 `solve_greedy(instance)`
- **Mục đích:** Tìm phương án điều chuyển tối ưu cục bộ theo nguyên tắc tham lam: ưu tiên cặp cửa hàng `(i → j)` có chi phí đơn vị thấp nhất trước, chỉ chuyển khi lợi ích biên lớn hơn chi phí vận chuyển.
- **Đầu vào:** `instance` (`ProblemInstance`).
- **Đầu ra:** `(transfers[S, S, K], info)`: ma trận lệnh điều chuyển khả thi và thông tin thời gian chạy, giá trị hàm mục tiêu dự báo.

```text
HÀM solve_greedy(instance):
    Bắt đầu bấm giờ t0
    transfers ← mảng số nguyên 0 kích thước (S, S, K)

    // Sắp xếp toàn bộ S*(S-1) cặp có hướng (i -> j, i != j) theo chi phí tăng dần
    pairs ← [(transfer_cost[i, j], i, j) với mọi i != j]
    SẮP XẾP pairs tăng dần theo (unit_cost, i, j)

    LẶP VỚI MỖI sản phẩm k từ 0 đến K-1:
        surplus[i]       ← max_surplus_send[i, k]
        rem_cap[j]       ← max(0, capacity[j, k] - initial_inventory[j, k])
        deficit_float[j] ← max(0.0, forecast_demand[j, k] - initial_inventory[j, k])

        LẶP VỚI MỖI (unit_cost, i, j) trong pairs:
            NẾU surplus[i] <= 0 HOẶC rem_cap[j] <= 0 HOẶC deficit_float[j] <= 0 THÌ
                TIẾP TỤC cặp tiếp theo

            // Bước A: Chuyển phần nguyên của lượng thiếu nếu lợi ích biên dương
            int_need ← floor(deficit_float[j])
            NẾU int_need > 0 VÀ unit_cost < (shortage_penalty + leftover_cost) THÌ
                qty ← min(surplus[i], int_need, rem_cap[j])
                NẾU qty > 0 THÌ
                    transfers[i, j, k] += qty
                    surplus[i]         -= qty
                    rem_cap[j]         -= qty
                    deficit_float[j]   -= qty

            // Bước B: Xét thêm 1 đơn vị cho phần thiếu lẻ (0 < frac < 1, ví dụ từ +30%/+50%)
            frac ← deficit_float[j]
            NẾU (0 < frac < 1) VÀ surplus[i] > 0 VÀ rem_cap[j] > 0 THÌ
                marginal_saving ← frac × shortage_penalty + leftover_cost - (1 - frac) × leftover_cost
                NẾU unit_cost < marginal_saving THÌ
                    transfers[i, j, k] += 1
                    surplus[i]         -= 1
                    rem_cap[j]         -= 1
                    deficit_float[j]   ← 0.0

    elapsed ← kết thúc bấm giờ - t0
    (feasible, reasons) ← check_feasible(transfers, instance)
    NẾU KHÔNG feasible THÌ BÁO LỖI RuntimeError
    forecast_eval ← compute_objective_from_demand(transfers, instance, forecast_demand)
    TRẢ VỀ (transfers, {method="Greedy", runtime_sec=elapsed, feasible, forecast_objective})
```

> **Ghi chú kỹ thuật:** Lợi ích biên của 1 đơn vị chuyển từ $i$ (đang dư) sang $j$ (đang thiếu) bằng `10.0` (giảm phạt thiếu tại $j$) + `0.5` (giảm tồn dư tại $i$) = `10.5 USD` > chi phí chuyển lớn nhất `3.5 USD`.

---

## CHƯƠNG 4. `src/kla.py` — Thuật toán Kirchhoff’s Law Algorithm (KLA)

*Thực hiện STEP 6: Cài đặt thuật toán tiến hóa lấy cảm hứng từ Định luật dòng điện Kirchhoff (Ghasemi et al., Artificial Intelligence Review, 2025), gồm lõi tối ưu liên tục, toán tử mã hóa/giải mã/sửa nghiệm khả thi và bộ giải KLA thuần.*

### 4.1 `kla_optimize_continuous(objective_fn, lb, ub, pop_size=15, max_iter=35, seed=42, init_solutions=None, post_iter_hook=None)`
- **Mục đích:** Lõi tối ưu hóa hàm liên tục nhiều chiều theo thuật toán KLA (không cần tham số điều khiển phức tạp), sử dụng chuẩn hóa cường độ `min/max` ổn định số học và hỗ trợ gắn hook hậu xử lý sau mỗi vòng lặp (dùng cho `KLA–LS`).
- **Đầu vào:** `objective_fn`, cận dưới `lb`, cận trên `ub`, `pop_size >= 4`, `max_iter`, `seed`, `init_solutions`, `post_iter_hook`.
- **Đầu ra:** `(best_x, best_f, history)`: vector nghiệm tốt nhất, giá trị hàm mục tiêu tốt nhất, và danh sách giá trị hội tụ qua `max_iter + 1` bước.

```text
HÀM kla_optimize_continuous(objective_fn, lb, ub, pop_size, max_iter, seed, init_solutions, post_iter_hook):
    Khởi tạo bộ sinh số ngẫu nhiên rng theo seed
    NẾU pop_size < 4 THÌ BÁO LỖI ValueError (cần chọn 3 cá thể khác biệt j, a, b)

    // Trường hợp biên: miền tìm kiếm suy biến (ub == lb ở mọi chiều)
    NẾU mọi chiều đều có ub <= lb + EPS THÌ
        f0 ← objective_fn(lb)
        TRẢ VỀ (lb, f0, [f0 lặp lại max_iter + 1 lần])

    // Khởi tạo quần thể đều trong [lb, ub] và gieo nghiệm khởi tạo (nếu có)
    pop ← sinh ngẫu nhiên (pop_size × dim) phân phối đều trong [lb, ub]
    NẾU có init_solutions THÌ ghi đè các cá thể đầu tiên của pop (đã kẹp vào [lb, ub])

    fitness[i] ← objective_fn(pop[i]) với mọi i = 0..pop_size-1
    (best_x, best_f) ← cá thể có fitness nhỏ nhất; history ← [best_f]

    LẶP vòng it từ 1 đến max_iter:
        LẶP từng cá thể i từ 0 đến pop_size-1:
            Chọn ngẫu nhiên 3 cá thể phân biệt (j, a, b) khác i (không hoàn lại)

            // Hướng dòng tương tác: +1 nếu đối tác tốt hơn i (hút về), -1 nếu kém hơn (đẩy ra)
            q   ← (f_i - f_j + EPS) / (|f_i - f_j| + EPS)
            q_a ← (f_i - f_a + EPS) / (|f_i - f_a| + EPS)
            q_b ← (f_i - f_b + EPS) / (|f_i - f_b| + EPS)

            // Tỷ lệ cường độ chuẩn hóa ổn định trong (0, 1]
            Sinh r_q, r_a, r_b ngẫu nhiên trong [0, 1)
            q1  ← ((min(|f_i|, |f_j|) + EPS) / (max(|f_i|, |f_j|) + EPS))^(2 * r_q)
            qa1 ← ((min(|f_i|, |f_a|) + EPS) / (max(|f_i|, |f_a|) + EPS))^(2 * r_a)
            qb1 ← ((min(|f_i|, |f_b|) + EPS) / (max(|f_i|, |f_b|) + EPS))^(2 * r_b)

            // 3 thành phần dòng tương tác theo định luật Kirchhoff
            s1 ← (q1  / q)   × (pop[j] - pop[i])
            s2 ← (qa1 / q_a) × (pop[a] - pop[i])
            s3 ← (qb1 / q_b) × (pop[b] - pop[i])

            Sinh 3 vector trọng số ngẫu nhiên r1, r2, r3 thuộc [0, 1)^dim
            x_new ← kẹp(pop[i] + r1*s1 + r2*s2 + r3*s3, lb, ub)
            f_new ← objective_fn(x_new)

            // Chọn lọc tham lam (Greedy selection)
            NẾU f_new < fitness[i] THÌ
                pop[i] ← x_new; fitness[i] ← f_new
                NẾU f_new < best_f THÌ best_f ← f_new; best_x ← x_new

        // Gọi hook tùy chọn (chèn Local Search trong KLA-LS)
        NẾU post_iter_hook khác None THÌ
            (cand_x, cand_f) ← post_iter_hook(it, best_x, best_f, rng)
            NẾU cand_f < best_f THÌ
                best_x ← kẹp(cand_x, lb, ub); best_f ← cand_f
                worst_idx ← chỉ số cá thể có fitness lớn nhất trong quần thể
                pop[worst_idx] ← best_x; fitness[worst_idx] ← best_f

        THÊM best_f vào history
    TRẢ VỀ (best_x, best_f, history)
```

### 4.2 `verify_kla_on_continuous_benchmark(seed=42)`
- **Mục đích:** Kiểm chứng hoạt động đúng đắn của lõi KLA trên hàm chuẩn liên tục Sphere 5 chiều $f(x) = \sum x_d^2$ trên $[-10, 10]^5$ (cực tiểu lý thuyết tại $0$).
- **Đầu vào:** `seed` (mặc định `42`).
- **Đầu ra:** Từ điển gồm `initial_best_f`, `final_best_f` và chuẩn Euclidean `best_norm`.

```text
HÀM verify_kla_on_continuous_benchmark(seed):
    Đặt dim = 5, lb = [-10, ..., -10], ub = [10, ..., 10]
    (best_x, best_f, history) ← kla_optimize_continuous(
        hàm_mục_tiêu = tổng(x^2), lb, ub, pop_size=20, max_iter=80, seed=seed
    )
    TRẢ VỀ {initial_best_f = history[0], final_best_f = best_f, best_norm = ||best_x||_2}
```

### 4.3 `_off_diagonal_indices(n_stores)`
- **Mục đích:** Sinh danh sách các cặp chỉ số `(i, j)` với `i != j` đại diện cho tất cả các tuyến điều chuyển có hướng hợp lệ (với 4 cửa hàng có đúng 12 tuyến).
- **Đầu vào:** `n_stores` (số cửa hàng $S$).
- **Đầu ra:** Danh sách các tuple `(i, j)` với `i != j`.

```text
HÀM _off_diagonal_indices(n_stores):
    TRẢ VỀ danh sách tất cả cặp (i, j) với i thuộc 0..n_stores-1, j thuộc 0..n_stores-1, i != j
```

### 4.4 `repair_sku_matrix(raw_mat, initial_inv_k, max_surplus_k, capacity_k, transfer_cost)`
- **Mục đích:** Toán tử sửa nghiệm (*repair operator*) biến một ma trận số thực bất kỳ thành ma trận điều chuyển nguyên không âm khả thi tuyệt đối cho 1 sản phẩm.
- **Đầu vào:** `raw_mat[S, S]`, `initial_inv_k[S]`, `max_surplus_k[S]`, `capacity_k[S]`, `transfer_cost[S, S]`.
- **Đầu ra:** Ma trận nguyên `x[S, S]` thỏa mãn 100% các ràng buộc khả thi.

```text
HÀM repair_sku_matrix(raw_mat, initial_inv_k, max_surplus_k, capacity_k, transfer_cost):
    // Bước 1: Làm tròn về số nguyên >= 0 và đặt đường chéo = 0
    x ← max(0, làm_tròn(raw_mat))
    Đặt x[i, i] ← 0 với mọi i

    // Bước 2: Triệt tiêu chuyển hai chiều ngược nhau giữa cùng một cặp (i, j)
    LẶP i từ 0 đến S-1:
        LẶP j từ i+1 đến S-1:
            both ← min(x[i, j], x[j, i])
            NẾU both > 0 THÌ x[i, j] -= both; x[j, i] -= both

    // Bước 3: Cắt tổng xuất vượt quá ngưỡng hàng dư an toàn max_surplus_k[i]
    LẶP i từ 0 đến S-1:
        out_sum ← tổng_j(x[i, j])
        NẾU out_sum > max_surplus_k[i] THÌ
            NẾU max_surplus_k[i] <= 0 THÌ đặt toàn bộ hàng x[i, :] ← 0
            NGƯỢC LẠI
                Co tỷ lệ x[i, :] ← floor(x[i, :] × max_surplus_k[i] / out_sum)
                excess ← tổng(x[i, :]) - max_surplus_k[i]
                NẾU vẫn còn excess > 0 THÌ cắt bớt theo thứ tự tuyến chi phí transfer_cost[i, j] giảm dần

    // Bước 4: Cắt tổng nhập nếu tồn sau điều chuyển vượt sức chứa capacity_k[j]
    LẶP j từ 0 đến S-1:
        inv_after ← initial_inv_k[j] + tổng_nhập(x[:, j]) - tổng_xuất(x[j, :])
        excess    ← inv_after - capacity_k[j]
        NẾU excess > 0 THÌ
            Duyệt các cửa hàng gửi i theo thứ tự transfer_cost[i, j] giảm dần (tuyến đắt cắt trước):
                cut ← min(x[i, j], excess)
                x[i, j] -= cut; excess -= cut
                NẾU excess <= 0 THÌ DỪNG
    TRẢ VỀ x
```

### 4.5 `decode_vector_to_sku_matrix(vec, n_stores, pairs, initial_inv_k, max_surplus_k, capacity_k, transfer_cost)`
- **Mục đích:** Giải mã vector liên tục 12 chiều (vị trí cá thể KLA) thành ma trận điều chuyển $S \times S$ nguyên khả thi.
- **Đầu vào:** `vec` (12 chiều), `n_stores`, `pairs`, và các vector ràng buộc của SKU $k$.
- **Đầu ra:** Ma trận nguyên khả thi `x[S, S]`.

```text
HÀM decode_vector_to_sku_matrix(vec, n_stores, pairs, ...):
    mat ← ma trận 0 kích thước (n_stores, n_stores)
    LẶP idx, (i, j) TRONG pairs:
        mat[i, j] ← vec[idx]
    TRẢ VỀ repair_sku_matrix(mat, initial_inv_k, max_surplus_k, capacity_k, transfer_cost)
```

### 4.6 `encode_sku_matrix_to_vector(mat, pairs)`
- **Mục đích:** Mã hóa ngược ma trận điều chuyển $S \times S$ thành vector thực 12 chiều để phản hồi nghiệm sau Local Search về lại quần thể KLA.
- **Đầu vào:** `mat[S, S]`, `pairs`.
- **Đầu ra:** Vector thực 1D độ dài `len(pairs)`.

```text
HÀM encode_sku_matrix_to_vector(mat, pairs):
    TRẢ VỀ vector [float(mat[i, j]) với từng (i, j) trong pairs]
```

### 4.7 `evaluate_sku_objective(x_mat, initial_inv_k, forecast_k, transfer_cost, shortage_penalty, leftover_cost)`
- **Mục đích:** Tính nhanh giá trị hàm mục tiêu dự báo $F_k(X_k)$ cho riêng một sản phẩm $k$.
- **Đầu vào:** `x_mat[S, S]`, `initial_inv_k`, `forecast_k`, `transfer_cost`, `shortage_penalty`, `leftover_cost`.
- **Đầu ra:** Số thực biểu diễn tổng chi phí dự báo của SKU $k$ (USD).

```text
HÀM evaluate_sku_objective(x_mat, initial_inv_k, forecast_k, transfer_cost, shortage_penalty, leftover_cost):
    inv_after ← initial_inv_k + tổng_cột(x_mat) - tổng_hàng(x_mat)
    move_cost ← TỔNG(x_mat × transfer_cost)
    shortage  ← TỔNG(max(0, forecast_k - inv_after))
    leftover  ← TỔNG(max(0, inv_after - forecast_k))
    TRẢ VỀ move_cost + shortage_penalty × shortage + leftover_cost × leftover
```

### 4.8 `build_sku_bounds(instance, k, pairs)`
- **Mục đích:** Xây dựng cận dưới (`lb = 0`) và cận trên tự nhiên (`ub`) cho từng biến quyết định ứng với cung `(i → j)` của sản phẩm $k$.
- **Đầu vào:** `instance`, chỉ số sản phẩm `k`, `pairs`.
- **Đầu ra:** `(lb, ub)`: hai vector cận độ dài `len(pairs)`.

```text
HÀM build_sku_bounds(instance, k, pairs):
    lb ← vector 0 độ dài len(pairs)
    ub ← vector 0 độ dài len(pairs)
    LẶP idx, (i, j) TRONG pairs:
        max_send ← max_surplus_send[i, k]
        max_recv ← max(0, (capacity[j, k] - initial_inventory[j, k]) + max_surplus_send[j, k])
        ub[idx]  ← max(0, min(max_send, max_recv))
    TRẢ VỀ (lb, ub)
```

### 4.9 `solve_kla(instance, pop_size=15, max_iter=35, seed=42)` *(kèm hàm nội bộ `obj_k(vec)`)*
- **Mục đích:** Chạy thuật toán KLA thuần (không Local Search) trên nhu cầu dự báo cho bài toán đa sản phẩm bằng cách tối ưu từng SKU độc lập trong không gian 12 chiều rồi ghép lại.
- **Đầu vào:** `instance`, `pop_size` (15), `max_iter` (35), `seed` (42).
- **Đầu ra:** `(transfers[S, S, K], info)`: phương án điều chuyển khả thi kèm lịch sử hội tụ `convergence_history` và thời gian chạy.

```text
HÀM solve_kla(instance, pop_size, max_iter, seed):
    Bắt đầu bấm giờ t0
    pairs         ← _off_diagonal_indices(S)
    transfers     ← mảng nguyên 0 kích thước (S, S, K)
    total_history ← vector 0 độ dài (max_iter + 1)

    LẶP VỚI MỖI sản phẩm k từ 0 đến K-1:
        (lb, ub) ← build_sku_bounds(instance, k, pairs)

        ĐỊNH NGHĨA HÀM NỘI BỘ obj_k(vec):
            mat ← decode_vector_to_sku_matrix(vec, S, pairs, init_k, surp_k, cap_k, transfer_cost)
            TRẢ VỀ evaluate_sku_objective(mat, init_k, fore_k, transfer_cost, ...)

        // Gieo sẵn 1 cá thể là nghiệm rỗng (x = 0) để KLA không bao giờ kém hơn No Transfer
        init_pop ← [vector 0 độ dài 12]

        (best_vec, _, hist_k) ← kla_optimize_continuous(
            obj_k, lb, ub, pop_size, max_iter, seed = seed*100 + k, init_solutions = init_pop
        )
        transfers[:, :, k] ← decode_vector_to_sku_matrix(best_vec, ...)
        total_history      += hist_k

    elapsed ← kết thúc bấm giờ - t0
    Kiểm tra check_feasible(transfers, instance) (nếu sai -> BÁO LỖI RuntimeError)
    forecast_eval ← compute_objective_from_demand(transfers, instance, forecast_demand)
    TRẢ VỀ (transfers, {method="KLA", seed, pop_size, max_iter, runtime_sec, feasible,
                        forecast_objective, convergence_history = total_history})
```

---

## CHƯƠNG 5. `src/local_search.py` — Tìm kiếm cục bộ & Thuật toán lai KLA–LS

*Thực hiện STEP 6: Cài đặt Tìm kiếm cục bộ (`Local Search`) với 3 phép biến đổi lân cận rời rạc và kết hợp dạng memetic với KLA (gọi Local Search định kỳ sau mỗi `ls_every` vòng và ở vòng cuối).*

### 5.1 `_is_sku_matrix_feasible(x_mat, initial_inv_k, max_surplus_k, capacity_k)`
- **Mục đích:** Kiểm tra nhanh tính khả thi của một ma trận ứng viên $S \times S$ trong quá trình duyệt lân cận của Local Search.
- **Đầu vào:** `x_mat[S, S]` và các vector ràng buộc của sản phẩm $k$.
- **Đầu ra:** `bool` (`True` nếu khả thi, `False` nếu vi phạm bất kỳ ràng buộc nào).

```text
HÀM _is_sku_matrix_feasible(x_mat, initial_inv_k, max_surplus_k, capacity_k):
    NẾU có phần tử x_mat < 0 HOẶC có phần tử đường chéo x_mat[i, i] != 0 THÌ TRẢ VỀ False
    outflow ← tổng theo hàng của x_mat
    NẾU tồn tại i có outflow[i] > max_surplus_k[i] THÌ TRẢ VỀ False
    inv_after ← initial_inv_k + tổng theo cột của x_mat - outflow
    NẾU tồn tại inv_after < 0 HOẶC inv_after > capacity_k THÌ TRẢ VỀ False
    TRẢ VỀ True
```

### 5.2 `local_search_sku(start_mat, ..., max_rounds=30, steps=(5, 2, 1))`
- **Mục đích:** Tìm kiếm cục bộ leo đồi (*first-improvement*) duyệt lần lượt 3 cấu trúc lân cận trên ma trận điều chuyển của 1 SKU, chỉ chấp nhận nước đi khả thi làm giảm chặt hàm mục tiêu dự báo.
- **Đầu vào:** `start_mat[S, S]`, các tham số SKU $k$, `max_rounds=30`, `steps=(5, 2, 1)`.
- **Đầu ra:** `(best_mat, best_obj, improvements)`: ma trận đã tinh chỉnh, giá trị mục tiêu mới và số bước cải thiện thành công.

```text
HÀM local_search_sku(start_mat, ..., max_rounds=30, steps=(5, 2, 1)):
    best_mat     ← bản sao nguyên(start_mat)
    best_obj     ← evaluate_sku_objective(best_mat, ...)
    improvements ← 0

    LẶP tối đa max_rounds vòng:
        improved_this_round ← False

        // LÂN CẬN 1: Tăng hoặc giảm lượng chuyển trên một tuyến (i -> j) theo bước ±5, ±2, ±1
        LẶP i từ 0 đến S-1 (bỏ qua nếu max_surplus_k[i] <= 0):
            LẶP j từ 0 đến S-1 (j != i):
                LẶP delta TRONG (5, 2, 1) VÀ sign TRONG (-1, +1):
                    new_val ← best_mat[i, j] + sign × delta
                    NẾU new_val < 0 THÌ bỏ qua
                    Tạo cand ← best_mat với cand[i, j] = new_val
                    NẾU _is_sku_matrix_feasible(cand) VÀ evaluate_sku_objective(cand) < best_obj - 1e-9 THÌ
                        Cập nhật best_mat ← cand; best_obj ← cand_obj
                        improvements += 1; improved_this_round ← True

        // LÂN CẬN 2 (Swap Sender): Chuyển bớt hoặc toàn bộ lượng từ (i -> j) sang (i_alt -> j)
        LẶP mọi cặp (i, j) có i != j VÀ best_mat[i, j] > 0:
            LẶP mọi cửa hàng gửi thay thế i_alt khác i, j có max_surplus_k[i_alt] > 0:
                Thử các mức delta giảm dần trong {cur_qty, min(cur_qty, 5), min(cur_qty, 2), min(cur_qty, 1)}:
                    Tạo cand: cand[i, j] -= delta; cand[i_alt, j] += delta
                    NẾU cand khả thi VÀ cand_obj < best_obj - 1e-9 THÌ
                        Cập nhật best_mat, best_obj, improvements += 1, improved_this_round ← True
                        THOÁT vòng delta

        // LÂN CẬN 3 (Swap Receiver): Chuyển hướng lượng từ (i -> j) sang (i -> j_alt)
        LẶP mọi cặp (i, j) có i != j VÀ best_mat[i, j] > 0:
            LẶP mọi cửa hàng nhận thay thế j_alt khác i, j:
                Thử các mức delta giảm dần trong {cur_qty, min(cur_qty, 5), min(cur_qty, 2), min(cur_qty, 1)}:
                    Tạo cand: cand[i, j] -= delta; cand[i, j_alt] += delta
                    NẾU cand khả thi VÀ cand_obj < best_obj - 1e-9 THÌ
                        Cập nhật best_mat, best_obj, improvements += 1, improved_this_round ← True
                        THOÁT vòng delta

        NẾU KHÔNG improved_this_round THÌ THOÁT vòng lặp (đã đạt cực tiểu cục bộ)

    TRẢ VỀ (best_mat, best_obj, improvements)
```

### 5.3 `improve_transfers_local_search(start_transfers, instance, max_rounds=30)`
- **Mục đích:** Áp dụng Local Search độc lập lên toàn bộ phương án điều chuyển 3D `(S × S × K)` cho tất cả các sản phẩm.
- **Đầu vào:** `start_transfers[S, S, K]`, `instance`, `max_rounds`.
- **Đầu ra:** `(improved[S, S, K], info)`: phương án sau cải thiện và số bước cải thiện tổng cộng.

```text
HÀM improve_transfers_local_search(start_transfers, instance, max_rounds):
    Bắt đầu bấm giờ t0
    improved ← mảng nguyên 0 kích thước (S, S, K); total_improvements ← 0
    LẶP k từ 0 đến K-1:
        (mat_k, _, n_imp) ← local_search_sku(start_transfers[:, :, k], tham số SKU k, max_rounds)
        improved[:, :, k]  ← mat_k
        total_improvements += n_imp

    Kiểm tra check_feasible(improved, instance) (nếu sai -> BÁO LỖI RuntimeError)
    eval_res ← compute_objective_from_demand(improved, instance, forecast_demand)
    TRẢ VỀ (improved, {runtime_sec, feasible, n_improvements = total_improvements,
                       forecast_objective = eval_res.total_cost})
```

### 5.4 `solve_kla_ls(instance, pop_size=15, max_iter=35, seed=42, ls_every=10, ls_max_rounds=25)` *(kèm `obj_k`, `ls_hook`)*
- **Mục đích:** Thuật toán lai memetic `KLA–LS`: kết hợp khả năng khám phá toàn cục của KLA với khả năng khai thác cục bộ chính xác của Local Search định kỳ sau mỗi `ls_every` vòng lặp và tại vòng cuối.
- **Đầu vào:** `instance`, `pop_size`, `max_iter`, `seed`, `ls_every` (mặc định `10`), `ls_max_rounds`.
- **Đầu ra:** `(transfers[S, S, K], info)`: phương án tối ưu khả thi kèm số lần Local Search cải thiện và đường hội tụ.

```text
HÀM solve_kla_ls(instance, pop_size, max_iter, seed, ls_every, ls_max_rounds):
    Bắt đầu bấm giờ t0
    pairs ← _off_diagonal_indices(S)
    transfers ← mảng 0 (S, S, K); total_history ← vector 0; total_ls_improvements ← 0

    LẶP VỚI MỖI sản phẩm k từ 0 đến K-1:
        (lb, ub) ← build_sku_bounds(instance, k, pairs)
        ĐỊNH NGHĨA obj_k(vec): giải mã vec -> sửa khả thi -> trả về evaluate_sku_objective

        ĐỊNH NGHĨA HÀM HOOK ls_hook(it, best_vec, best_f, rng):
            NẾU (it không chia hết cho ls_every) VÀ (it != max_iter) THÌ
                TRẢ VỀ (best_vec, best_f)   // Không chạy LS ở vòng này
            cur_mat ← decode_vector_to_sku_matrix(best_vec, ...)
            (imp_mat, imp_f, n_imp) ← local_search_sku(cur_mat, ..., max_rounds=ls_max_rounds)
            Cộng dồn n_imp vào bộ đếm ls_counter
            NẾU imp_f < best_f THÌ
                TRẢ VỀ (encode_sku_matrix_to_vector(imp_mat, pairs), imp_f)
            TRẢ VỀ (best_vec, best_f)

        init_pop ← [vector 0 độ dài 12]
        (best_vec, _, hist_k) ← kla_optimize_continuous(
            obj_k, lb, ub, pop_size, max_iter, seed = seed*100 + k,
            init_solutions = init_pop, post_iter_hook = ls_hook
        )
        transfers[:, :, k]    ← decode_vector_to_sku_matrix(best_vec, ...)
        total_history         += hist_k
        total_ls_improvements += ls_counter

    Kiểm tra check_feasible(transfers, instance)
    forecast_eval ← compute_objective_from_demand(transfers, instance, forecast_demand)
    TRẢ VỀ (transfers, {method="KLA-LS", seed, pop_size, max_iter, ls_every,
                        ls_improvements, runtime_sec, feasible, forecast_objective,
                        convergence_history = total_history})
```

---

## CHƯƠNG 6. `src/run_experiments.py` — Thực nghiệm, Tổng hợp & Trực quan hóa

*Thực hiện STEP 7: Chạy đánh giá thống nhất 03 kịch bản (`NORMAL`, `FLASH_SALE`, `SURGE`) × 04 phương pháp (`No Transfer`, `Greedy`, `KLA` 5 seeds, `KLA–LS` 5 seeds), xuất toàn bộ CSV, nhật ký Markdown và 03 biểu đồ so sánh.*

### 6.1 `_build_record(instance, method, seed, transfers, runtime_sec)`
- **Mục đích:** Đánh giá một phương án điều chuyển trên cả tập kiểm tra 7 ngày thực tế (*out-of-sample*) và trên nhu cầu dự báo (*in-sample fitness*) để tạo 1 bản ghi dòng cho `results/summary.csv`.
- **Đầu vào:** `instance`, `method`, `seed`, `transfers`, `runtime_sec`.
- **Đầu ra:** Từ điển 14 trường KPI chuẩn.

```text
HÀM _build_record(instance, method, seed, transfers, runtime_sec):
    sim_actual    ← simulate_7_days(transfers, instance, use_actual=True)
    eval_forecast ← compute_objective_from_demand(transfers, instance, instance.forecast_demand)
    m ← sim_actual.metrics
    TRẢ VỀ từ điển làm tròn đầy đủ 14 trường KPI cùng trạng thái sim_actual.feasible
```

### 6.2 `plot_compare_cost(summary_df, out_path)`
- **Mục đích:** Vẽ biểu đồ cột nhóm so sánh tổng chi phí trung bình (USD) trên tập test 7 ngày giữa 4 phương pháp qua 3 kịch bản.
- **Đầu vào:** `summary_df`, `out_path` (`results/compare_cost.png`).
- **Đầu ra:** Lưu ảnh PNG độ phân giải 150 DPI tại `out_path`.

```text
HÀM plot_compare_cost(summary_df, out_path):
    Nhóm summary_df theo (scenario, method) và tính trung bình total_cost
    Xoay bảng pivot: hàng = [NORMAL, FLASH_SALE, SURGE], cột = [No Transfer, Greedy, KLA, KLA-LS]
    Vẽ biểu đồ cột nhóm, gắn nhãn số liệu trên đỉnh mỗi cột, thêm chú giải & lưới ngang
    LƯU ảnh ra out_path và đóng hình
```

### 6.3 `plot_compare_service(summary_df, out_path)`
- **Mục đích:** Vẽ biểu đồ đôi (1 hàng × 2 cột): bên trái là tổng lượng thiếu hàng trung bình (đơn vị), bên phải là tỷ lệ đáp ứng Fill Rate (%) theo kịch bản.
- **Đầu vào:** `summary_df`, `out_path` (`results/compare_service.png`).
- **Đầu ra:** Lưu ảnh PNG độ phân giải 150 DPI tại `out_path`.

```text
HÀM plot_compare_service(summary_df, out_path):
    Tính bảng pivot trung bình cho unmet_demand và fill_rate theo (scenario × method)
    Tạo khung hình gồm 2 biểu đồ con cạnh nhau (axes[0], axes[1]):
      - axes[0]: Biểu đồ cột lượng thiếu hàng (trục y kéo giãn 1.15 lần giá trị lớn nhất)
      - axes[1]: Biểu đồ cột Fill Rate (%) (trục y từ 0..128 để không đè chú giải, nhãn xoay 25°)
    Đặt tiêu đề chung, căn lề gọn và LƯU ảnh ra out_path
```

### 6.4 `plot_convergence(conv_df, out_path)`
- **Mục đích:** Vẽ đường hội tụ hàm mục tiêu dự báo $F(X)$ trung bình qua 5 seeds của `KLA` (nét đứt xanh) và `KLA–LS` (nét liền đỏ) trên 3 kịch bản.
- **Đầu vào:** `conv_df`, `out_path` (`results/convergence_kla.png`).
- **Đầu ra:** Lưu ảnh PNG tại `out_path`.

```text
HÀM plot_convergence(conv_df, out_path):
    NẾU conv_df rỗng THÌ DỪNG
    Tạo 1 hàng gồm 3 đồ thị con ứng với 3 kịch bản (NORMAL, FLASH_SALE, SURGE)
    LẶP từng kịch bản scen:
        LẶP method TRONG ["KLA", "KLA-LS"]:
            Tính giá trị trung bình forecast_objective theo từng vòng lặp (iteration 0..35)
            Vẽ đường hội tụ trên đồ thị con tương ứng
    Đặt tiêu đề chung và LƯU ảnh ra out_path
```

### 6.5 `main()`
- **Mục đích:** Điều phối toàn bộ thí nghiệm: kiểm tra đầu vào, chạy 4 thuật toán trên 3 kịch bản (5 seeds cho KLA & KLA–LS), xuất 8 file CSV, 3 ảnh PNG và file nhật ký `results/experiment_log.md`.
- **Đầu vào:** Không có tham số.
- **Đầu ra:** Mã thoát `0` khi hoàn tất.

```text
HÀM main():
    Kiểm tra sự tồn tại của 4 file CSV đầu vào trong data/ (thiếu -> BÁO LỖI FileNotFoundError)
    Khởi tạo các danh sách gom kết quả: all_records, transfer_dfs, conv_rows, log_lines

    LẶP VỚI MỖI kịch bản scen TRONG ["NORMAL", "FLASH_SALE", "SURGE"]:
        instance ← load_problem_instance(DATA_DIR, scen)

        // 1. No Transfer (seed = 0)
        x_none ← no_transfer_plan(instance)
        Thêm _build_record(instance, "No Transfer", 0, x_none, rt) vào all_records

        // 2. Greedy (seed = 0)
        (x_greedy, info_g) ← solve_greedy(instance)
        Thêm _build_record(...) vào all_records; lưu lệnh chuyển vào greedy_transfer_dfs

        // 3. KLA (5 seeds: 42..46)
        LẶP seed TRONG [42, 43, 44, 45, 46]:
            (x_kla, info_k) ← solve_kla(instance, pop_size=15, max_iter=35, seed=seed)
            Thêm _build_record(...) vào all_records
            Cập nhật best_kla_x nếu forecast_objective nhỏ hơn kỷ lục hiện tại
            Ghi toàn bộ convergence_history vào conv_rows
        Lưu bảng lệnh chuyển của best_kla_x vào kla_transfer_dfs

        // 4. KLA-LS (5 seeds: 42..46)
        LẶP seed TRONG [42, 43, 44, 45, 46]:
            (x_klals, info_kl) ← solve_kla_ls(instance, pop_size=15, max_iter=35, seed=seed, ls_every=10)
            Thêm _build_record(...) vào all_records
            Cập nhật best_klals_x nếu forecast_objective nhỏ hơn kỷ lục hiện tại
            Ghi toàn bộ convergence_history vào conv_rows
        Lưu bảng lệnh chuyển của best_klals_x vào kla_ls_transfer_dfs

    // Xuất toàn bộ kết quả
    GHI summary.csv, baseline_summary.csv, kla_summary.csv, kla_ls_summary.csv
    GHI transfers_greedy.csv, transfers_kla.csv, transfers_kla_ls.csv, kla_convergence.csv
    GỌI plot_compare_cost, plot_compare_service, plot_convergence
    Tính bảng tổng hợp trung bình & độ lệch chuẩn theo (scenario, method)
    GHI toàn bộ nhật ký thực nghiệm ra "results/experiment_log.md"
    TRẢ VỀ 0
```

---

## CHƯƠNG 7. `src/verify_pipeline.py` — Bộ Kiểm thử Độc lập Toàn hệ thống

*Chạy kiểm thử đơn vị và tích hợp hoàn toàn trong RAM (không ghi đè bất kỳ file dữ liệu thật nào) nhằm xác minh tất cả yêu cầu kiểm định kỹ thuật (STEP 4.6 và STEP 6.2).*

### 7.1 `_make_instance(name, init_inv, safety, cap, forecast, actual_total=None)`
- **Mục đích:** Hàm tiện ích tạo nhanh một đối tượng `ProblemInstance` nhân tạo (4 cửa hàng × 2 SKU) trong RAM để phục vụ các bài kiểm thử.
- **Đầu vào:** Tên kịch bản, ma trận tồn đầu kỳ, ngưỡng an toàn, sức chứa, nhu cầu dự báo và nhu cầu thực tế tùy chọn.
- **Đầu ra:** Đối tượng `ProblemInstance` hoàn chỉnh với ma trận chi phí cố định `COST_MAT`.

```text
HÀM _make_instance(name, init_inv, safety, cap, forecast, actual_total):
    max_surp ← max(0, init_inv - safety)
    NẾU actual_total là None THÌ actual_total ← bản sao(forecast)
    actual_daily ← chia đều actual_total / 7.0 cho 7 ngày
    TRẢ VỀ ProblemInstance(name, STORES, ITEMS, init_inv, safety, max_surp, cap,
                           forecast, actual_daily, actual_total, COST_MAT)
```

### 7.2 `test_edge_cases()`
- **Mục đích:** Kiểm chứng 03 trường hợp biên bằng tay theo STEP 4.6: cả `Greedy` và `KLA–LS` bắt buộc trả về phương án rỗng (`tổng lượng chuyển = 0`).
- **Đầu vào / Đầu ra:** Không có tham số; ném `AssertionError` nếu bất kỳ điều kiện biên nào sai.

```text
HÀM test_edge_cases():
    // Case A: Không có cửa hàng nào thiếu hàng (tồn đầu >= dự báo ở mọi nơi)
    Tạo inst_a -> chạy Greedy & KLA-LS -> KIỂM CHỨNG tổng lượng chuyển == 0

    // Case B: Không có cửa hàng nào có hàng dư an toàn (tồn đầu <= safety_target ở mọi nơi)
    Tạo inst_b -> chạy Greedy & KLA-LS -> KIỂM CHỨNG tổng lượng chuyển == 0

    // Case C: Cửa hàng đang thiếu (CA_1) đã đầy kho (capacity == initial_inventory)
    Tạo inst_c -> chạy Greedy & KLA-LS -> KIỂM CHỨNG tổng lượng chuyển == 0
```

### 7.3 `test_fill_rate_zero_demand_and_feasibility()`
- **Mục đích:** Kiểm chứng quy tắc đặc biệt `FillRate = None` khi `TotalDemand = 0` (không tự gán `100%`) và xác nhận `check_feasible` phát hiện đúng nghiệm vi phạm.
- **Đầu vào / Đầu ra:** Không có tham số; ném `AssertionError` nếu sai.

```text
HÀM test_fill_rate_zero_demand_and_feasibility():
    Tạo kịch bản inst_z có tổng nhu cầu = 0 ở mọi cửa hàng
    Chạy simulate_7_days(no_transfer_plan) -> KIỂM CHỨNG fill_rate IS None

    Tạo nghiệm lỗi bad_self có x[0, 0, 0] = 2 (tự chuyển cho chính mình)
    KIỂM CHỨNG check_feasible(bad_self) trả về False

    Tạo nghiệm lỗi bad_neg có x[1, 0, 0] = -1 (chuyển số lượng âm)
    KIỂM CHỨNG check_feasible(bad_neg) trả về False
```

### 7.4 `test_kla_continuous()`
- **Mục đích:** Kiểm chứng lõi thuật toán KLA liên tục hội tụ tốt trên hàm chuẩn Sphere 5 chiều (STEP 6.2).
- **Đầu vào / Đầu ra:** Không có tham số; ném `AssertionError` nếu giá trị tốt nhất `>= 0.01`.

```text
HÀM test_kla_continuous():
    res ← verify_kla_on_continuous_benchmark(seed=42)
    KIỂM CHỨNG res.final_best_f < 1e-2
```

### 7.5 `test_optimization_and_local_search()`
- **Mục đích:** Kiểm thử tích hợp end-to-end 4 phương pháp trên kịch bản nhân tạo có thiếu–dư rõ rệt, và kiểm thử riêng Local Search trên một nghiệm cố tình dưới tối ưu.
- **Đầu vào / Đầu ra:** Không có tham số; ném `AssertionError` nếu sai.

```text
HÀM test_optimization_and_local_search():
    Tạo kịch bản nhân tạo SYNTHETIC_CHECK theo đúng quy tắc 50% / 160% / 160% / 110%
    Chạy cả 4 phương pháp: No Transfer, Greedy, KLA, KLA-LS -> tính F_none, F_greedy, F_kla, F_klals

    // Tạo nghiệm cố tình dưới tối ưu (chuyển từ tuyến đắt CA_4 -> CA_1 thay vì CA_2 -> CA_1)
    Chạy improve_transfers_local_search(subopt, inst) -> nhận F_sub_after

    KIỂM CHỨNG:
        1. F_greedy < F_none              (Điều chuyển có lợi hơn không chuyển)
        2. F_klals <= F_kla + 1e-6        (KLA-LS tốt hơn hoặc bằng KLA thuần)
        3. F_sub_after < F_sub_before     (Local Search thực sự cải thiện nghiệm)
        4. check_feasible == True cho cả KLA và KLA-LS
```

### 7.6 `main()`
- **Mục đích:** Điểm vào chạy tuần tự tất cả 4 bộ kiểm thử và in thông báo `ALL PASS`.
- **Đầu vào / Đầu ra:** Không có tham số; trả về mã thoát `0`.

```text
HÀM main():
    GỌI test_edge_cases()
    GỌI test_fill_rate_zero_demand_and_feasibility()
    GỌI test_kla_continuous()
    GỌI test_optimization_and_local_search()
    IN "=== TẤT CẢ KIỂM THỬ LOGIC ĐỀU ĐẠT (ALL PASS) ==="
    TRẢ VỀ 0
```
