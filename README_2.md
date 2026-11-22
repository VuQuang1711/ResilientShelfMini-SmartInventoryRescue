# Team Chuyển Kho — HUIT Data Science Challenge 2026 *(Phiên bản cập nhật Part 2)*

**Tên đề tài:** Tối ưu điều chuyển tồn kho đa cửa hàng dưới nhu cầu bất định bằng thuật toán lai Kirchhoff's Law Algorithm và Tìm kiếm cục bộ (`KLA–LS`).

---

## 1. Phạm vi phiên bản sơ khảo (MVP)
- **Mạng lưới:** 04 cửa hàng `CA_1`, `CA_2`, `CA_3`, `CA_4` từ bộ dữ liệu công khai **Kaggle M5 Forecasting – Accuracy**.
- **Hàng hóa:** 03 sản phẩm (`FOODS_3_090`, `FOODS_3_586`, `FOODS_3_252`) có doanh số dương tại cả 04 cửa hàng trong 49 ngày cuối (`d_1865..d_1913`).
- **Phân tách thời gian (chống rò rỉ dữ liệu):** 42 ngày đầu (`d_1865..d_1906`) làm lịch sử cho EDA, dự báo và sinh kịch bản tồn kho; 07 ngày cuối (`d_1907..d_1913`) giữ riêng hoàn toàn làm tập kiểm tra (*out-of-sample*).
- **Bài toán:** Một quyết định điều chuyển tồn kho đầu kỳ, quan sát tác động trong 07 ngày *(giả định chuyển ngay trước ngày 1, chưa xét độ trễ vận chuyển — `lead time = 0`)*.
- **Thuật toán đối chứng & đề xuất:** `No Transfer`, `Greedy`, `KLA` (*Kirchhoff's Law Algorithm*) và `KLA–LS` (*KLA lai Local Search 3 lân cận*).

---

## 2. Cấu trúc thư mục hiện tại (Cập nhật Part 2)

```text
Team_Chuyen_Kho/
├── 0_menu_chay_du_an.bat              # Menu tổng: kiểm tra Python/thư viện & chọn chạy bước 1–6
├── 1_cai_dat_thu_vien.bat             # Bước 1: Kiểm tra phiên bản Python/thư viện & hỏi cài/nâng cấp
├── 2_kiem_thu_logic.bat               # Bước 2: Chạy kiểm thử tự động src/verify_pipeline.py
├── 3_xu_ly_du_lieu_va_eda.bat         # Bước 3: Chạy src/prepare_data.py (lọc M5, EDA, kịch bản)
├── 4_chay_thuc_nghiem.bat             # Bước 4: Chạy src/run_experiments.py (3 kịch bản × 4 thuật toán)
├── 5_mo_app_streamlit.bat             # Bước 5: Khởi chạy ứng dụng Demo Streamlit (app/app.py)
├── data/
│   ├── raw/                           # Đặt file sales_train_validation.csv gốc từ Kaggle (bị .gitignore chặn)
│   ├── m5_selected.csv                # 12 chuỗi (4 cửa hàng × 3 sản phẩm) × 49 ngày cuối
│   ├── demand_forecast.csv            # Dự báo 7 ngày từ 7 ngày lịch sử cuối
│   ├── demand_actual_test.csv         # Doanh số thực 7 ngày cuối (chỉ dùng kiểm tra)
│   ├── inventory_scenario.csv         # Tồn đầu kỳ, ngưỡng an toàn, sức chứa (mô phỏng minh bạch)
│   ├── transfer_cost.csv              # Ma trận chi phí điều chuyển đối xứng 1.0–3.5 USD/đv (mô phỏng)
│   ├── scenarios.csv                  # 3 kịch bản: NORMAL, FLASH_SALE (+30%), SURGE (+50%)
│   └── data_dictionary.md             # Từ điển dữ liệu đầy đủ
├── docs/
│   ├── assumptions.md                 # Công bố toàn bộ giả định và quy tắc mô phỏng
│   ├── giai_thich_code_source.md      # Giải thích toàn bộ 42/42 lớp và hàm trong src/ bằng mã giả
│   ├── tom_tat_y_tuong_dang_ky.md     # Bản nháp tóm tắt ý tưởng đăng ký (280 từ <= 300 từ)
│   ├── kich_ban_video_va_qa.md        # Kịch bản video sơ khảo (< 5 phút) & 7 câu Q&A phản biện
│   ├── thanh_vien_va_phan_cong.md     # Danh sách thành viên & bảng phân công STEP 0
│   ├── review_task.md                 # Báo cáo rà soát toàn dự án so với hướng dẫn
│   ├── Form đăng ký Team Chuyển Kho.pdf
│   ├── Thông báo Tổ chức cuộc thi HUIT Data Science Challenge 2026.pdf
│   ├── Team_Chuyen_Kho_Huong_dan_So_khao_Step_by_Step.pdf
│   └── document/
│       └── s10462-025-11289-5.pdf     # Bài báo gốc KLA (Ghasemi et al., Artificial Intelligence Review 2025)
├── src/
│   ├── prepare_data.py                # STEP 1–3: Cắt M5, vẽ eda_01..03.png, dự báo, sinh kịch bản
│   ├── simulator.py                   # STEP 4: Mô phỏng 7 ngày, kiểm tra 5 ràng buộc khả thi, hàm F(X)
│   ├── greedy.py                      # STEP 4: Thuật toán tham lam đối chứng (Greedy)
│   ├── kla.py                         # STEP 6: Lõi KLA liên tục + toán tử sửa nghiệm khả thi + KLA thuần
│   ├── local_search.py                # STEP 6: Tìm kiếm cục bộ 3 lân cận & thuật toán lai KLA–LS
│   ├── run_experiments.py             # STEP 7: Thực nghiệm 3 kịch bản × 4 phương pháp × 5 seeds, tự đánh số run_XX
│   └── verify_pipeline.py             # Bộ kiểm thử độc lập trong RAM (3 case biên, Sphere 5D, KLA–LS)
├── app/
│   └── app.py                         # Ứng dụng Demo 1 trang Streamlit (chế độ Tính trước & Chạy trực tiếp Live)
├── results/                           # Kết quả chuẩn nộp vòng sơ khảo
│   ├── summary.csv                    # Tổng hợp đầy đủ 36 lần chạy (chuẩn STEP 7.4)
│   ├── baseline_summary.csv           # Đối chứng No Transfer & Greedy (STEP 4)
│   ├── kla_summary.csv                # Kết quả KLA thuần qua 5 seeds (STEP 6)
│   ├── kla_ls_summary.csv             # Kết quả KLA–LS qua 5 seeds (STEP 6)
│   ├── transfers_greedy.csv           # Lệnh điều chuyển chi tiết của Greedy (STEP 4)
│   ├── kla_convergence.csv            # Dữ liệu nguồn lịch sử hội tụ qua 35 vòng lặp
│   ├── eda_01.png .. eda_03.png       # 3 biểu đồ EDA dữ liệu M5
│   ├── compare_cost.png               # Biểu đồ so sánh tổng chi phí theo kịch bản
│   ├── compare_service.png            # Biểu đồ lượng thiếu hàng & Fill Rate
│   ├── convergence_kla.png            # Biểu đồ hội tụ KLA vs KLA–LS (5 seeds)
│   ├── eda_report.md                  # Báo cáo kiểm tra chất lượng dữ liệu & EDA
│   ├── experiment_log.md              # Nhật ký lần chạy mới nhất (ghi mã run_01, run_02, ...)
│   ├── logs/                          # Lưu trữ riêng từng lần chạy không bị ghi đè (run_01.md, run_02.md, ...)
│   └── SummaryForDemo/                # Bộ 4 file CSV tách riêng phục vụ trực tiếp cho App Streamlit
│       ├── summary.csv
│       ├── transfers_greedy.csv
│       ├── transfers_kla.csv
│       └── transfers_kla_ls.csv
├── report/                            # Nơi lưu bao_cao_so_khao.pdf
├── video/                             # Nơi lưu demo_so_khao.mp4
├── requirements.txt
├── README.md
└── README_2.md
```

---

## 3. Hướng dẫn chạy nhanh trên Windows

### Cách 1: Chạy trực tiếp bằng Menu `.bat` *(Khuyến nghị — chỉ cần bấm đúp)*
1. Đảm bảo đã tải `sales_train_validation.csv` từ [Kaggle M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) và đặt vào `data/raw/sales_train_validation.csv`.
2. Bấm đúp vào **`0_menu_chay_du_an.bat`**:
   - Tự động hiển thị phiên bản Python và kiểm tra nhanh toàn bộ thư viện trong `requirements.txt`.
   - Chọn số **`1`–`5`** để chạy từng bước riêng lẻ, hoặc chọn **`6`** để chạy tự động liên hoàn từ Bước 2 → Mở App Streamlit.
3. Hoặc bấm đúp lần lượt từng file riêng: `1_cai_dat_thu_vien.bat` → `2_kiem_thu_logic.bat` → `3_xu_ly_du_lieu_va_eda.bat` → `4_chay_thuc_nghiem.bat` → `5_mo_app_streamlit.bat`.

### Cách 2: Chạy bằng dòng lệnh PowerShell / Terminal
```powershell
# Bước 1: Cài đặt thư viện
py -m pip install -r requirements.txt

# Bước 2: Kiểm thử logic thuật toán (chạy độc lập trong RAM)
py src/verify_pipeline.py

# Bước 3: Xử lý dữ liệu M5, vẽ EDA và tạo kịch bản
py src/prepare_data.py

# Bước 4: Chạy thực nghiệm 3 kịch bản × 4 thuật toán × 5 seeds (tự đánh số run_01, run_02...)
py src/run_experiments.py

# Bước 5: Khởi chạy giao diện Demo Streamlit
py -m streamlit run app/app.py
```

---

## 4. Tóm tắt kết quả thực nghiệm (Tập kiểm tra 7 ngày thực tế)

| Kịch bản | No Transfer *(Chi phí \| Thiếu \| FR)* | Greedy *(Chi phí \| Thiếu \| FR)* | KLA 5 seeds *(Chi phí ± Std \| FR)* | KLA–LS 5 seeds *(Chi phí ± Std \| FR)* | Mức cải thiện KLA–LS vs No Transfer |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NORMAL** | `5,532.5 ` \| `471.0 đv` \| `87.40%` | `1,946.0 ` \| `67.0 đv` \| `98.21%` | `1,972.7 ± 55.2 ` \| `98.21%` | `1,946.0 ± 0.0 ` \| `67.0 đv` \| `98.21%` | **-64.8% chi phí**, **+10.8% Fill Rate** |
| **FLASH_SALE (+30%)** | `6,675.5 ` \| `585.3 đv` \| `84.81%` | `2,415.0 ` \| `93.3 đv` \| `97.58%` | `2,430.6 ± 63.3 ` \| `97.79%` | `2,415.0 ± 0.0 ` \| `93.3 đv` \| `97.58%` | **-63.8% chi phí**, **+12.8% Fill Rate** |
| **SURGE (+50%)** | `7,437.5 ` \| `661.5 đv` \| `83.16%` | `2,585.0 ` \| `95.5 đv` \| `97.57%` | `2,585.3 ± 37.5 ` \| `97.63%` | `2,585.0 ± 0.0 ` \| `95.5 đv` \| `97.57%` | **-65.2% chi phí**, **+14.4% Fill Rate** |

---

## 5. Tài liệu tham khảo chính
1. Khoa Công nghệ Thông tin, Trường Đại học Công Thương TP.HCM (2026). *Thể lệ & Thông báo số 2 — HUIT Data Science Challenge 2026*.
2. Kaggle, *M5 Forecasting – Accuracy*, https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
3. Ghasemi, M., Khodadadi, N., Trojovský, P., Li, L., Mansor, Z., Abualigah, L., Alharbi, A. H., & El-Kenawy, E.-S. M. (2025). *Kirchhoff’s law algorithm (KLA): a novel physics-inspired non-parametric metaheuristic algorithm for optimization problems*. Artificial Intelligence Review, 58, 325. https://doi.org/10.1007/s10462-025-11289-5
4. Mã nguồn tham khảo KLA trên MathWorks File Exchange: https://www.mathworks.com/matlabcentral/fileexchange/181589-kirchhoff-s-law-algorithm-kla
