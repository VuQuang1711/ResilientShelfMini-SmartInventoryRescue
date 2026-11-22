# Báo cáo Rà soát Toàn bộ Dự án so với Tài liệu Hướng dẫn (Review Task)

- **Đội thi:** Team Chuyển Kho — HUIT Data Science Challenge 2026
- **Thời điểm rà soát:** 22/11/2026
- **Tài liệu đối chiếu:** `docs/Team_Chuyen_Kho_Huong_dan_So_khao_Step_by_Step.pdf`

---

## 1. Kết quả kiểm tra tự động toàn hệ thống

| Hạng mục kiểm tra | Kết quả | Chi tiết xác minh |
| :--- | :---: | :--- |
| Bộ kiểm thử logic (`src/verify_pipeline.py`) | ✅ PASS | Pass cả 3 case biên (không thiếu / không dư / hết sức chứa), `FillRate(TotalDemand=0) = None`, KLA Sphere 5D (`48.73 → 0.000000`), KLA–LS khả thi & cải thiện nghiệm. |
| Xử lý dữ liệu & EDA (`src/prepare_data.py`) | ✅ PASS | 12 chuỗi × 49 ngày (`FOODS_3_090`, `FOODS_3_586`, `FOODS_3_252`), `NaN = 0`, `âm = 0`, tách cứng 42/7 ngày, xuất đủ `results/eda_01..03.png` và `results/eda_report.md`. |
| Thực nghiệm (`src/run_experiments.py`) | ✅ PASS | Chạy lại ra số liệu khớp 100%, `results/summary.csv` đủ **36 dòng**, đủ tất cả cột chuẩn STEP 7.4, `feasible = True` 100%, xuất `results/experiment_log.md` & 3 biểu đồ PNG. |
| Rà soát tham chiếu lỗi / file thừa | ✅ SẠCH | 0 tham chiếu sót tới `download_data.py` hay file `.txt` cũ; đã xóa file zip 16.3 MB trong `docs/`; chỉ giữ `requirements.txt`, `report/note.txt`, `video/note.txt`. |
| Độ dài tóm tắt đăng ký | ✅ ĐẠT | `docs/tom_tat_y_tuong_dang_ky.md`: **280 từ** (`<= 300`); tóm tắt trong `docs/Form đăng ký Team Chuyển Kho.pdf`: **~222 từ** (`<= 300`). |

---

## 2. Đối chiếu từng STEP trong tài liệu hướng dẫn

| Bước | Nội dung | Trạng thái | Ghi chú |
| :--- | :--- | :---: | :--- |
| **STEP 0** | Tổ chức nhóm, thư mục, bảng phân công | ⚠️ Gần xong | Đã có đủ thư mục và danh sách 4 thành viên trong `docs/Form đăng ký Team Chuyển Kho.pdf`; cần điền tên 4 thành viên vào `docs/thanh_vien_va_phan_cong.md`. |
| **STEP 1** | Dữ liệu M5 thật + từ điển dữ liệu | ✅ Hoàn thành | `data/m5_selected.csv`, `data/data_dictionary.md`. |
| **STEP 2** | EDA + dự báo không rò rỉ dữ liệu | ✅ Hoàn thành | `results/eda_01..03.png`, `data/demand_forecast.csv`, `data/demand_actual_test.csv`. |
| **STEP 3** | Kịch bản tồn kho, chi phí, 3 kịch bản cầu | ✅ Hoàn thành | `data/inventory_scenario.csv`, `data/transfer_cost.csv`, `data/scenarios.csv`, `docs/assumptions.md`. |
| **STEP 4** | Bộ mô phỏng + Greedy + 3 case biên | ✅ Hoàn thành | `src/simulator.py`, `src/greedy.py`, `results/baseline_summary.csv`, `results/transfers_greedy.csv`. |
| **STEP 5** | App Demo Streamlit 1 trang | ✅ Hoàn thành | `app/app.py` đủ KPI, bảng tồn kho trước–sau, bảng lệnh chuyển, biểu đồ so sánh, chế độ Precomputed & Live. *(Nhớ chụp 3–5 ảnh màn hình đưa vào báo cáo).* |
| **STEP 6** | Cài KLA & Local Search | ✅ Hoàn thành | `src/kla.py`, `src/local_search.py`, `results/kla_summary.csv`, `results/kla_ls_summary.csv`. |
| **STEP 7** | Thực nghiệm 3 kịch bản × 4 phương pháp × 5 seeds | ✅ Hoàn thành | Đủ CSV, log chạy và biểu đồ `compare_cost.png`, `compare_service.png`, `convergence_kla.png`. |
| **STEP 8** | Phiếu đăng ký đội (hạn < 02/12) | ⚠️ Cần rà soát lỗi gõ | Đã có `docs/Form đăng ký Team Chuyển Kho.pdf` có chữ ký đội trưởng, cần kiểm tra 3 chi tiết ở Mục 3 trước khi chốt nộp. |
| **STEP 9** | Báo cáo PDF mẫu BTC & Video < 5 phút | ❌ Chưa làm | `report/` và `video/` hiện mới có `note.txt`. Chưa có `report/bao_cao_so_khao.pdf` và `video/demo_so_khao.mp4`. |
| **STEP 10** | Kiểm tra độc lập & nộp trước 20:00 ngày 06/12 | ⏳ Chưa tới | Thực hiện sau khi hoàn tất STEP 9. |

---

## 3. Phát hiện cần kiểm tra trong `docs/Form đăng ký Team Chuyển Kho.pdf`

Theo yêu cầu **STEP 8.4** (*"Kiểm tra từng MSSV, lớp, email, số điện thoại"*), cần kiểm tra lại 3 chi tiết sau (nếu chưa nộp hoặc còn sửa được):

1. **Lỗi dấu họ thành viên số 04:** Đang ghi `[Nguyên Hoàng Khiêm]` — thiếu dấu ngã (`Nguyễn Hoàng Khiêm`).
2. **Kiểm tra lại MSSV đội trưởng (01):** Võ Tấn Đạt đang ghi `2045250050`, trong khi 3 thành viên cùng lớp `15DHKHDL03` đều bắt đầu bằng `204524xxxx` (`2045240253`, `2045240270`, `2045240129`). Cần đối chiếu xem có gõ nhầm `24 → 25` hay không.
3. **Trình bày phiếu:** Các trường thông tin còn dấu ngoặc vuông mẫu (`[Võ Tấn Đạt]`, `[15DHKHDL03]`...) và phần chữ ký Đội trưởng bị rớt sang trang 2 riêng lẻ — nên xóa dấu `[...]` và thu gọn nhẹ khoảng cách dòng để gói gọn trong 1 trang A4.

---

## 4. Checklist 10 tiêu chí nghiệm thu tối thiểu (Mục D)

| # | Câu hỏi nghiệm thu | Trạng thái |
| :-: | :--- | :---: |
| 1 | Dữ liệu thật ở đâu? (`data/raw/` & `data/m5_selected.csv`) | ✅ Đạt |
| 2 | Dữ liệu giả lập là gì? (`docs/assumptions.md`) | ✅ Đạt |
| 3 | Có data leakage không? (Tách cứng 42/7 ngày, không dùng test khi tối ưu) | ✅ Đạt |
| 4 | Bộ mô phỏng chạy đúng không? (Bảo toàn hàng hóa, chuyển đối xứng, không bán vượt tồn) | ✅ Đạt |
| 5 | Greedy có khả thi không? (Pass 3 case biên, trả về rỗng khi không cần chuyển) | ✅ Đạt |
| 6 | KLA–LS có thật sự được cài không? (Có mã nguồn + log chạy 5 seeds) | ✅ Đạt |
| 7 | Kết quả có trung thực không? (Mọi số liệu truy xuất từ CSV/log cùng nguồn) | ✅ Đạt |
| 8 | Có sản phẩm xem được không? (`app/app.py` chạy Streamlit hoàn chỉnh) | ✅ Đạt |
| 9 | Hồ sơ đủ chưa? (`report/bao_cao_so_khao.pdf` & `video/demo_so_khao.mp4`) | ❌ Chưa xong |
| 10 | Nộp đúng giờ chưa? (Đăng ký < 02/12; nộp hồ sơ < 23:59 ngày 06/12/2026) | ⏳ Chưa tới |

---

## 5. Việc cần hoàn thành tiếp theo

- [ ] Rà soát & sửa 3 chi tiết trong `docs/Form đăng ký Team Chuyển Kho.pdf` (nếu chưa nộp), nộp đăng ký và lưu bằng chứng xác nhận **trước 02/12/2026** (khuyến nghị trước 30/11).
- [ ] Điền thông tin 4 thành viên vào `docs/thanh_vien_va_phan_cong.md`.
- [ ] Soạn và xuất `report/bao_cao_so_khao.pdf` theo đúng mẫu BTC (chèn ảnh EDA, biểu đồ so sánh, ảnh chụp Streamlit, đủ 4 tài liệu tham khảo Mục F và khai báo vai trò AI theo Điều 8 Mục E).
- [ ] Quay và xuất `video/demo_so_khao.mp4` (< 5 phút, bám sát `docs/kich_ban_video_va_qa.md`).
- [ ] Kiểm tra chéo số liệu, chạy thử trên máy sạch và nộp hồ sơ **trước 20:00 ngày 06/12/2026**.
