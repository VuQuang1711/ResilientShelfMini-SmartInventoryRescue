# Kịch bản Video Sơ khảo (~3 phút 45 giây < 5 phút) & Bộ 7 câu hỏi Q&A (STEP 9–10)

**Đội thi:** Team Chuyển Kho — HUIT Data Science Challenge 2026  
*Mọi số liệu dưới đây trích xuất trực tiếp từ `results/eda_report.md` và `results/summary.csv`.*

---

## Phần 1: Kịch bản Video giới thiệu giải pháp (Tổng ~3 phút 45 giây)

### `[00:00 – 00:30 | 30 giây]` Vấn đề & Người dùng
- Giới thiệu đội **Team Chuyển Kho** và tên đề tài.
- Nêu bài toán thực tế của quản lý chuỗi bán lẻ: cùng 1 mã hàng nhưng cửa hàng `CA_1` thiếu hàng trầm trọng (mất doanh thu) trong khi `CA_2`, `CA_3` lại dư hàng vượt ngưỡng an toàn. Điều chuyển tồn kho nội bộ giúp giải quyết bài toán này nhanh hơn đặt hàng mới từ nhà cung cấp.

### `[00:30 – 01:20 | 50 giây]` Dữ liệu & EDA *(hiển thị `eda_01.png`, `eda_02.png`, `eda_03.png`)*
- **Nguồn dữ liệu thật:** Kaggle M5 Forecasting – Accuracy (`sales_train_validation.csv`).
- **Phạm vi MVP:** 04 cửa hàng (`CA_1..CA_4`) × 03 sản phẩm bán chạy (`FOODS_3_090`, `FOODS_3_586`, `FOODS_3_252`) × 49 ngày cuối (`d_1865..d_1913`), tổng 588 quan sát, 0 giá trị thiếu, 0 giá trị âm.
- **Minh bạch chống rò rỉ dữ liệu:** 42 ngày đầu làm lịch sử & dự báo (trung bình 7 ngày cuối lịch sử); 7 ngày cuối giữ riêng để kiểm tra.
- **Công bố trung thực:** M5 là doanh số (không phải tồn kho thực), nên tồn đầu kỳ (`CA_1 ~50%`, `CA_2/CA_3 ~160%`, `CA_4 ~110%`), sức chứa (`>= 200%`), ngưỡng an toàn (`100%`) và chi phí chuyển (`1.0–3.5 USD/đv`) được mô phỏng minh bạch; giả định MVP chuyển ngay đầu kỳ (chưa xét độ trễ giao hàng).

### `[01:20 – 02:20 | 60 giây]` Phương pháp & Mô hình tối ưu *(hiển thị `convergence_kla.png`)*
- Biến quyết định nguyên không âm $x_{ijk}$; hàm mục tiêu thống nhất:
  $$F(X) = C_{\text{move}}(X) + 10 \times \text{Shortage}(X) + 0.5 \times \text{Leftover}(X)$$
- **Ràng buộc khả thi:** không tự chuyển, không xuất quá dư an toàn, không nhận quá sức chứa, bảo toàn tổng tồn kho.
- **Đối chứng 4 phương pháp:** `No Transfer`, `Greedy`, `KLA` (Ghasemi et al., *AI Review* 2025, không tham số điều khiển phức tạp) và thuật toán lai `KLA–LS` (kết hợp 3 phép tìm kiếm cục bộ: chỉnh số lượng, đổi nơi gửi, đổi nơi nhận).
- Chỉ vào đồ thị `convergence_kla.png`: `KLA–LS` kéo hàm mục tiêu xuống rất nhanh ngay vòng 10 và đạt phương sai `0` qua 5 seeds.

### `[02:20 – 03:20 | 60 giây]` Kết quả thực nghiệm & Demo Streamlit trực tiếp
- Hiển thị `compare_cost.png` và `compare_service.png`:
  - **Kịch bản `NORMAL`:** chi phí giảm `5,532.5 -> 1,946.0 USD` (`-64.8%`), thiếu hàng giảm `471 -> 67` đơn vị (`-85.8%`), Fill Rate tăng `87.4% -> 98.2%`.
  - **Kịch bản `FLASH_SALE (+30%)`:** chi phí giảm `6,675.5 -> 2,415.0 USD` (`-63.8%`), Fill Rate tăng `84.8% -> 97.6%`.
  - **Kịch bản `SURGE (+50%)`:** chi phí giảm `7,437.5 -> 2,585.0 USD` (`-65.2%`), Fill Rate tăng `83.2% -> 97.6%`.
- Mở trực tiếp app Streamlit (`py -m streamlit run app/app.py`):
  - Chuyển kịch bản `NORMAL -> FLASH_SALE -> SURGE`.
  - Cho thấy bảng tồn kho trước–sau, bảng lệnh gợi ý (ví dụ `FLASH_SALE`: chuyển 215 đv `FOODS_3_090` từ `CA_2` và 82 đv từ `CA_3` về `CA_1`).
  - Chuyển thử chế độ **"Chạy tối ưu trực tiếp (Live)"** chạy xong trong `< 1 giây`.

### `[03:20 – 03:45 | 25 giây]` Kết luận & Hướng phát triển
- Tổng kết giá trị thực tế và tính trung thực khoa học (với MVP tuyến tính từng SKU độc lập, `KLA–LS` đạt nghiệm tối ưu bằng `Greedy` và ổn định hơn `KLA` thuần).
- **Hướng phát triển chung kết:** tối ưu nhiều kỳ (*multi-period*), xét độ trễ giao hàng (*lead time*), ràng buộc chung tải trọng xe/kho và bất định nhu cầu đa kịch bản — nơi `KLA–LS` phát huy ưu thế vượt trội so với `Greedy`.

---

## Phần 2: 7 Câu hỏi Phản biện Vòng Sơ khảo & Gợi ý Trả lời ngắn (STEP 10.6)

1. **Dữ liệu thật lấy từ đâu?**
   > Lấy từ bộ công khai **Kaggle M5 Forecasting – Accuracy** (`sales_train_validation.csv`), lọc 4 cửa hàng `CA_1..CA_4` và 3 SKU (`FOODS_3_090`, `FOODS_3_586`, `FOODS_3_252`) trong 49 ngày cuối (`d_1865..d_1913`).

2. **Những dữ liệu nào là giả lập/mô phỏng?**
   > Tồn kho đầu kỳ (`50%/160%/160%/110%` dự báo), ngưỡng an toàn (`100%`), sức chứa (`>= 200%`), ma trận chi phí chuyển (`1.0–3.5 USD/đv`) và hệ số phạt thiếu/dư (`10` và `0.5 USD/đv`). Tất cả được công bố rõ quy tắc trong `docs/assumptions.md` và không dùng dữ liệu tập test để tạo.

3. **Làm sao đảm bảo không bị rò rỉ dữ liệu (data leakage)?**
   > 49 ngày được cắt cứng: 42 ngày đầu làm lịch sử để tính dự báo 7 ngày và sinh kịch bản tồn kho; 7 ngày cuối giữ riêng trong `demand_actual_test.csv`. Toàn bộ thuật toán tối ưu chỉ nhìn thấy `demand_forecast.csv`, sau đó mới đưa nghiệm vào mô phỏng trên 7 ngày kiểm tra thực tế.

4. **Vì sao dùng Greedy làm baseline và vì sao KLA–LS có kết quả bằng Greedy ở MVP?**
   > Greedy rất nhanh (`< 0.1 ms`) và minh bạch, là mốc đối chứng bắt buộc. Trong phạm vi MVP 1 kỳ, hàm mục tiêu tuyến tính từng khúc và các SKU tách rời nên Greedy chạm đúng nghiệm tối ưu toàn cục trên dự báo. `KLA–LS` đạt chính xác cùng nghiệm tối ưu toàn cục đó (`std = 0` trên cả 5 seeds) và vượt rõ `KLA` thuần. Ở vòng sau khi thêm ràng buộc tải trọng xe chung, chi phí cố định mỗi chuyến và nhiều kỳ, Greedy sẽ dễ rơi vào tối ưu cục bộ còn `KLA–LS` sẽ phát huy lợi thế.

5. **Thuật toán KLA hoạt động thế nào và đảm bảo nghiệm hợp lệ ra sao?**
   > KLA (Ghasemi et al., *AI Review* 2025) mô phỏng định luật dòng điện Kirchhoff: mỗi cá thể tương tác với 3 cá thể khác theo hướng chênh lệch hàm mục tiêu (hút về nghiệm tốt, đẩy khỏi nghiệm kém) mà không cần nhiều siêu tham số. Sau khi cập nhật vị trí liên tục, nghiệm đi qua toán tử sửa nghiệm (*repair*) để làm tròn nguyên, triệt tiêu chuyển hai chiều, cắt xuất không vượt dư an toàn và không vượt sức chứa; cuối cùng Local Search tinh chỉnh lượng chuyển và đổi cửa hàng gửi/nhận.

6. **Chỉ số đo lường chính là gì? Trường hợp `TotalDemand = 0` xử lý sao?**
   > Đánh giá đồng thời 3 KPI trên tập test 7 ngày: Tổng chi phí $F(X)$ (USD), Tổng lượng thiếu hàng (đơn vị) và Fill Rate (%). Khi `TotalDemand = 0`, hệ thống trả về `None` / `"N/A (không xác định)"` chứ không tự động gán `100%`.

7. **Giới hạn nghiên cứu hiện tại là gì?**
   > (i) Dùng doanh số M5 làm đại diện nhu cầu (có thể bị *censored demand* khi hết hàng); (ii) mới xét bài toán 1 kỳ đầu tuần và giả định `lead time = 0`; (iii) chưa xét chi phí cố định theo chuyến xe và sức chứa dùng chung giữa các SKU.
