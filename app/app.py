"""
app.py
------
Ứng dụng Demo 1 trang bằng Streamlit cho Team Chuyển Kho (STEP 5).
Chạy bằng lệnh: py -m streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from greedy import solve_greedy
from kla import solve_kla
from local_search import solve_kla_ls
from simulator import (
    load_problem_instance,
    no_transfer_plan,
    simulate_7_days,
    transfers_to_dataframe,
)

st.set_page_config(
    page_title="Team Chuyển Kho — Tối ưu điều chuyển tồn kho",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Team Chuyển Kho — Tối ưu điều chuyển tồn kho đa cửa hàng")
st.caption(
    "HUIT Data Science Challenge 2026 | Thuật toán lai Kirchhoff's Law Algorithm & Tìm kiếm cục bộ (KLA–LS)"
)

with st.expander("ℹ️ Nguồn dữ liệu & Lưu ý thông số mô phỏng (Bắt buộc đọc)", expanded=False):
    st.markdown(
        """
- **Dữ liệu doanh số thật:** Trích xuất từ bộ dữ liệu công khai **Kaggle M5 Forecasting – Accuracy** (`sales_train_validation.csv`), phạm vi **04 cửa hàng** (`CA_1`, `CA_2`, `CA_3`, `CA_4`) × **03 sản phẩm** × **49 ngày cuối** (42 ngày lịch sử để dự báo, 07 ngày cuối giữ riêng làm tập kiểm tra — không rò rỉ dữ liệu).
- **Dữ liệu mô phỏng (giả định minh bạch):** Do M5 là dữ liệu doanh số chứ không phải tồn kho thực, tồn kho đầu kỳ (`CA_1 ≈ 50%`, `CA_2/CA_3 ≈ 160%`, `CA_4 ≈ 110%` dự báo), ngưỡng an toàn (`100%`), sức chứa (`>= 200%`), chi phí điều chuyển (`1.0–3.5 USD/đv`), phạt thiếu hàng (`10 USD/đv`) và chi phí tồn dư (`0.5 USD/đv`) được mô phỏng theo quy tắc công bố trong `docs/assumptions.md`.
- **Giả định MVP:** Điều chuyển thực hiện ngay đầu kỳ trước ngày thứ nhất của chu kỳ 7 ngày (**chưa xét độ trễ vận chuyển**).
        """
    )

# Kiểm tra dữ liệu đã sẵn sàng chưa
required_data = [
    DATA_DIR / "inventory_scenario.csv",
    DATA_DIR / "transfer_cost.csv",
    DATA_DIR / "scenarios.csv",
    DATA_DIR / "demand_actual_test.csv",
]
missing_data = [p.name for p in required_data if not p.exists() or p.stat().st_size == 0]

if missing_data:
    st.warning(
        "Chưa có dữ liệu M5 đã xử lý để hiển thị. Thiếu các file trong `data/`: "
        + ", ".join(missing_data)
    )
    st.markdown(
        """
### Các bước chuẩn bị dữ liệu nhanh:
1. Tải `sales_train_validation.csv` từ [Kaggle M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) và đặt vào `data/raw/sales_train_validation.csv`.
2. Chạy lệnh xử lý dữ liệu & EDA: `py src/prepare_data.py`
3. Chạy lệnh thực nghiệm: `py src/run_experiments.py`
4. Tải lại trang Streamlit này!
        """
    )
    st.stop()

# Sidebar điều khiển
st.sidebar.header("⚙️ Bảng điều khiển")
scenarios_df = pd.read_csv(DATA_DIR / "scenarios.csv")
available_scenarios = scenarios_df["scenario"].unique().tolist()
selected_scenario = st.sidebar.selectbox("1. Chọn kịch bản nhu cầu", available_scenarios, index=0)

scen_desc = scenarios_df[scenarios_df["scenario"] == selected_scenario]["description"].iloc[0]
st.sidebar.info(f"**Mô tả kịch bản:** {scen_desc}")

selected_method = st.sidebar.selectbox(
    "2. Chọn phương pháp xem chi tiết",
    ["KLA-LS", "KLA", "Greedy", "No Transfer"],
    index=0,
)

run_mode = st.sidebar.radio(
    "3. Chế độ dữ liệu kết quả",
    ["Dùng kết quả đã tính trước (results/)", "Chạy tối ưu trực tiếp (Live)"],
    index=0,
)

pop_size = 15
max_iter = 35
seed_val = 42
if run_mode == "Chạy tối ưu trực tiếp (Live)":
    st.sidebar.subheader("Tham số chạy trực tiếp KLA / KLA–LS")
    pop_size = st.sidebar.slider("Kích thước quần thể (pop_size)", 8, 30, 15)
    max_iter = st.sidebar.slider("Số vòng lặp (max_iter)", 10, 60, 35)
    seed_val = st.sidebar.number_input("Seed lặp lại", min_value=0, max_value=9999, value=42, step=1)

# Nạp bài toán cho kịch bản đã chọn
instance = load_problem_instance(DATA_DIR, selected_scenario)
summary_path = RESULTS_DIR / "summary.csv"

# Thực hiện tính toán hoặc nạp kết quả đã tính trước
if run_mode == "Chạy tối ưu trực tiếp (Live)" or not summary_path.exists() or summary_path.stat().st_size == 0:
    if run_mode != "Chạy tối ưu trực tiếp (Live)":
        st.info("Chưa tìm thấy `results/summary.csv` nên hệ thống đang tự chạy trực tiếp cho kịch bản hiện tại.")

    x_none = no_transfer_plan(instance)
    x_greedy, info_g = solve_greedy(instance)
    x_kla, info_k = solve_kla(instance, pop_size=pop_size, max_iter=max_iter, seed=int(seed_val))
    x_klals, info_kl = solve_kla_ls(instance, pop_size=pop_size, max_iter=max_iter, seed=int(seed_val), ls_every=10)

    plans = {
        "No Transfer": x_none,
        "Greedy": x_greedy,
        "KLA": x_kla,
        "KLA-LS": x_klals,
    }
    comp_rows = []
    for m_name, x_plan in plans.items():
        sim_res = simulate_7_days(x_plan, instance, use_actual=True)
        mt = sim_res["metrics"]
        comp_rows.append(
            {
                "method": m_name,
                "total_cost": round(float(mt["total_cost"]), 2),
                "transfer_cost": round(float(mt["transfer_cost"]), 2),
                "shortage_cost": round(float(mt["shortage_cost"]), 2),
                "holding_cost": round(float(mt["holding_cost"]), 2),
                "unmet_demand": round(float(mt["unmet_demand"]), 2),
                "fill_rate": None if mt["fill_rate"] is None else round(float(mt["fill_rate"]), 2),
                "feasible": sim_res["feasible"],
            }
        )
    comp_df = pd.DataFrame(comp_rows)
    chosen_x = plans[selected_method]
    sim_chosen = simulate_7_days(chosen_x, instance, use_actual=True)
    sim_none = simulate_7_days(x_none, instance, use_actual=True)
    transfers_df = transfers_to_dataframe(chosen_x, instance, selected_method)
    mode_badge = f"Chế độ: **Tính trực tiếp (Live)** | seed={seed_val}, pop={pop_size}, iter={max_iter}"
else:
    full_summary = pd.read_csv(summary_path)
    scen_summary = full_summary[full_summary["scenario"] == selected_scenario]
    comp_df = (
        scen_summary.groupby("method", as_index=False)
        .agg(
            total_cost=("total_cost", "mean"),
            transfer_cost=("transfer_cost", "mean"),
            shortage_cost=("shortage_cost", "mean"),
            holding_cost=("holding_cost", "mean"),
            unmet_demand=("unmet_demand", "mean"),
            fill_rate=("fill_rate", "mean"),
            feasible=("feasible", "all"),
        )
    )
    order_map = {"No Transfer": 0, "Greedy": 1, "KLA": 2, "KLA-LS": 3}
    comp_df["_ord"] = comp_df["method"].map(order_map).fillna(99)
    comp_df = comp_df.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)

    # Khôi phục ma trận lệnh điều chuyển đã tính trước cho phương pháp đang chọn
    x_none = no_transfer_plan(instance)
    file_map = {
        "Greedy": RESULTS_DIR / "transfers_greedy.csv",
        "KLA": RESULTS_DIR / "transfers_kla.csv",
        "KLA-LS": RESULTS_DIR / "transfers_kla_ls.csv",
    }
    chosen_x = no_transfer_plan(instance)
    transfers_df = transfers_to_dataframe(chosen_x, instance, selected_method)
    if selected_method in file_map and file_map[selected_method].exists():
        tdf_all = pd.read_csv(file_map[selected_method])
        transfers_df = tdf_all[tdf_all["scenario"] == selected_scenario].copy()
        s_idx = {s: i for i, s in enumerate(instance.stores)}
        k_idx = {k: j for j, k in enumerate(instance.items)}
        for _, r in transfers_df.iterrows():
            chosen_x[s_idx[r["from_store"]], s_idx[r["to_store"]], k_idx[r["item_id"]]] = int(r["quantity"])

    sim_chosen = simulate_7_days(chosen_x, instance, use_actual=True)
    sim_none = simulate_7_days(x_none, instance, use_actual=True)
    mode_badge = "Chế độ: **Kịch bản đã tính trước từ `results/summary.csv`** (biểu đồ so sánh hiển thị trung bình 5 seeds; 4 thẻ KPI & bảng lệnh hiển thị nghiệm tốt nhất theo hàm mục tiêu dự báo)"

st.markdown(mode_badge)

# --- 3 KPI CARDS ---
m_sel = sim_chosen["metrics"]
m_base = sim_none["metrics"]

cost_delta = m_sel["total_cost"] - m_base["total_cost"]
cost_pct = (100.0 * cost_delta / m_base["total_cost"]) if m_base["total_cost"] > 0 else 0.0
unmet_delta = m_sel["unmet_demand"] - m_base["unmet_demand"]
fr_sel = m_sel["fill_rate"]
fr_base = m_base["fill_rate"]
fr_delta = None if (fr_sel is None or fr_base is None) else (fr_sel - fr_base)

kpi_suffix = (
    " — Nghiệm tốt nhất"
    if (run_mode != "Chạy tối ưu trực tiếp (Live)" and selected_method in ("KLA", "KLA-LS"))
    else ""
)
st.subheader(f"📊 Kết quả KPI trên tập kiểm tra 7 ngày — {selected_method} ({selected_scenario}){kpi_suffix}")
if run_mode != "Chạy tối ưu trực tiếp (Live)" and selected_method in ("KLA", "KLA-LS"):
    mean_row = comp_df[comp_df["method"] == selected_method]
    if not mean_row.empty:
        mean_cost_5s = float(mean_row["total_cost"].iloc[0])
        if abs(mean_cost_5s - float(m_sel["total_cost"])) > 0.01:
            st.caption(
                f"Lưu ý: 4 thẻ KPI bên dưới phản ánh phương án tốt nhất trong bảng lệnh điều chuyển "
                f"({m_sel['total_cost']:,.2f} USD); trung bình 5 seeds trên biểu đồ so sánh là {mean_cost_5s:,.2f} USD."
            )
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Tổng chi phí (USD)",
    f"{m_sel['total_cost']:,.2f} $",
    f"{cost_delta:+,.2f} $ ({cost_pct:+.1f}%)" if selected_method != "No Transfer" else "Mốc đối chứng",
    delta_color="inverse",
)
c2.metric(
    "Tổng lượng thiếu hàng (đơn vị)",
    f"{m_sel['unmet_demand']:,.1f}",
    f"{unmet_delta:+,.1f} đv" if selected_method != "No Transfer" else "Mốc đối chứng",
    delta_color="inverse",
)
c3.metric(
    "Tỷ lệ đáp ứng (Fill Rate)",
    "N/A" if fr_sel is None else f"{fr_sel:.2f}%",
    None if (fr_delta is None or selected_method == "No Transfer") else f"{fr_delta:+.2f}%",
)
c4.metric(
    "Tổng hàng điều chuyển & Khả thi",
    f"{m_sel['total_units_moved']} đv",
    "Khả thi 100%" if sim_chosen["feasible"] else "KHÔNG KHẢ THI",
)

st.divider()

# --- BIỂU ĐỒ SO SÁNH PHƯƠNG PHÁP ---
st.subheader("📈 So sánh các phương pháp điều chuyển")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    cost_long = comp_df.melt(
        id_vars="method",
        value_vars=["transfer_cost", "shortage_cost", "holding_cost"],
        var_name="Thành phần chi phí",
        value_name="Chi phí (USD)",
    )
    rename_map = {
        "transfer_cost": "Chi phí chuyển hàng",
        "shortage_cost": "Chi phí phạt thiếu hàng",
        "holding_cost": "Chi phí tồn dư cuối kỳ",
    }
    cost_long["Thành phần chi phí"] = cost_long["Thành phần chi phí"].map(rename_map)
    fig_cost = px.bar(
        cost_long,
        x="method",
        y="Chi phí (USD)",
        color="Thành phần chi phí",
        barmode="stack",
        title=f"Cơ cấu tổng chi phí theo phương pháp ({selected_scenario})",
        labels={"method": "Phương pháp"},
    )
    st.plotly_chart(fig_cost, width="stretch")

with col_chart2:
    fig_fr = px.bar(
        comp_df,
        x="method",
        y="fill_rate",
        text_auto=".2f",
        color="method",
        title=f"Tỷ lệ đáp ứng nhu cầu Fill Rate (%) — {selected_scenario}",
        labels={"method": "Phương pháp", "fill_rate": "Fill Rate (%)"},
    )
    fig_fr.update_yaxes(range=[0, 108])
    fig_fr.update_layout(showlegend=False)
    st.plotly_chart(fig_fr, width="stretch")

st.divider()

# --- BẢNG TỒN KHO TRƯỚC-SAU & LỆNH ĐIỀU CHUYỂN ---
col_tbl1, col_tbl2 = st.columns([1.35, 1.0])

with col_tbl1:
    st.subheader("🏪 Bảng tồn kho Trước – Sau điều chuyển (7 ngày kiểm tra)")
    base_tbl = sim_none["store_item_table"][["store_id", "item_id", "unmet_demand"]].rename(
        columns={"unmet_demand": "unmet_before"}
    )
    detail_tbl = sim_chosen["store_item_table"].merge(base_tbl, on=["store_id", "item_id"])
    display_cols = [
        "store_id",
        "item_id",
        "initial_inventory",
        "inflow",
        "outflow",
        "inventory_after",
        "forecast_7d",
        "demand_evaluated",
        "unmet_before",
        "unmet_demand",
        "leftover_inventory",
    ]
    st.dataframe(detail_tbl[display_cols], width="stretch", hide_index=True)

with col_tbl2:
    st.subheader(f"🚚 Gợi ý lệnh điều chuyển ({selected_method})")
    if transfers_df.empty:
        st.info("Không có lệnh điều chuyển nào được đề xuất (phương án rỗng).")
    else:
        st.dataframe(
            transfers_df[["item_id", "from_store", "to_store", "quantity", "unit_cost_usd", "transfer_cost_usd"]],
            width="stretch",
            hide_index=True,
        )

# --- TABS HÌNH ẢNH EDA & HỘI TỤ ---
st.divider()
tab_summary, tab_eda, tab_conv = st.tabs(["📋 Bảng tổng hợp số liệu", "🔍 Biểu đồ EDA dữ liệu M5", "⚡ Hội tụ KLA vs KLA–LS"])

with tab_summary:
    st.dataframe(comp_df, width="stretch", hide_index=True)
    if summary_path.exists() and summary_path.stat().st_size > 0:
        with st.expander("Xem toàn bộ bảng results/summary.csv (tất cả kịch bản & seeds)"):
            st.dataframe(pd.read_csv(summary_path), width="stretch", hide_index=True)

with tab_eda:
    eda_files = [("eda_01.png", "Biến động doanh số 49 ngày"), ("eda_02.png", "Tổng doanh số 42 ngày lịch sử"), ("eda_03.png", "Chênh lệch nhu cầu 7 ngày dự báo")]
    shown = False
    for fname, caption in eda_files:
        fpath = RESULTS_DIR / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            st.image(str(fpath), caption=caption, width="stretch")
            shown = True
    if not shown:
        st.info("Chưa có ảnh EDA. Hãy chạy `py src/prepare_data.py` để sinh `eda_01..03.png`.")

with tab_conv:
    conv_img = RESULTS_DIR / "convergence_kla.png"
    if conv_img.exists() and conv_img.stat().st_size > 0:
        st.image(str(conv_img), caption="Đường hội tụ trung bình 5 seeds của KLA và KLA–LS", width="stretch")
    else:
        st.info("Chưa có biểu đồ hội tụ. Hãy chạy `py src/run_experiments.py` để sinh `convergence_kla.png`.")
