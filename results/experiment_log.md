# Nhật ký Chạy Thực nghiệm — Team Chuyển Kho (HUIT DSC 2026)

- **Thời gian chạy:** `2026-11-22 11:20:01`
- **Kịch bản:** `['NORMAL', 'FLASH_SALE', 'SURGE']`
- **Cấu hình KLA:** `pop_size=15, max_iter=35, seeds=[42, 43, 44, 45, 46]`
- **Cấu hình KLA-LS:** `pop_size=15, max_iter=35, ls_every=10, seeds=[42, 43, 44, 45, 46]`

---

## Bảng tổng hợp kết quả trung bình trên tập test 7 ngày

```text
  scenario      method  mean_total_cost  std_total_cost  mean_unmet  mean_fill_rate  mean_forecast_obj  mean_runtime_sec  all_feasible
    NORMAL No Transfer           5532.5             NaN       471.0        87.39970            5142.00          0.000006          True
    NORMAL      Greedy           1946.0             NaN        67.0        98.20760            1209.00          0.000097          True
    NORMAL         KLA           1972.7       55.166339        66.8        98.21296            1237.80          0.261074          True
    NORMAL      KLA-LS           1946.0        0.000000        67.0        98.20760            1209.00          0.344500          True
FLASH_SALE No Transfer           6675.5             NaN       585.3        84.80650            6255.00          0.000005          True
FLASH_SALE      Greedy           2415.0             NaN        93.3        97.57810            1403.35          0.000083          True
FLASH_SALE         KLA           2430.6       63.324956        85.3        97.78576            1503.58          0.267329          True
FLASH_SALE      KLA-LS           2415.0        0.000000        93.3        97.57810            1403.35          0.390312          True
     SURGE No Transfer           7437.5             NaN       661.5        83.16150            6997.00          0.000004          True
     SURGE      Greedy           2585.0             NaN        95.5        97.56900            1551.25          0.000106          True
     SURGE         KLA           2585.3       37.520994        93.1        97.63012            1578.85          0.278139          True
     SURGE      KLA-LS           2585.0        0.000000        95.5        97.56900            1551.25          0.373509          True
```

## Chi tiết tất cả các lần chạy

```text
  scenario      method  seed  total_cost  transfer_cost  shortage_cost  holding_cost  unmet_demand  leftover_inventory  fill_rate  total_units_moved  forecast_objective  runtime_sec  feasible
    NORMAL No Transfer     0      5532.5            0.0         4710.0         822.5         471.0              1645.0    87.3997                  0             5142.00     0.000006      True
    NORMAL      Greedy     0      1946.0          655.5          670.0         620.5          67.0              1241.0    98.2076                437             1209.00     0.000097      True
    NORMAL         KLA    42      1963.5          704.5          640.0         619.0          64.0              1238.0    98.2879                442             1258.00     0.257453      True
    NORMAL         KLA    43      1961.0          670.5          670.0         620.5          67.0              1241.0    98.2076                446             1224.00     0.277465      True
    NORMAL         KLA    44      1920.5          703.5          600.0         617.0          60.0              1234.0    98.3949                467             1257.00     0.254692      True
    NORMAL         KLA    45      2066.5          681.5          760.0         625.0          76.0              1250.0    97.9668                452             1235.00     0.255080      True
    NORMAL         KLA    46      1952.0          661.5          670.0         620.5          67.0              1241.0    98.2076                441             1215.00     0.260679      True
    NORMAL      KLA-LS    42      1946.0          655.5          670.0         620.5          67.0              1241.0    98.2076                437             1209.00     0.363285      True
    NORMAL      KLA-LS    43      1946.0          655.5          670.0         620.5          67.0              1241.0    98.2076                437             1209.00     0.353085      True
    NORMAL      KLA-LS    44      1946.0          655.5          670.0         620.5          67.0              1241.0    98.2076                437             1209.00     0.334822      True
    NORMAL      KLA-LS    45      1946.0          655.5          670.0         620.5          67.0              1241.0    98.2076                437             1209.00     0.359573      True
    NORMAL      KLA-LS    46      1946.0          655.5          670.0         620.5          67.0              1241.0    98.2076                437             1209.00     0.311737      True
FLASH_SALE No Transfer     0      6675.5            0.0         5853.0         822.5         585.3              1645.0    84.8065                  0             6255.00     0.000005      True
FLASH_SALE      Greedy     0      2415.0          905.5          933.0         576.5          93.3              1153.0    97.5781                549             1403.35     0.000083      True
FLASH_SALE         KLA    42      2502.5         1276.5          663.0         563.0          66.3              1126.0    98.2790                697             1774.35     0.263823      True
FLASH_SALE         KLA    43      2434.5          925.0          933.0         576.5          93.3              1153.0    97.5781                549             1426.00     0.265116      True
FLASH_SALE         KLA    44      2352.5          948.0          833.0         571.5          83.3              1143.0    97.8377                565             1445.85     0.231933      True
FLASH_SALE         KLA    45      2480.5          939.5          963.0         578.0          96.3              1156.0    97.5002                563             1437.35     0.291739      True
FLASH_SALE         KLA    46      2383.0          936.5          873.0         573.5          87.3              1147.0    97.7338                557             1434.35     0.284032      True
FLASH_SALE      KLA-LS    42      2415.0          905.5          933.0         576.5          93.3              1153.0    97.5781                549             1403.35     0.569493      True
FLASH_SALE      KLA-LS    43      2415.0          905.5          933.0         576.5          93.3              1153.0    97.5781                549             1403.35     0.325134      True
FLASH_SALE      KLA-LS    44      2415.0          905.5          933.0         576.5          93.3              1153.0    97.5781                549             1403.35     0.368921      True
FLASH_SALE      KLA-LS    45      2415.0          905.5          933.0         576.5          93.3              1153.0    97.5781                549             1403.35     0.383122      True
FLASH_SALE      KLA-LS    46      2415.0          905.5          933.0         576.5          93.3              1153.0    97.5781                549             1403.35     0.304888      True
     SURGE No Transfer     0      7437.5            0.0         6615.0         822.5         661.5              1645.0    83.1615                  0             6997.00     0.000004      True
     SURGE      Greedy     0      2585.0         1090.5          955.0         539.5          95.5              1079.0    97.5690                623             1551.25     0.000106      True
     SURGE         KLA    42      2570.0         1107.0          925.0         538.0          92.5              1076.0    97.6454                628             1573.00     0.272802      True
     SURGE         KLA    43      2585.5         1122.5          925.0         538.0          92.5              1076.0    97.6454                630             1583.25     0.273542      True
     SURGE         KLA    44      2587.0         1103.0          945.0         539.0          94.5              1078.0    97.5945                629             1569.00     0.258684      True
     SURGE         KLA    45      2643.5         1128.0          975.0         540.5          97.5              1081.0    97.5181                644             1588.75     0.262130      True
     SURGE         KLA    46      2540.5         1119.5          885.0         536.0          88.5              1072.0    97.7472                631             1580.25     0.323535      True
     SURGE      KLA-LS    42      2585.0         1090.5          955.0         539.5          95.5              1079.0    97.5690                623             1551.25     0.468240      True
     SURGE      KLA-LS    43      2585.0         1090.5          955.0         539.5          95.5              1079.0    97.5690                623             1551.25     0.325476      True
     SURGE      KLA-LS    44      2585.0         1090.5          955.0         539.5          95.5              1079.0    97.5690                623             1551.25     0.407357      True
     SURGE      KLA-LS    45      2585.0         1090.5          955.0         539.5          95.5              1079.0    97.5690                623             1551.25     0.355816      True
     SURGE      KLA-LS    46      2585.0         1090.5          955.0         539.5          95.5              1079.0    97.5690                623             1551.25     0.310654      True
```
