@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo ========================================================================
echo   Bước 3: Xử lý dữ liệu M5, vẽ EDA và tạo kịch bản (src/prepare_data.py)
echo ========================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    set "PY_CMD=.venv\Scripts\python.exe"
) else (
    where py >nul 2>nul
    if %errorlevel%==0 (
        set "PY_CMD=py"
    ) else (
        set "PY_CMD=python"
    )
)

%PY_CMD% src/prepare_data.py

echo.
if %errorlevel%==0 (
    echo [Hoàn tất] Đã xử lý dữ liệu và tạo kịch bản xong. Bạn có thể bấm tiếp: 4_chay_thuc_nghiem.bat
) else (
    echo [Lỗi] Xử lý dữ liệu thất bại - mã lỗi: %errorlevel%.
    echo       Kiểm tra lại file data/raw/sales_train_validation.csv đã tồn tại chưa.
)
echo.
pause
