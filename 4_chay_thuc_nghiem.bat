@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo ========================================================================
echo   Bước 4: Chạy thực nghiệm 3 kịch bản x 4 thuật toán (src/run_experiments.py)
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

%PY_CMD% src/run_experiments.py

echo.
if %errorlevel%==0 (
    echo [Hoàn tất] Đã chạy thực nghiệm và lưu kết quả xong. Bạn có thể bấm tiếp: 5_mo_app_streamlit.bat
) else (
    echo [Lỗi] Chạy thực nghiệm thất bại - mã lỗi: %errorlevel%.
)
echo.
pause
