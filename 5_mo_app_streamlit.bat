@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo ========================================================================
echo   Bước 5: Khởi chạy ứng dụng Demo Streamlit (app/app.py)
echo ========================================================================
echo   Trình duyệt sẽ tự động mở. Nhấn Ctrl + C hoặc đóng cửa sổ này để tắt App.
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

%PY_CMD% -m streamlit run app/app.py

echo.
pause
