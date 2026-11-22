@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo ========================================================================
echo   Bước 2: Kiểm thử logic thuật toán (src/verify_pipeline.py)
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

%PY_CMD% src/verify_pipeline.py

echo.
if %errorlevel%==0 (
    echo [Hoàn tất] Kiểm thử logic đạt yêu cầu. Bạn có thể bấm tiếp: 3_xu_ly_du_lieu_va_eda.bat
) else (
    echo [Lỗi] Kiểm thử logic thất bại - mã lỗi: %errorlevel%.
)
echo.
pause
