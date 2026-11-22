@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

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

:MENU
cls
echo ========================================================================
echo   Team Chuyển Kho - HUIT Data Science Challenge 2026
echo   Menu điều khiển chạy dự án (chọn bước bạn muốn thực hiện)
echo ========================================================================
echo.
echo   --- Kiểm tra nhanh môi trường Python và thư viện ---
%PY_CMD% -c "import re, sys, importlib.metadata as m; from pathlib import Path; vt=lambda s: tuple(int(x) for x in re.findall(r'\d+', s)[:3]); print(f'  Python: {sys.version.split()[0]} ({sys.executable})'); lines=[l.strip() for l in Path('requirements.txt').read_text(encoding='utf-8').splitlines() if l.strip() and not l.strip().startswith('#')]; out=[]; [out.append((p, req, m.version(p) if p in {d.metadata['Name'].lower() for d in m.distributions() if d.metadata['Name']} else None)) for line in lines for p, req in [(re.split(r'>=|==|<|>', line)[0].strip().lower(), line.split('>=')[1].strip() if '>=' in line else '')]]; [print(f'  - {p:<11}: ' + (f'{v:<8} (Đạt >= {req})' if (v and (not req or vt(v)>=vt(req))) else (f'{v:<8} (Cũ, cần >= {req})' if v else f'Chưa cài (cần >= {req})'))) for p, req, v in out]"
if not %errorlevel%==0 (
    echo   [Cảnh báo] Không tìm thấy Python hợp lệ trên máy.
)
echo.
echo   --- Danh sách các bước ---
echo   [1] Bước 1: Kiểm tra và cài đặt / nâng cấp thư viện (requirements.txt)
echo   [2] Bước 2: Kiểm thử logic thuật toán (src/verify_pipeline.py)
echo   [3] Bước 3: Xử lý dữ liệu M5, vẽ EDA và tạo kịch bản (src/prepare_data.py)
echo   [4] Bước 4: Chạy thực nghiệm 3 kịch bản x 4 thuật toán (src/run_experiments.py)
echo   [5] Bước 5: Mở ứng dụng Demo Streamlit (app/app.py)
echo   ----------------------------------------------------------------------
echo   [6] Chạy tự động liên hoàn: Bước 2 -^> Bước 3 -^> Bước 4 -^> Mở App
echo   [0] Thoát
echo.
echo ========================================================================
set "chon="
set /p "chon=Nhập lựa chọn của bạn (0-6) rồi nhấn Enter: "
if defined chon set "chon=%chon: =%"

if "%chon%"=="1" goto RUN1
if "%chon%"=="2" goto RUN2
if "%chon%"=="3" goto RUN3
if "%chon%"=="4" goto RUN4
if "%chon%"=="5" goto RUN5
if "%chon%"=="6" goto RUNALL
if "%chon%"=="0" goto END

echo.
echo [Cảnh báo] Lựa chọn không hợp lệ. Vui lòng nhập số từ 0 đến 6.
pause
goto MENU

:RUN1
echo.
call "%~dp01_cai_dat_thu_vien.bat"
goto MENU

:RUN2
echo.
call "%~dp02_kiem_thu_logic.bat"
goto MENU

:RUN3
echo.
call "%~dp03_xu_ly_du_lieu_va_eda.bat"
goto MENU

:RUN4
echo.
call "%~dp04_chay_thuc_nghiem.bat"
goto MENU

:RUN5
echo.
call "%~dp05_mo_app_streamlit.bat"
goto MENU

:RUNALL
echo.
echo ========================================================================
echo   Chạy tự động liên hoàn: Bước 2 -^> Bước 3 -^> Bước 4 -^> Bước 5
echo ========================================================================

echo.
echo [1/4] Đang kiểm thử logic thuật toán...
%PY_CMD% src/verify_pipeline.py
if not %errorlevel%==0 (
    echo [Lỗi] Dừng lại do Bước 2 thất bại.
    pause
    goto MENU
)

echo.
echo [2/4] Đang xử lý dữ liệu M5, vẽ EDA và tạo kịch bản...
%PY_CMD% src/prepare_data.py
if not %errorlevel%==0 (
    echo [Lỗi] Dừng lại do Bước 3 thất bại.
    pause
    goto MENU
)

echo.
echo [3/4] Đang chạy thực nghiệm 3 kịch bản x 4 thuật toán...
%PY_CMD% src/run_experiments.py
if not %errorlevel%==0 (
    echo [Lỗi] Dừng lại do Bước 4 thất bại.
    pause
    goto MENU
)

echo.
echo [4/4] Đang mở ứng dụng Demo Streamlit...
%PY_CMD% -m streamlit run app/app.py
pause
goto MENU

:END
echo.
echo Hoàn Thành.
exit /b 0
