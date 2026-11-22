@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo ========================================================================
echo   Bước 1: Kiểm tra phiên bản Python và cài đặt thư viện (requirements.txt)
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

echo [Thông tin] Đang kiểm tra môi trường hiện tại...
%PY_CMD% -c "import re, sys, importlib.metadata as m; from pathlib import Path; vt=lambda s: tuple(int(x) for x in re.findall(r'\d+', s)[:3]); print(f'  Python hiện tại : {sys.version.split()[0]}'); print(f'  Đường dẫn       : {sys.executable}'); print('  Trạng thái thư viện so với requirements.txt:'); need=False; lines=[l.strip() for l in Path('requirements.txt').read_text(encoding='utf-8').splitlines() if l.strip() and not l.strip().startswith('#')]; inst={d.metadata['Name'].lower(): d.version for d in m.distributions() if d.metadata['Name']}; [print(f'    - {p:<11}: ' + (f'{v:<8} [Đạt >= {req}]' if (v and (not req or vt(v)>=vt(req))) else (f'{v:<8} [Phiên bản cũ, cần >= {req}]' if v else f'Chưa cài  [Thiếu, cần >= {req}]'))) for line in lines for p, req in [(re.split(r'>=|==|<|>', line)[0].strip().lower(), line.split('>=')[1].strip() if '>=' in line else '')] for v in [inst.get(p)]]; need=any((inst.get(re.split(r'>=|==|<|>', line)[0].strip().lower()) is None) or ('>=' in line and vt(inst.get(re.split(r'>=|==|<|>', line)[0].strip().lower())) < vt(line.split('>=')[1].strip())) for line in lines); sys.exit(2 if need else 0)"
set "CHECK_CODE=%errorlevel%"

if "%CHECK_CODE%"=="1" (
    echo.
    echo [Lỗi] Không thể chạy Python. Vui lòng kiểm tra đã cài Python 3.10+ chưa.
    pause
    exit /b 1
)

echo.
if "%CHECK_CODE%"=="2" goto ASK_NEED
goto ASK_OK

:ASK_NEED
echo [Thông báo] Phát hiện có thư viện chưa cài hoặc đang ở phiên bản cũ hơn yêu cầu.
echo.
echo   [1] Cài đặt / cập nhật các thư viện bị thiếu hoặc cũ theo requirements.txt
echo   [2] Nâng cấp toàn bộ thư viện lên phiên bản mới nhất
echo   [0] Bỏ qua, không cài đặt lúc này
echo.
set "opt="
set /p "opt=Nhập lựa chọn của bạn (0-2) rồi nhấn Enter: "
if defined opt set "opt=%opt: =%"
if "%opt%"=="1" goto INSTALL_REQ
if "%opt%"=="2" goto UPGRADE_ALL
if "%opt%"=="0" goto SKIP_INSTALL
goto INVALID_OPT

:ASK_OK
echo [Thông báo] Tất cả thư viện đều đã cài đủ và đạt chuẩn phiên bản.
echo.
echo   [1] Vẫn nâng cấp toàn bộ thư viện lên phiên bản mới nhất
echo   [0] Giữ nguyên phiên bản hiện tại, không cần cài lại
echo.
set "opt="
set /p "opt=Nhập lựa chọn của bạn (0-1) rồi nhấn Enter: "
if defined opt set "opt=%opt: =%"
if "%opt%"=="1" goto UPGRADE_ALL
if "%opt%"=="0" goto SKIP_INSTALL
goto INVALID_OPT

:SKIP_INSTALL
echo.
echo [Thông tin] Đã bỏ qua bước cài đặt. Bạn có thể bấm tiếp: 2_kiem_thu_logic.bat
echo.
pause
exit /b 0

:INVALID_OPT
echo.
echo [Cảnh báo] Lựa chọn không hợp lệ. Đã hủy thao tác cài đặt.
echo.
pause
exit /b 0

:INSTALL_REQ
echo.
echo [Thông tin] Đang cài đặt / cập nhật thư viện theo requirements.txt...
%PY_CMD% -m pip install -r requirements.txt
goto DONE_CHECK

:UPGRADE_ALL
echo.
echo [Thông tin] Đang nâng cấp toàn bộ thư viện lên phiên bản mới nhất...
%PY_CMD% -m pip install --upgrade -r requirements.txt
goto DONE_CHECK

:DONE_CHECK
echo.
if %errorlevel%==0 (
    echo [Hoàn tất] Cài đặt / cập nhật thư viện thành công. Bạn có thể bấm tiếp: 2_kiem_thu_logic.bat
) else (
    echo [Lỗi] Cài đặt thư viện thất bại - mã lỗi: %errorlevel%.
)
echo.
pause
