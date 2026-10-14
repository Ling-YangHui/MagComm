@echo off
setlocal

rem ============================================================
rem  One-click wrapper for script\export_ds.py
rem  Usage:
rem    run_export_ds.bat "C:/.../xxx.ds"
rem  Or double-click it, then paste the .ds path when prompted.
rem ============================================================

set "ADS_PYTHON=C:\Program Files\Keysight\ADS2026\tools\python\python.exe"
set "ROOT_DIR=%~dp0"

set "DAT_PATH=%~1"
if not defined DAT_PATH (
    set /p "DAT_PATH=Enter .ds file path: "
)
if not defined DAT_PATH (
    echo No .ds path provided. Exiting.
    exit /b 1
)

echo Exporting: %DAT_PATH%
"%ADS_PYTHON%" "%ROOT_DIR%script\export_ds.py" "%DAT_PATH%" --out-dir "%ROOT_DIR%out"

if errorlevel 1 (
    echo Export failed with exit code %errorlevel%.
    pause
    exit /b 1
)

echo Done. Output written to: %ROOT_DIR%out
