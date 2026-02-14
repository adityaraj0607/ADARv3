@echo off
REM ============================================================
REM  ADAR V3.0 — Master Launcher
REM  Starts both the Fleet Server and Driver Client in
REM  separate terminal windows.
REM
REM  Usage:  double-click run_all.bat  (or run from terminal)
REM ============================================================

echo.
echo  ============================================================
echo    ADAR V3.0 — Launching Both Systems
echo  ============================================================
echo.

set ROOT=%~dp0
set VENV=%ROOT%.venv\Scripts\activate.bat

REM ── 1. Start Fleet Command Server (Port 8000) ──────────────
echo  [1/2] Starting Fleet Server on port 8000...
start "ADAR Fleet Server :8000" cmd /k "cd /d "%ROOT%ADAR_FLEET_SERVER" && call "%VENV%" && echo. && echo  ============================================= && echo   ADAR FLEET COMMAND SERVER  [Port 8000] && echo  ============================================= && echo. && python server.py"

REM Give the server a moment to boot before the client connects
timeout /t 3 /nobreak > nul

REM ── 2. Start Driver Client (Port 5000) ─────────────────────
echo  [2/2] Starting Driver Client on port 5000...
start "ADAR Driver Client :5000" cmd /k "cd /d "%ROOT%ADAR_DRIVER_CLIENT" && call "%VENV%" && echo. && echo  ============================================= && echo   ADAR DRIVER CLIENT  [Port 5000] && echo  ============================================= && echo. && python main.py"

echo.
echo  ============================================================
echo   Both systems launched!
echo.
echo   Driver HUD:       http://localhost:5000
echo   Fleet Dashboard:  http://localhost:8000
echo.
echo   Close this window or press any key to exit.
echo  ============================================================

pause > nul
