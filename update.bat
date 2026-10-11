@echo off
cd /d "%~dp0"
echo [FishWatch] Stopping running instance...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
echo [FishWatch] Pulling latest code...
git -c safe.directory=* pull --ff-only
if errorlevel 1 (
  echo [FishWatch] git pull failed, see message above.
  pause
  exit /b 1
)
if exist .venv .venv\Scripts\python -m pip install -q -r requirements.txt
call start.bat
