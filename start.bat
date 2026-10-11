@echo off

cd /d "%~dp0"
if not exist .venv (
  echo [FishWatch] First run: creating venv and installing deps...
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python -m pip install -U pip
  .venv\Scripts\python -m pip install -r requirements.txt
)
.venv\Scripts\python app.py
pause
