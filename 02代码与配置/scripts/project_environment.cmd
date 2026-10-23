@echo off
set "PROJECT_ROOT=%~1"
set "PROJECT_PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
set "HF_HUB_OFFLINE=1"
set "TORCH_HOME=%PROJECT_ROOT%.cache\models\torch"
set "HF_HOME=%PROJECT_ROOT%.cache\models\huggingface"
set "XDG_CACHE_HOME=%PROJECT_ROOT%.cache\models"
set "MPLCONFIGDIR=%PROJECT_ROOT%.cache\matplotlib"
set "TEMP=%PROJECT_ROOT%.cache\tmp"
set "TMP=%PROJECT_ROOT%.cache\tmp"
if not exist "%PROJECT_PYTHON%" (
  echo Project Python environment is missing: %PROJECT_PYTHON%
  exit /b 1
)
exit /b 0
