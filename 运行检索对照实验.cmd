@echo off
chcp 65001 >nul
setlocal
call "%~dp0scripts\project_environment.cmd" "%~dp0"
if errorlevel 1 (
  pause
  exit /b 1
)
"%PROJECT_PYTHON%" -X utf8 -u "%~dp0scripts\compare_retrieval.py" --baseline-dir "%~dp0experiments\baseline_20261016_204723_088238" --data-root "%~dp0datasets\MVTecAD" --policy "%~dp0configs\decisions\metal_nut-q975-provisional.json" --output-root "%~dp0experiments" %*
set "PROJECT_EXIT=%ERRORLEVEL%"
echo.
echo Exit code: %PROJECT_EXIT%
echo Results: %~dp0experiments
pause
exit /b %PROJECT_EXIT%
