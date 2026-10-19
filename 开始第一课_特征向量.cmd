@echo off
chcp 65001 >nul
setlocal
call "%~dp0scripts\project_environment.cmd" "%~dp0"
if errorlevel 1 (
  pause
  exit /b 1
)
"%PROJECT_PYTHON%" -X utf8 -u "%~dp0scripts\inspect_feature_flow.py" --baseline-dir "%~dp0experiments\baseline_20261016_204723_088238" --data-root "%~dp0datasets\MVTecAD" --output-root "%~dp0artifacts\learning" %*
set "PROJECT_EXIT=%ERRORLEVEL%"
echo.
echo Exit code: %PROJECT_EXIT%
echo Project folder: %~dp0
pause
exit /b %PROJECT_EXIT%
