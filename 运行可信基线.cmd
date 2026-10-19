@echo off
chcp 65001 >nul
setlocal
call "%~dp0scripts\project_environment.cmd" "%~dp0"
if errorlevel 1 (
  pause
  exit /b 1
)
"%PROJECT_PYTHON%" -X utf8 -u "%~dp0run_trusted_baseline_on_this_laptop.py" --data-root "%~dp0datasets\MVTecAD" --output-root "%~dp0experiments" %*
set "PROJECT_EXIT=%ERRORLEVEL%"
echo.
echo Exit code: %PROJECT_EXIT%
echo Project folder: %~dp0
pause
exit /b %PROJECT_EXIT%
