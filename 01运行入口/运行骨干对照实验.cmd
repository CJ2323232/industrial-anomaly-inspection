@echo off
chcp 65001 >nul
setlocal
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI\"
call "%PROJECT_ROOT%02代码与配置\scripts\project_environment.cmd" "%PROJECT_ROOT%"
if errorlevel 1 (
  if /i not "%~1"=="--check-only" pause
  exit /b 1
)
"%PROJECT_PYTHON%" -X utf8 -u "%PROJECT_ROOT%02代码与配置\scripts\compare_backbones.py" %*
set "PROJECT_EXIT=%ERRORLEVEL%"
echo.
echo Exit code: %PROJECT_EXIT%
echo Project folder: %PROJECT_ROOT%
if /i not "%~1"=="--check-only" pause
exit /b %PROJECT_EXIT%
