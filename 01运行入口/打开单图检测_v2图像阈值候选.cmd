@echo off
chcp 65001 >nul
setlocal
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI\"
call "%PROJECT_ROOT%02代码与配置\scripts\project_environment.cmd" "%PROJECT_ROOT%"
if errorlevel 1 goto :failed
set "PYTHONDONTWRITEBYTECODE=1"
"%PROJECT_PYTHON%" -X utf8 -u "%PROJECT_ROOT%02代码与配置\scripts\inspection_image_v2.py" %*
if errorlevel 1 goto :failed
exit /b 0
:failed
echo.
echo 图像阈值v2候选入口未正常完成，请保留上方错误信息。
pause
exit /b 1
