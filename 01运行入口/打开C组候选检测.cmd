@echo off
chcp 65001 >nul
set "PROJECT_ROOT=%~dp0..\"
call "%PROJECT_ROOT%02代码与配置\scripts\project_environment.cmd" "%PROJECT_ROOT%"
if errorlevel 1 goto end
"%PROJECT_PYTHON%" -B "%PROJECT_ROOT%02代码与配置\scripts\delivery_app.py" --profile wrn50_c_experimental
:end
pause
