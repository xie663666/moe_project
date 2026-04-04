@echo off
setlocal
set TARGET=F:\mycode\keyan_mouth_three\moe_project\six_moe
set SOURCE=%~dp0

if not exist "F:\mycode\keyan_mouth_three\moe_project" mkdir "F:\mycode\keyan_mouth_three\moe_project"
if not exist "%TARGET%" mkdir "%TARGET%"

robocopy "%SOURCE%" "%TARGET%" /E /XD .venv __pycache__ results\runs checkpoints logs /XF *.pyc
if %ERRORLEVEL% GEQ 8 (
  echo Copy failed. robocopy errorlevel=%ERRORLEVEL%
  exit /b 1
)

echo Project copied to %TARGET%
echo Now run:
echo   cd /d %TARGET%
echo   bootstrap_local_windows.bat
endlocal
