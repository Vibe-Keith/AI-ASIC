@echo off
REM AI-ASIC graphical interface launcher for Windows (cmd.exe).
REM   run-gui.bat
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -m ai_asic.gui %*
) else (
  python -m ai_asic.gui %*
)
