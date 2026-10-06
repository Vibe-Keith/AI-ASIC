@echo off
REM AI-ASIC launcher for Windows (cmd.exe). Examples:
REM   run.bat detect
REM   run.bat profiles
REM   run.bat infer "hello world"
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -m ai_asic.cli %*
) else (
  python -m ai_asic.cli %*
)
