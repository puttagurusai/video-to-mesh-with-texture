@echo off
rem Downloads the third-party pieces this project needs into tools\ and installs the Python packages (requirements.txt).
rem It lists what it will download and asks before starting.  Options (add after the file name):
rem    --check            show what is installed / missing, change nothing
rem    --dry-run          show what would be downloaded, change nothing
rem    --yes              do not ask for confirmation
rem    --with-optional    also networkx + pycolmap and the Depth-Anything-V2-Small model (edge route / COLMAP backend)
rem    --from-existing "D:\old\project"   reuse the DA3 source and weights of an existing install (no download, no extra disk)
rem    --only a,b / --skip a,b   components: deps da3-src da3-model dav2-metric optional-deps dav2-small
rem Python used: the first "python" on PATH (tested: 3.13.7).
setlocal DisableDelayedExpansion
cd /d "%~dp0"
title Setup: download third-party tools
python -u "setup_tools.py" %*
set RC=%ERRORLEVEL%
echo.
if "%RC%"=="0" goto ok
echo Setup finished with exit code %RC%. Read the messages above.
goto end
:ok
echo Setup finished.
:end
pause
exit /b %RC%
