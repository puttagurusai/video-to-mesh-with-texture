@echo off
rem Video -> textured 3D scene.  Drag a video onto this file, or:  run.bat "C:\path\to\video.mp4" [output folder]
rem Output goes to runs\<video name> (or the folder given as 2nd argument); every run writes a NEW log file to <output>\logs\
rem and the 3D viewer opens when it finishes. The window always stays open at the end so nothing is lost.
rem
rem Written without parenthesised blocks on purpose: a video path containing ( or ) breaks those and closed the window.
setlocal DisableDelayedExpansion
cd /d "%~dp0"
title Video to 3D

if "%~1"=="" goto usage
if not exist "%~f1" goto notfound

echo ============================================================
echo  Video to 3D
echo  video : %~f1
echo ============================================================
echo.

set "OUTARG="
if not "%~2"=="" set OUTARG=--out "%~f2"

echo Checking this computer ...
python run_pipeline.py --check >nul 2>&1
if errorlevel 1 goto notready

echo Ready. Starting the pipeline - each step is shown below as it runs.
echo.
python run_pipeline.py --video "%~f1" %OUTARG% --open
set RC=%ERRORLEVEL%
echo.
if "%RC%"=="0" goto done
echo ============================================================
echo  The run FAILED (exit code %RC%). The error is shown above and in the log file.
echo ============================================================
goto end

:done
echo ============================================================
echo  Finished. The output folder, viewer and log file are listed above.
echo ============================================================
goto end

:usage
echo Usage: run.bat "path\to\video.mp4" [output folder]
echo        or drag a video file onto run.bat
set RC=2
goto end

:notfound
echo Video not found: %~f1
set RC=2
goto end

:notready
echo This computer is not ready. Details:
echo.
python run_pipeline.py --check
set RC=1
goto end

:end
echo.
pause
exit /b %RC%
