@echo off
rem Run the IsaacLab conda env's Python with the Isaac Sim (binary install)
rem environment variables that C:\isaacsim\python.bat would set.
rem Usage: scripts\isaac_python.bat <script.py> [args...]
rem Override locations with ISAACSIM_ROOT / ISAACLAB_PYTHON.
setlocal
if "%ISAACSIM_ROOT%"=="" set "ISAACSIM_ROOT=C:\isaacsim"
if "%ISAACLAB_PYTHON%"=="" set "ISAACLAB_PYTHON=%USERPROFILE%\anaconda3\envs\IsaacLab\python.exe"
set "PYTHONPATH=%PYTHONPATH%;%ISAACSIM_ROOT%\site"
set "CARB_APP_PATH=%ISAACSIM_ROOT%\kit"
set "ISAAC_PATH=%ISAACSIM_ROOT%"
set "EXP_PATH=%ISAACSIM_ROOT%\apps"
set "PYTHONUNBUFFERED=1"
"%ISAACLAB_PYTHON%" %*
exit /B %ERRORLEVEL%
