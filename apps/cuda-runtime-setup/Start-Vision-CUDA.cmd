@echo off
setlocal
pushd "%~dp0"
if not exist "VisionHardwareCheck.exe" if not exist "VisionStudio.exe" goto missing
if not exist "vision-runtime\python.exe" goto missing
"vision-runtime\python.exe" -B -X utf8 "setup-vision-cuda.py"
if errorlevel 1 goto failed
popd
exit /b 0

:missing
echo Place the CUDA setup files beside VisionStudio.exe or VisionHardwareCheck.exe.
:failed
echo CUDA setup did not complete. See the message above.
pause
popd
exit /b 1
