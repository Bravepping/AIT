@echo off
setlocal
pushd "%~dp0"
if not exist "VisionHardwareCheck.exe" if not exist "VisionStudio.exe" goto missing
if not exist "vision-runtime\python.exe" goto missing
"vision-runtime\python.exe" -B -X utf8 "setup-vision-tensorrt.py" %*
if errorlevel 1 goto failed
if "%~1"=="" pause
popd
exit /b 0

:missing
echo Place Install-TensorRT.cmd beside VisionStudio.exe or VisionHardwareCheck.exe.
:failed
echo TensorRT installation did not complete. Close VisionHardwareCheck and see the message above.
if "%~1"=="" pause
popd
exit /b 1
