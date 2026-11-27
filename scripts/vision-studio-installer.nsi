Unicode true
RequestExecutionLevel user
Name "NDI Vision Control V1.7 Final 候选"
OutFile "${SETUP_EXE}"
InstallDir "$LOCALAPPDATA\Programs\NDI Vision Control\V1.7 Final Candidate"
SetCompressor /SOLID lzma

!include "MUI2.nsh"
!include "x64.nsh"
!include "FileFunc.nsh"

!define APP_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\NDIVisionControl-V17FinalCandidate"
!define INSTALL_MARKER "VisionStudio-V1.7-Final-Candidate-install.marker"
!define SHORTCUT "NDI Vision Control V1.7 Final Candidate.lnk"

Var Smoke
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "SimpChinese"

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "此版本需要 Windows x64。"
    SetErrorLevel 1
    Quit
  ${EndIf}
  ${GetParameters} $0
  ClearErrors
  ${GetOptions} $0 "/SMOKETEST" $Smoke
  ${If} ${Errors}
    StrCpy $Smoke 0
  ${Else}
    StrCpy $Smoke 1
  ${EndIf}
FunctionEnd

Section "NDI Vision Control"
  IfFileExists "$INSTDIR\${INSTALL_MARKER}" CopyFiles 0
  ClearErrors
  FindFirst $0 $1 "$INSTDIR\*"
  IfErrors CopyFiles
CheckEntry:
  StrCmp $1 "" EmptyDirectory
  StrCmp $1 "." NextEntry
  StrCmp $1 ".." NextEntry
  FindClose $0
  MessageBox MB_ICONSTOP "安装目录已有其他文件。请选择空目录。" /SD IDOK
  SetErrorLevel 1
  Abort
NextEntry:
  ClearErrors
  FindNext $0 $1
  IfErrors EmptyDirectory CheckEntry
EmptyDirectory:
  FindClose $0
CopyFiles:
  SetOutPath "$INSTDIR"
  File /r "${PAYLOAD_DIR}\*"
  FileOpen $0 "$INSTDIR\${INSTALL_MARKER}" w
  FileWrite $0 "$Smoke"
  FileClose $0
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  ${If} $Smoke == 0
    SetShellVarContext current
    CreateShortCut "$SMPROGRAMS\${SHORTCUT}" "$INSTDIR\VisionStudio.exe"
    WriteRegStr HKCU "${APP_KEY}" "DisplayName" "NDI Vision Control V1.7 Final 候选"
    WriteRegStr HKCU "${APP_KEY}" "DisplayVersion" "1.7.0-rc.2"
    WriteRegStr HKCU "${APP_KEY}" "InstallLocation" "$INSTDIR"
    WriteRegStr HKCU "${APP_KEY}" "UninstallString" "$\"$INSTDIR\Uninstall.exe$\""
  ${EndIf}
SectionEnd

Section "Uninstall"
  IfFileExists "$INSTDIR\${INSTALL_MARKER}" 0 EndUninstall
  FileOpen $0 "$INSTDIR\${INSTALL_MARKER}" r
  FileRead $0 $Smoke
  FileClose $0
  ${If} $Smoke == 0
    ReadRegStr $0 HKCU "${APP_KEY}" "InstallLocation"
    ${If} $0 == $INSTDIR
      DeleteRegKey HKCU "${APP_KEY}"
      SetShellVarContext current
      Delete "$SMPROGRAMS\${SHORTCUT}"
    ${EndIf}
  ${EndIf}
  Delete "$INSTDIR\VisionStudio.exe"
  Delete "$INSTDIR\VisionKmMonitor.exe"
  Delete "$INSTDIR\VisionKmCalibration.exe"
  Delete "$INSTDIR\VisionButtonStatus.exe"
  Delete "$INSTDIR\1013v26s_640.onnx"
  Delete "$INSTDIR\README.md"
  Delete "$INSTDIR\README-CUDA.txt"
  Delete "$INSTDIR\README-TensorRT.txt"
  Delete "$INSTDIR\Start-Vision-CUDA.cmd"
  Delete "$INSTDIR\Install-TensorRT.cmd"
  Delete "$INSTDIR\setup-vision-cuda.py"
  Delete "$INSTDIR\setup-vision-tensorrt.py"
  Delete "$INSTDIR\cuda-packages.json"
  Delete "$INSTDIR\tensorrt-packages.json"
  RMDir /r "$INSTDIR\vision-runtime"
  Delete "$INSTDIR\${INSTALL_MARKER}"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
EndUninstall:
SectionEnd
