@echo off
setlocal
chcp 65001 >nul
python -m panther %*
if %ERRORLEVEL% NEQ 0 (
    if "%~1"=="" (
        echo.
        pause
    )
)
endlocal
