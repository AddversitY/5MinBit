@echo off
rem Double-click to start the btc5m paper test (paper mode only: no wallet, keys or money).
powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%~dp0run-paper-test.ps1" %*
pause
