@echo off
REM Voice AI Assistant — start all three model servers
REM Run this from the project root: app\start_servers.bat

set ROOT=%~dp0..

echo Starting ASR server (moonshine env) on :8001 ...
start "ASR :8001" cmd /k "conda activate moonshine && python "%ROOT%\app\asr_server.py""

echo Starting LLM server (llm env) on :8002 ...
start "LLM :8002" cmd /k "conda activate llm && python "%ROOT%\app\llm_server.py""

echo Starting TTS server (gptsovits env) on :8003 ...
start "TTS :8003" cmd /k "conda activate gptsovits && python "%ROOT%\app\tts_server.py""

echo.
echo Three windows opened. Wait for all three to print "Ready" then run:
echo   python app\voice_assistant.py
echo.
pause
