@echo off
cd /d "C:\Users\SherdllStore\Desktop\BotNet"
echo Starting BotNet Dashboard...
echo.
echo When you see "Local URL: http://localhost:8501"
echo open Chrome and go to: http://localhost:8501
echo.
echo DO NOT close this window while using the app.
echo.
.venv\Scripts\streamlit.exe run app.py
pause
