@echo off
cd /d "%~dp0"
start http://localhost:8501
streamlit run app.py
pause
