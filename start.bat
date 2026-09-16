@echo off
python run_pipeline.py
if errorlevel 1 exit /b 1
streamlit run dashboard/app.py