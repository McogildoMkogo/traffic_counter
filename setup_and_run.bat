@echo off
echo Setting up Traffic Counter Application...

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install requirements
echo Installing required packages...
pip install streamlit ultralytics opencv-python numpy pandas matplotlib seaborn

REM Run the application
echo Starting the application...
python -m streamlit run traffic_counter.py

pause 