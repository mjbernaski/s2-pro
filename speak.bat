@echo off
REM Launcher for speak.py - ensures conda base env is active

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" base

python -c "import loguru, torch" 2>nul || (
    echo Installing missing dependencies...
    pip install loguru
)

cd /d "%~dp0"
python speak.py %*
