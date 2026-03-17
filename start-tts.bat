@echo off
title Fish Audio S2 Pro TTS Server
cd /d E:\Development\s2-pro
call C:\Users\berna\miniconda3\condabin\activate.bat
python server.py %*
pause
