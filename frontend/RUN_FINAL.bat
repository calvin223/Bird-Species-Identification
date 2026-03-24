@echo off
cls
color 0B
echo.
echo ==============================================================
echo      BIRD SPECIES ID - PROFESSIONAL ORNITHOLOGIST EDITION    
echo ==============================================================
echo.
echo Features:
echo   - Real individual model predictions
echo   - Clickable recent predictions (last 5)
echo   - Agreement matrix showing model consensus
echo   - Confusion likelihood for similar species
echo   - Quality assessment indicators
echo   - Field notes and survey mode
echo.
echo ==============================================================
echo.

cd /d "D:\University\Comp702\project\frontend\backend_integration"

REM Kill old process
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000') do (
    taskkill /PID %%a /F 2>nul
)

timeout /t 2 /nobreak >nul

echo Starting backend with real predictions...
echo.
echo Once backend starts:
echo   1. Open: index_ornithologist.html
echo   2. Upload an audio file
echo   3. Click recent predictions to review
echo   4. Check agreement matrix and confusion species
echo.
echo Backend URL: http://localhost:5000
echo ==============================================================

python backend_direct.py
