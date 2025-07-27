@echo off
echo Connect 4 GPU Game - Windows Launcher
echo ====================================

REM Check if CUDA is available
nvcc --version >nul 2>&1
if errorlevel 1 (
    echo Error: CUDA compiler (nvcc) not found!
    echo Please install CUDA Toolkit and add it to your PATH.
    pause
    exit /b 1
)

echo CUDA compiler found. Building project...

REM Compile the main program
nvcc -o connect4_gpu.exe connect4_gpu.cu -O3
if errorlevel 1 (
    echo Error: Compilation failed!
    pause
    exit /b 1
)

echo Compilation successful!

REM Check if Python is available for visualizer
python --version >nul 2>&1
if errorlevel 1 (
    echo Warning: Python not found. Visualizer will not be available.
    echo You can still run the game without visualization.
) else (
    echo Python found. Visualizer is available.
)

echo.
echo Choose an option:
echo 1. Run a single game
echo 2. Run multiple games (5)
echo 3. Run visualizer (requires Python)
echo 4. Test CUDA environment
echo 5. Exit

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo Running single game...
    connect4_gpu.exe
) else if "%choice%"=="2" (
    echo Running 5 games...
    for /l %%i in (1,1,5) do (
        echo.
        echo Game %%i:
        connect4_gpu.exe
        echo -------------------
    )
) else if "%choice%"=="3" (
    echo Starting visualizer...
    python game_visualizer.py
) else if "%choice%"=="4" (
    echo Testing CUDA environment...
    nvcc -o test_compile.exe test_compile.cu
    if not errorlevel 1 (
        test_compile.exe
    )
) else if "%choice%"=="5" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice!
)

echo.
echo Press any key to exit...
pause >nul 