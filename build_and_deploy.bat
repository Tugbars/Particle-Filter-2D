@echo off
REM ============================================================================
REM build_and_deploy.bat - Build particle filter and deploy DLL to Python folder
REM Run from: PARTICLE-FILTER-2D\ (project root)
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================
echo Particle Filter 2D - Build ^& Deploy
echo ============================================
echo.

REM Check we're in the right directory
if not exist "CMakeLists.txt" (
    echo ERROR: Run this from the project root directory!
    echo        ^(where CMakeLists.txt is located^)
    pause
    exit /b 1
)

REM Create build directory if needed
if not exist "build" (
    echo Creating build directory...
    mkdir build
    pushd build
    cmake .. -G "Visual Studio 17 2022" -A x64
    popd
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: CMake configuration failed!
        pause
        exit /b 1
    )
)

REM Build shared library
echo.
echo Building particle_filter_2d_shared [Release]...
cmake --build build --config Release --target particle_filter_2d_shared

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo Build successful!

REM Create python directory if needed
if not exist "python" (
    echo Creating python directory...
    mkdir python
)

REM Copy DLL
echo.
echo Deploying to python folder...

set "DLL_SRC=build\Release\particle_filter_2d.dll"
set "DLL_DST=python\particle_filter_2d.dll"

if exist "%DLL_SRC%" (
    copy /Y "%DLL_SRC%" "%DLL_DST%" >nul
    echo   [OK] particle_filter_2d.dll
) else (
    echo   [FAIL] %DLL_SRC% not found!
    pause
    exit /b 1
)

echo.
echo ============================================
echo Done! 
echo ============================================
echo.
echo Python folder contents:
echo.
dir /B python\*.dll python\*.py 2>nul
echo.
echo To test:
echo   cd python
echo   python pf2d.py
echo.
pause