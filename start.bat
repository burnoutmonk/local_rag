@echo off
setlocal enabledelayedexpansion

:: Load .env file
for /f "tokens=1,2 delims==" %%a in ('type .env ^| findstr /v "^#"') do (
    set %%a=%%b
)

echo CUDA_AVAILABLE=!CUDA_AVAILABLE!

if "!CUDA_AVAILABLE!"=="true" (
    echo Starting with GPU support...
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
) else (
    echo Starting in CPU mode...
    docker compose up -d --build
)

if errorlevel 1 (
    echo ERROR: Docker Compose failed.
    pause
    exit /b 1
)

echo Waiting for services to be ready...
:wait
docker inspect rag_ready --format "{{.State.Status}}" 2>nul | findstr "exited" >nul
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto wait
)

echo Opening browser...
start http://localhost:8000
pause
