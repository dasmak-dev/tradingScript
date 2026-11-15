@echo off
chcp 65001 >nul
title Bot de Trading - instalacion y arranque
cd /d "%~dp0"
echo ==== Bot de Trading Intradia: instalacion y arranque ====

set PY=
py -3 -c "import sys" >nul 2>nul && set PY=py -3
if not defined PY python -c "import sys" >nul 2>nul && set PY=python
if not defined PY (
  echo [1/4] Python no encontrado. Instalando Python 3.12 con winget...
  winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
  set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
)
echo [1/4] Usando Python: %PY%
%PY% --version || (echo ERROR: no se pudo ejecutar Python. & pause & exit /b 1)

if not exist ".venv\Scripts\python.exe" (
  echo [2/4] Creando entorno virtual .venv ...
  %PY% -m venv .venv || (echo ERROR creando el entorno virtual & pause & exit /b 1)
) else (
  echo [2/4] Entorno virtual .venv ya existe
)

echo [3/4] Instalando dependencias ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo ERROR instalando dependencias & pause & exit /b 1)

echo [4/4] Arrancando la aplicacion en http://localhost:8501 ...
start "" cmd /c "timeout /t 8 >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless true --server.port 8501
echo La aplicacion se ha detenido.
pause
