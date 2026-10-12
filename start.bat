@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "HF_HUB_OFFLINE=1"
title Bilingual Text Analyzer

set "PY="
if exist "python\python.exe" set "PY=python\python.exe"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
    echo [ОШИБКА] Не найден python\python.exe рядом с start.bat.
    echo Распакуйте архив целиком в обычную папку и запускайте start.bat из неё.
    pause
    exit /b 1
)
if not exist "app\server.py" (
    echo [ОШИБКА] Не найдена папка app. Архив распакован не полностью.
    pause
    exit /b 1
)

set "HASDB="
for /d %%D in ("library\*") do if exist "%%~D\book.db" set "HASDB=1"
if not defined HASDB (
    echo [ВНИМАНИЕ] В папке library нет готовой базы книги.
    echo Откройте вкладку "База данных" и постройте её из исходного файла.
)
if not exist "tools\llama.cpp\bin\llama-server.exe" (
    echo [ВНИМАНИЕ] Нет tools\llama.cpp\bin\llama-server.exe: поиск по смыслу недоступен,
    echo работает только полнотекстовый поиск.
)

echo Запуск... браузер откроется автоматически, адрес страницы будет показан ниже.
echo Чтобы выйти, закройте это окно.
"%PY%" main.py %*
if errorlevel 1 (
    echo.
    echo Программа завершилась с ошибкой. Подробности в папке logs.
    pause
)
