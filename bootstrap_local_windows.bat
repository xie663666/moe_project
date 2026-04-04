@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"
echo [1/6] Project root: %CD%

if not exist ".venv" (
    echo [2/6] Creating virtual environment...
    py -3 -m venv .venv
) else (
    echo [2/6] Virtual environment already exists.
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo [3/6] Expanding directed task pairs...
python scripts\expand_pairs.py --pairs configs\pairs\main_pairs.yaml --out configs\pairs\directed_main_pairs.yaml

echo [4/6] Generating run configs...
python scripts\gen_run_configs.py --round main --tasks configs\tasks\cifar100_superclass_tasks.yaml --pairs configs\pairs\directed_main_pairs.yaml --output configs\generated\main_round

echo [5/6] Running smoke test...
python scripts\smoke_test.py --project-root .

echo [6/6] Checking key directories...
if exist "configs\generated\main_round" (echo OK configs\generated\main_round) else (echo MISSING configs\generated\main_round)
if exist "results" (echo OK results) else (echo MISSING results)
if exist "logs" (echo OK logs) else (echo MISSING logs)
if exist "checkpoints" (echo OK checkpoints) else (echo MISSING checkpoints)

echo Done.
endlocal
