@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

if "%1"=="" (
    set DEVICE=cuda
) else (
    set DEVICE=%1
)

call bootstrap_local_windows.bat || exit /b 1
call .venv\Scripts\activate.bat

echo [REF] Running reference stage...
python scripts\run_stage.py --config-dir configs\generated\main_round\ref --device %DEVICE% --skip-existing --failed-list logs\failed_ref_runs.txt || exit /b 1

echo [CHECK] Reference outputs...
python scripts\preflight_check.py --project-root . --stage refs_ready || exit /b 1

echo [DYN] Running target dynamic stage...
python scripts\run_stage.py --config-dir configs\generated\main_round\dynamic --device %DEVICE% --skip-existing --failed-list logs\failed_dynamic_runs.txt || exit /b 1

echo [HYB] Running hybrid/fixed stage...
python scripts\preflight_check.py --project-root . --stage hybrid_ready || exit /b 1
python scripts\run_stage.py --config-dir configs\generated\main_round\hybrid --device %DEVICE% --skip-existing --failed-list logs\failed_hybrid_runs.txt || exit /b 1

echo [ANALYSIS] Aggregating, making tables, and plotting...
python scripts\aggregate_results.py --results-root results\runs --out-dir results\summaries || exit /b 1
python scripts\make_analysis_tables.py --summaries-dir results\summaries --out-dir results\analysis\tables || exit /b 1
python scripts\plot_results.py --summaries-dir results\summaries --tables-dir results\analysis\tables --out-dir results\analysis\plots || exit /b 1

echo Full pipeline finished.
endlocal
