@echo off
setlocal
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python scripts\aggregate_results.py --results-root results\runs --out-dir results\summaries || exit /b 1
python scripts\make_analysis_tables.py --summaries-dir results\summaries --out-dir results\analysis\tables || exit /b 1
python scripts\plot_results.py --summaries-dir results\summaries --tables-dir results\analysis\tables --out-dir results\analysis\plots || exit /b 1
echo Analysis finished.
endlocal
