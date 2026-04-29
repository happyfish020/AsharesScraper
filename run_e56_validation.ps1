# V8A-E56 validation runner
# Run from project root: D:\LHJ\PythonWS\MarketMon\GrowthAlpha_V7
$ErrorActionPreference = "Stop"

$PY = "$env:C:\Apps\Python\Python312\python.exe"
if (!(Test-Path $PY)) { $PY = "python" }

$years = @(2022, 2023, 2024, 2025, 2010, 2013, 2015, 2020)
foreach ($y in $years) {
    Write-Host "===== E56 backtest year $y ====="
    & $PY scripts\backtest_adaptive.py --config config\config_e56.yaml --version V8A-E56 --year $y
}
