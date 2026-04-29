param(
    [string]$StartDate = "2010-01-01",
    [string]$EndDate = "latest",
    [int]$SleepSec = 1,
    [switch]$NoVpn
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

function Get-LatestTradeDate {
    @'
from app.trading_day import get_latest_trade_date
print(get_latest_trade_date())
'@ | & $python -
}

if ($EndDate -eq "latest") {
    $EndDate = (Get-LatestTradeDate).Trim()
    if ($EndDate -match "^\d{8}$") {
        $EndDate = "$($EndDate.Substring(0,4))-$($EndDate.Substring(4,2))-$($EndDate.Substring(6,2))"
    }
}

$start = [datetime]::Parse($StartDate)
$end = [datetime]::Parse($EndDate)
if ($start -gt $end) {
    throw "StartDate must be <= EndDate"
}

$cur = Get-Date -Year $start.Year -Month $start.Month -Day 1
while ($cur -le $end) {
    $monthStart = $cur
    $monthEnd = $cur.AddMonths(1).AddDays(-1)
    if ($monthStart -lt $start) { $monthStart = $start }
    if ($monthEnd -gt $end) { $monthEnd = $end }

    $days = [int]($monthEnd - $monthStart).TotalDays + 1
    $asof = $monthEnd.ToString("yyyyMMdd")

    Write-Host ("[BACKFILL][INDEX] {0} -> {1} (days={2})" -f $monthStart.ToString("yyyy-MM-dd"), $monthEnd.ToString("yyyy-MM-dd"), $days)

    $args = @(
        (Join-Path $root "runner.py"),
        "--tasks", "index",
        "--asof", $asof,
        "--days", $days
    )
    if ($NoVpn) { $args += "--no-vpn" }

    & $python @args

    if ($SleepSec -gt 0) { Start-Sleep -Seconds $SleepSec }
    $cur = $cur.AddMonths(1)
}
