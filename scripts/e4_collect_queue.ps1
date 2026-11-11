# E4 terrain data collection in resumable chunks (one Isaac process per chunk).
#   powershell -ExecutionPolicy Bypass -File scripts\e4_collect_queue.ps1 [-TrainEpisodes 144] [-TestEpisodes 48] [-Chunk 24]
#   E4-S (with height scans): ... -DataTag e4s -ExtraArgs "--record-scan"
# A chunk whose episodes already exist is skipped, so re-running resumes after an interruption.
param(
    [int]$TrainEpisodes = 144,
    [int]$TestEpisodes = 48,
    [int]$Chunk = 24,
    [string]$LogDir = "D:\go2wm\logs\e4_collect",
    [string]$DataTag = "e4",
    [string]$ExtraArgs = ""
)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force $LogDir | Out-Null
$splits = @(@{name = "train"; n = $TrainEpisodes; seed = 70000; dir = "D:\go2wm\data\go2_${DataTag}_train"},
            @{name = "test"; n = $TestEpisodes; seed = 80000; dir = "D:\go2wm\data\go2_${DataTag}_test"})
foreach ($sp in $splits) {
    for ($first = 0; $first -lt $sp.n; $first += $Chunk) {
        $count = [math]::Min($Chunk, $sp.n - $first)
        $last = $sp.seed + $first + $count - 1
        $done = Get-ChildItem "$($sp.dir)\episodes" -Filter "*_s$('{0:D6}' -f $last)*" -ErrorAction SilentlyContinue
        if ($done) { Write-Output "[queue] skip $($sp.name) $first (done)"; continue }
        $free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory / 1MB, 1)
        $tag = "${DataTag}_$($sp.name)_$first"
        Write-Output "[queue] $(Get-Date -Format s) start $tag ($count eps, free commit $free GB)"
        cmd /c "scripts\isaac_python.bat scripts\collect_terrain.py --split $($sp.name) --episodes $count --first $first --data data/go2_${DataTag}_$($sp.name) $ExtraArgs > $LogDir\$tag.log 2>&1"
        Write-Output "[queue] $(Get-Date -Format s) end $tag exit=$LASTEXITCODE"
    }
}
Write-Output "[queue] all done"
