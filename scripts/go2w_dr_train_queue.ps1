# Go2-W sim-to-real fine-tune (task Go2WM-Velocity-Rough-Go2W-DR-v0: friction / mass / CoM / gains / 0-40 ms action
# delay / pushes, plus scan crops and random sidewalk tiles in the terrain), resumed from the current rough policy.
# Waits for the scan seed queue (scripts/scan_seeds_queue.ps1) to finish so the two never share the GPU.
param([string]$Resume = "D:/go2wm/ext/go2w_runs/rough/model_1999.pt", [int]$Iterations = 2000,
      [string]$Out = "D:/go2wm/ext/go2w_runs/rough_dr", [string]$WaitFor = "D:/go2wm/outputs/scan_missions/seeds/queue.log")
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "src"
New-Item -ItemType Directory -Force $Out | Out-Null
$log = "$Out/queue.log"
"[$(Get-Date -Format 'MM-dd HH:mm:ss')] waiting for $WaitFor" | Out-File -Append -Encoding utf8 $log
while ($WaitFor -and -not (Select-String -Path $WaitFor -Pattern "QUEUE DONE" -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 60
}
"[$(Get-Date -Format 'MM-dd HH:mm:ss')] START train (resume $Resume, $Iterations iterations)" | Out-File -Append -Encoding utf8 $log
cmd /c "scripts\isaac_python.bat scripts\train_go2w.py --task Go2WM-Velocity-Rough-Go2W-DR-v0 --num-envs 4096 --iterations $Iterations --resume $Resume --out $Out > $Out/train.log 2>&1"
"[$(Get-Date -Format 'MM-dd HH:mm:ss')] END train (exit $LASTEXITCODE)" | Out-File -Append -Encoding utf8 $log
