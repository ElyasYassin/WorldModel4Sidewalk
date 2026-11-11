# E4-C: train terrain GRUs fed each sensor's map (plus GRU / perfect map / height scanner), 3 seeds each.
#   powershell -ExecutionPolicy Bypass -File scripts\e4c_train_queue.ps1
#   fused maps: ... -Models depth_d555_k10_gru,depth_d435_k10_gru,l1_k10_gru,mid360_tilt_k10_gru,depth_d555_l1_k10_gru
# A run whose train_metrics.json exists (written at the end) is skipped, so re-running resumes.
param([string]$LogDir = "D:\go2wm\logs\e4c_train", [int[]]$Seeds = @(0, 1, 2), [string[]]$Models = @())
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force $LogDir | Out-Null
$py = "$env:USERPROFILE\anaconda3\envs\IsaacLab\python.exe"
$default = @("gru", "terrain_gru", "scan_gru", "depth_d555_gru", "depth_d435_gru", "l1_gru", "mid360_tilt_gru", "mid360_gru")
$models = if ($Models.Count) { $Models -split "," | Where-Object { $_ } } else { $default }  # -File passes "a,b" as one string
foreach ($seed in $Seeds) {
    foreach ($m in $models) {
        $out = "outputs/e4c_${m}_s$seed"
        if (Test-Path "D:\go2wm\$out\train_metrics.json") { Write-Output "[queue] skip $out"; continue }
        Write-Output "[queue] $(Get-Date -Format s) start $out"
        cmd /c "`"$py`" scripts\train.py --config configs\world_model_e4c_$m.yaml --data data/go2_e4c_train --out $out --seed $seed > $LogDir\e4c_${m}_s$seed.log 2>&1"
        Write-Output "[queue] $(Get-Date -Format s) end $out exit=$LASTEXITCODE"
    }
}
