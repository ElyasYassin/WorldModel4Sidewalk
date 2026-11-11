# E5: pretrained-model variants (A1 image features, C monocular depth), 3 seeds each. Waits for nothing:
# derived features must exist (scripts/extract_rgb_features.py, scripts/rgb_depth_maps.py).
#   powershell -ExecutionPolicy Bypass -File scripts\e5_queue.ps1 [-Models a,b]
# A run whose best.pt exists is skipped, so re-running resumes.
param([string]$LogDir = "D:\go2wm\logs\e5_train", [int[]]$Seeds = @(0, 1, 2), [string[]]$Models = @())
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force $LogDir | Out-Null
$py = "$env:USERPROFILE\anaconda3\envs\IsaacLab\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$default = @("dinov2s_gru", "dinov2s_grid_gru", "dinov2s_grid_d555_l1_k10_gru", "rgbdacal_k10_gru")
$models = if ($Models.Count) { $Models -split "," | Where-Object { $_ } } else { $default }
foreach ($seed in $Seeds) {
    foreach ($m in $models) {
        $out = "outputs/e5_${m}_s$seed"
        if (Test-Path "D:\go2wm\$out\best.pt") { Write-Output "[queue] skip $out"; continue }
        Write-Output "[queue] $(Get-Date -Format s) start $out"
        cmd /c "`"$py`" scripts\train.py --config configs\world_model_e5_$m.yaml --data data/go2_e4c_train --out $out --seed $seed > $LogDir\e5_${m}_s$seed.log 2>&1"
        Write-Output "[queue] $(Get-Date -Format s) end $out exit=$LASTEXITCODE"
    }
}
