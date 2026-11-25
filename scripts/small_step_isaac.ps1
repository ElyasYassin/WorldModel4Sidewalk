# Isaac demo street, Go2-W stop & look + crouch, curb-aware + robust + the transverse step detector (one process).
param([string]$Ckpt = "D:/go2wm/ext/go2w_runs/rough/model_1999.pt")
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "src"
$out = "D:\go2wm\outputs\small_step"
New-Item -ItemType Directory -Force $out | Out-Null
$W = "--config configs/isaac_go2w_rough.yaml --checkpoint $Ckpt"
$DM = "--tiles D:/go2wm/outputs/demo_mission/tiles.json --tile 0 --profile realistic --pose-errors --robust --curb-aware --step-thresh 0.006 --max-steps 3000 --obstacle-slots 28 --obstacle-radius 0.12"
"[$(Get-Date -Format HH:mm:ss)] START go2w posture curb step" | Out-File -Append -Encoding utf8 "$out\pipeline.log"
cmd /c "scripts\isaac_python.bat scripts\isaac_mission.py $W $DM --policies approach --posture 0.05 0.10 --dyn-d405 --tag _go2w_posture_curb_step --out $out >> $out\pipeline_detail.log 2>&1"
"[$(Get-Date -Format HH:mm:ss)] END (exit $LASTEXITCODE)" | Out-File -Append -Encoding utf8 "$out\pipeline.log"
