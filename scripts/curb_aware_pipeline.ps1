# Rerun the demo-street missions with --curb-aware (curb masked in each depth frame + lateral/heading correction
# against the curb), Go2-W and Go2, ONE Isaac process at a time (memory).
#   powershell -File scripts\curb_aware_pipeline.ps1
param([string]$Ckpt = "D:/go2wm/ext/go2w_runs/rough/model_1999.pt")
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src"
$out = "D:\go2wm\outputs\curb_aware"
$log = "$out\pipeline.log"
New-Item -ItemType Directory -Force $out | Out-Null
function Run($desc, $cmd) {
    "[$(Get-Date -Format HH:mm:ss)] START $desc" | Out-File -Append -Encoding utf8 $log
    cmd /c "$cmd >> $out\pipeline_detail.log 2>&1"
    "[$(Get-Date -Format HH:mm:ss)] END   $desc (exit $LASTEXITCODE)" | Out-File -Append -Encoding utf8 $log
}
$W = "--config configs/isaac_go2w_rough.yaml --checkpoint $Ckpt"
$DM = "--tiles D:/go2wm/outputs/demo_mission/tiles.json --tile 0 --profile realistic --pose-errors --robust --curb-aware --max-steps 3000 --obstacle-slots 28 --obstacle-radius 0.12"
Run "go2w demo approach" "scripts\isaac_python.bat scripts\isaac_mission.py $W $DM --policies approach --dyn-d405 --tag _go2w_curb --out $out"
Run "go2w demo approach+posture" "scripts\isaac_python.bat scripts\isaac_mission.py $W $DM --policies approach --posture 0.05 0.10 --dyn-d405 --tag _go2w_posture_curb --out $out"
Run "go2w demo walkover" "scripts\isaac_python.bat scripts\isaac_mission.py $W $DM --policies walkover --dyn-d405 --tag _go2w_curb --out $out"
Run "go2 demo approach+walkover" "scripts\isaac_python.bat scripts\isaac_mission.py $DM --policies approach walkover --tag _go2_curb --out $out"
"[$(Get-Date -Format HH:mm:ss)] PIPELINE DONE" | Out-File -Append -Encoding utf8 $log
