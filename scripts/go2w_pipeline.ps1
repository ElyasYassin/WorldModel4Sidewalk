# Go2-W demo pipeline (run after scripts/train_go2w.py), ONE Isaac process at a time (memory).
#   powershell -File scripts\go2w_pipeline.ps1 -Ckpt D:/go2wm/ext/go2w_runs/rough/model_1999.pt
# All missions: realistic sensor errors + odometry drift + camera latency, robust finding estimates.
param([string]$Ckpt = "D:/go2wm/ext/go2w_runs/rough/model_latest.pt", [switch]$SkipGo2)
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src"
$log = "D:\go2wm\outputs\go2w\pipeline.log"
New-Item -ItemType Directory -Force D:\go2wm\outputs\go2w | Out-Null
function Run($desc, $cmd) {
    "[$(Get-Date -Format HH:mm:ss)] START $desc" | Out-File -Append -Encoding utf8 $log
    cmd /c "$cmd >> D:\go2wm\outputs\go2w\pipeline_detail.log 2>&1"
    "[$(Get-Date -Format HH:mm:ss)] END   $desc (exit $LASTEXITCODE)" | Out-File -Append -Encoding utf8 $log
}
$W = "--config configs/isaac_go2w_rough.yaml --checkpoint $Ckpt"
$T8 = "--tiles outputs/defect_tiles_final2/tiles.json --tile 8 --profile realistic --pose-errors --robust"
$DM = "--tiles D:/go2wm/outputs/demo_mission/tiles.json --tile 0 --profile realistic --pose-errors --robust --max-steps 3000 --obstacle-slots 28 --obstacle-radius 0.12"

Run "go2w smoke" "scripts\isaac_python.bat scripts\isaac_smoke.py $W --schedule both --out D:/go2wm/outputs/go2w/smoke_final.json"
Run "go2w tile8 walkover+approach" "scripts\isaac_python.bat scripts\isaac_mission.py $W $T8 --policies walkover approach --dyn-d405 --tag _go2w --out D:/go2wm/outputs/go2w/missions"
Run "go2w tile8 approach+posture" "scripts\isaac_python.bat scripts\isaac_mission.py $W $T8 --policies approach --posture 0.05 0.10 --dyn-d405 --tag _go2w_posture --out D:/go2wm/outputs/go2w/missions"
Run "go2w demo approach+posture" "scripts\isaac_python.bat scripts\isaac_mission.py $W $DM --policies approach --posture 0.05 0.10 --dyn-d405 --tag _go2w_posture --out D:/go2wm/outputs/go2w/demo"
Run "go2w demo walkover+approach" "scripts\isaac_python.bat scripts\isaac_mission.py $W $DM --policies walkover approach --dyn-d405 --tag _go2w --out D:/go2wm/outputs/go2w/demo"
if (-not $SkipGo2) {
    Run "go2 demo approach+walkover (robust)" "scripts\isaac_python.bat scripts\isaac_mission.py $DM --policies approach walkover --tag _robust --out D:/go2wm/outputs/demo_mission/isaac"
    Run "go2 demo approach+posture (robust)" "scripts\isaac_python.bat scripts\isaac_mission.py $DM --policies approach --posture 0.05 0.0 --tag _posture_robust --out D:/go2wm/outputs/demo_mission/isaac"
}
"[$(Get-Date -Format HH:mm:ss)] PIPELINE DONE" | Out-File -Append -Encoding utf8 $log
