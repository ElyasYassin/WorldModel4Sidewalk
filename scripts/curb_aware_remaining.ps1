# Remaining curb-aware demo-street runs (the 2026-11-15 batch stopped after run 1). One Isaac process per policy, so
# every run is saved before the next starts. Skips runs whose output JSON already exists.
#   powershell -File scripts\curb_aware_remaining.ps1
param([string]$Ckpt = "D:/go2wm/ext/go2w_runs/rough/model_1999.pt")
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src"
$out = "D:\go2wm\outputs\curb_aware"
$log = "$out\pipeline.log"
function Run($desc, $result, $cmd) {
    if (Test-Path "$out\$result") { "[$(Get-Date -Format HH:mm:ss)] SKIP  $desc ($result exists)" | Out-File -Append -Encoding utf8 $log; return }
    "[$(Get-Date -Format HH:mm:ss)] START $desc" | Out-File -Append -Encoding utf8 $log
    cmd /c "$cmd >> $out\pipeline_detail.log 2>&1"
    "[$(Get-Date -Format HH:mm:ss)] END   $desc (exit $LASTEXITCODE)" | Out-File -Append -Encoding utf8 $log
}
$W = "--config configs/isaac_go2w_rough.yaml --checkpoint $Ckpt"
$DM = "--tiles D:/go2wm/outputs/demo_mission/tiles.json --tile 0 --profile realistic --pose-errors --robust --curb-aware --max-steps 3000 --obstacle-slots 28 --obstacle-radius 0.12"
Run "go2w demo approach+posture" "approach_go2w_posture_curb_tile0.json" "scripts\isaac_python.bat scripts\isaac_mission.py $W $DM --policies approach --posture 0.05 0.10 --dyn-d405 --tag _go2w_posture_curb --out $out"
Run "go2w demo walkover" "walkover_go2w_curb_tile0.json" "scripts\isaac_python.bat scripts\isaac_mission.py $W $DM --policies walkover --dyn-d405 --tag _go2w_curb --out $out"
Run "go2 demo approach" "approach_go2_curb_tile0.json" "scripts\isaac_python.bat scripts\isaac_mission.py $DM --policies approach --tag _go2_curb --out $out"
Run "go2 demo walkover" "walkover_go2_curb_tile0.json" "scripts\isaac_python.bat scripts\isaac_mission.py $DM --policies walkover --tag _go2_curb --out $out"
"[$(Get-Date -Format HH:mm:ss)] REMAINING DONE" | Out-File -Append -Encoding utf8 $log
