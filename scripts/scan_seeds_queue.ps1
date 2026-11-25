# Repeatability on the scanned sidewalk (Trimble X7 tile): Go2-W missions over seeds (pose errors + sensor noise
# differ per seed), one Isaac process per run. Arms: stop & look + crouch with step detector v2 / with the default
# detector only / walk past with v2. Rescored against the scan-measured fault (scripts/rescore_scan_mission.py).
param([int]$Seeds = 10, [string]$Out = "D:/go2wm/outputs/scan_missions/seeds")
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "src"
New-Item -ItemType Directory -Force $Out | Out-Null
$log = "$Out/queue.log"
$W = "--config configs/isaac_go2w_rough.yaml --checkpoint D:/go2wm/ext/go2w_runs/rough/model_1999.pt"
$SCN = "--tiles D:/go2wm/scan/sidewalk_tdx/tiles.json --tile 0 --profile realistic --pose-errors --robust --curb-aware --max-steps 3000 --dyn-d405"
$V2 = "--step-thresh 0.006 --step-max 0.02 --step-clear 0.6"
$arms = @(
    @{tag = "look_v2";   args = "--policies approach --posture 0.05 0.10 $V2"},
    @{tag = "look_base"; args = "--policies approach --posture 0.05 0.10"},
    @{tag = "walk_v2";   args = "--policies walkover $V2"}
)
for ($k = 0; $k -lt $Seeds; $k++) {
    foreach ($arm in $arms) {
        $tag = "_$($arm.tag)_s$k"
        if (Get-ChildItem "$Out/*$tag`_tile0.json" -ErrorAction SilentlyContinue) { continue }
        "[$(Get-Date -Format 'MM-dd HH:mm:ss')] START $tag" | Out-File -Append -Encoding utf8 $log
        cmd /c "scripts\isaac_python.bat scripts\isaac_mission.py $W $SCN $($arm.args) --seed $k --tag $tag --out $Out >> $Out/detail.log 2>&1"
        "[$(Get-Date -Format 'MM-dd HH:mm:ss')] END   $tag (exit $LASTEXITCODE)" | Out-File -Append -Encoding utf8 $log
        $f = Get-ChildItem "$Out/*$tag`_tile0.json" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($f) { cmd /c "python scripts/rescore_scan_mission.py --mission $($f.FullName) >> $Out/rescore.log 2>&1" }
    }
}
"[$(Get-Date -Format 'MM-dd HH:mm:ss')] QUEUE DONE" | Out-File -Append -Encoding utf8 $log
