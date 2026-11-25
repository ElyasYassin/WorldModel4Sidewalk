# Video 21: Go2-W stop & look + crouch on the scanned sidewalk (Trimble X7 tile from scripts/scan_to_tile.py, photo
# textures from scripts/scan_photo_textures.py). Mission (Isaac Lab, curb-aware + robust + step detector v2) ->
# rescore against the scan-measured fault -> camera timeline -> photoreal RTX frames (Isaac Sim 6.0.1) -> HUD video.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "src"
$D = "D:/go2wm/outputs/scan_missions/stepv2"
$M0 = "$D/approach_go2w_posture_curb_stepv2_scan_tile0.json"
$M = "$D/approach_go2w_posture_curb_stepv2_scan_tile0_rescored.json"
$R = "D:/go2wm/render_env/Scripts/python.exe"
$T = "Go2-W inspection on a real scanned sidewalk (Trimble X7): stop, crouch, measure"
$OUT = "D:/go2wm/outputs/videos/21_scan_sidewalk_go2w_stop_and_look_isaac_rtx.mp4"
New-Item -ItemType Directory -Force $D | Out-Null
cmd /c "scripts\isaac_python.bat scripts\isaac_mission.py --config configs/isaac_go2w_rough.yaml --checkpoint D:/go2wm/ext/go2w_runs/rough/model_1999.pt --tiles D:/go2wm/scan/sidewalk_tdx/tiles.json --tile 0 --profile realistic --pose-errors --robust --curb-aware --step-thresh 0.006 --step-max 0.02 --step-clear 0.6 --max-steps 3000 --policies approach --posture 0.05 0.10 --dyn-d405 --tag _go2w_posture_curb_stepv2_scan --out $D > $D/mission.log 2>&1"
cmd /c "python scripts/rescore_scan_mission.py --mission $M0 > $D/rescore.log 2>&1"
cmd /c "$R scripts/render_demo_mission.py --mission $M --meshes D:/go2wm/outputs/go2w_meshes.npz --timeline-out $D/timeline.json --title `"$T`" --out $D/hud_preview.mp4 > $D/timeline.log 2>&1"
cmd /c "D:\Downloads\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat scripts\isaac_rtx_scan.py --mission $M --timeline $D/timeline.json --out-dir $D/rtx_frames --views fault --width 1344 --height 756 --subframes 16 --robot-usd D:/go2wm/cache/go2w_usd/go2w_description.usd > $D/rtx.log 2>&1"
cmd /c "$R scripts/render_demo_mission.py --mission $M --meshes D:/go2wm/outputs/go2w_meshes.npz --view-dir $D/rtx_frames --title `"$T`" --out $OUT > $D/compose.log 2>&1"
"DONE (exit $LASTEXITCODE)"
