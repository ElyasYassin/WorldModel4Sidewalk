# RTX demo video of the Go2-W curb-aware stop & look mission (timeline -> Isaac Sim 6.0.1 RTX frames -> HUD video).
$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src"
$M = "D:/go2wm/outputs/curb_aware/approach_go2w_curb_tile0.json"
$TL = "D:/go2wm/outputs/curb_aware/timeline_go2w_curb.json"
$FR = "D:/go2wm/outputs/curb_aware/rtx_frames"
$OUT = "D:/go2wm/outputs/videos/19_demo_mission_go2w_curb_isaac_rtx.mp4"
$R = "D:/go2wm/render_env/Scripts/python.exe"
$T = "Autonomous sidewalk inspection mission: Unitree Go2-W (Isaac Lab physics, curb-aware perception)"
$log = "D:\go2wm\outputs\curb_aware\video_render.log"
"[$(Get-Date -Format HH:mm:ss)] timeline" | Out-File -Append -Encoding utf8 $log
cmd /c "$R scripts/render_demo_mission.py --mission $M --meshes D:/go2wm/outputs/go2w_meshes.npz --timeline-out $TL --title `"$T`" --out $OUT >> $log 2>&1"
"[$(Get-Date -Format HH:mm:ss)] rtx (exit $LASTEXITCODE)" | Out-File -Append -Encoding utf8 $log
cmd /c "D:\Downloads\isaac-sim-standalone-6.0.1-windows-x86_64\python.bat scripts\isaac_rtx_replay.py --mission $M --timeline $TL --out-dir $FR --robot-usd D:/go2wm/cache/go2w_usd/go2w_description.usd >> $log 2>&1"
"[$(Get-Date -Format HH:mm:ss)] compose (exit $LASTEXITCODE)" | Out-File -Append -Encoding utf8 $log
cmd /c "$R scripts/render_demo_mission.py --mission $M --meshes D:/go2wm/outputs/go2w_meshes.npz --view-dir $FR --title `"$T`" --out $OUT >> $log 2>&1"
"[$(Get-Date -Format HH:mm:ss)] DONE (exit $LASTEXITCODE)" | Out-File -Append -Encoding utf8 $log
