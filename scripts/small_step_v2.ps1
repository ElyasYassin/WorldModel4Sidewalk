# Step detector v2 (cap 20 mm, skip steps within 0.6 m of any residual detection in the frame), frozen after the
# seed-500 check exposed misplaced edges next to large step-down faults (seed 500 is now used). Dev check on seed 100
# (off arm reused from small_step_dev.ps1), then a fresh set: 24 new tiles, seed 600, off vs v2.
param([string]$LogDir = "D:/go2wm/outputs/overnight_logs", [int]$Workers = 6)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = "C:\Users\Elyas\Anaconda3\envs\IsaacLab\python.exe"
$all = "$(0..23 -join ' ')"
$v2 = "--step-thresh 0.006 --step-max 0.02 --step-clear 0.6"
$dev = "scripts/run_mission.py --tiles outputs/defect_tiles_eval/tiles.json --use $all --policies walkover approach --workers $Workers"
$fr = "scripts/run_mission.py --tiles outputs/defect_tiles_final5/tiles.json --use $all --policies walkover approach --workers $Workers"
$steps = @(
  @("V1_dev_v2",       "outputs/dev_step/v2/comparison.md", "$dev $v2 --out outputs/dev_step/v2"),
  @("V2_final5_tiles", "outputs/defect_tiles_final5/tiles.json", "scripts/defect_tiles.py --n 24 --seed 600 --out outputs/defect_tiles_final5"),
  @("V3_final5_off",   "outputs/missions_final5_off/comparison.md", "$fr --out outputs/missions_final5_off"),
  @("V4_final5_v2",    "outputs/missions_final5_v2/comparison.md", "$fr $v2 --out outputs/missions_final5_v2")
)
foreach ($s in $steps) {
  $name, $done, $cmd = $s
  if (Test-Path "D:/go2wm/$done") { Write-Output "[v2] skip $name"; continue }
  Write-Output "[v2] $(Get-Date -Format s) start $name"
  cmd /c "`"$py`" $cmd > `"$LogDir\$name.log`" 2>&1"
  Write-Output "[v2] $(Get-Date -Format s) end $name (exit $LASTEXITCODE)"
}
