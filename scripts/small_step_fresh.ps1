# Fresh-tile check of the frozen transverse step detector: 24 new tiles (seed 500, never looked at), mock, realistic
# profile, no pose errors (the v2 protocol), detector off vs on (--step-thresh 0.006), walkover + approach.
param([string]$LogDir = "D:/go2wm/outputs/overnight_logs", [int]$Workers = 4)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = "C:\Users\Elyas\Anaconda3\envs\IsaacLab\python.exe"
$all = "$(0..23 -join ' ')"
$base = "scripts/run_mission.py --tiles outputs/defect_tiles_final4/tiles.json --use $all --policies walkover approach --workers $Workers"
$steps = @(
  @("T1_final4_tiles", "outputs/defect_tiles_final4/tiles.json", "scripts/defect_tiles.py --n 24 --seed 500 --out outputs/defect_tiles_final4"),
  @("T2_final4_off",   "outputs/missions_final4_off/comparison.md",   "$base --out outputs/missions_final4_off"),
  @("T3_final4_step6", "outputs/missions_final4_step6/comparison.md", "$base --step-thresh 0.006 --out outputs/missions_final4_step6")
)
foreach ($s in $steps) {
  $name, $done, $cmd = $s
  if (Test-Path "D:/go2wm/$done") { Write-Output "[step4] skip $name"; continue }
  Write-Output "[step4] $(Get-Date -Format s) start $name"
  cmd /c "`"$py`" $cmd > `"$LogDir\$name.log`" 2>&1"
  Write-Output "[step4] $(Get-Date -Format s) end $name (exit $LASTEXITCODE)"
}
