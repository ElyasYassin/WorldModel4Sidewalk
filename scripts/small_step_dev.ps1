# Dev check of the transverse step detector (--step-thresh) on the seed-100 dev tiles (mock, realistic, no pose
# errors as in the frozen v2 protocol): same code, detector off vs on.
param([string]$LogDir = "D:/go2wm/outputs/overnight_logs", [int]$Workers = 4)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = "C:\Users\Elyas\Anaconda3\envs\IsaacLab\python.exe"
$all = "$(0..23 -join ' ')"
$base = "scripts/run_mission.py --tiles outputs/defect_tiles_eval/tiles.json --use $all --policies walkover approach --workers $Workers"
$steps = @(
  @("S1_dev_nostep", "outputs/dev_step/off/comparison.md", "$base --out outputs/dev_step/off"),
  @("S2_dev_step6",  "outputs/dev_step/on6/comparison.md", "$base --step-thresh 0.006 --out outputs/dev_step/on6")
)
foreach ($s in $steps) {
  $name, $done, $cmd = $s
  if (Test-Path "D:/go2wm/$done") { Write-Output "[step] skip $name"; continue }
  Write-Output "[step] $(Get-Date -Format s) start $name"
  cmd /c "`"$py`" $cmd > `"$LogDir\$name.log`" 2>&1"
  Write-Output "[step] $(Get-Date -Format s) end $name (exit $LASTEXITCODE)"
}
