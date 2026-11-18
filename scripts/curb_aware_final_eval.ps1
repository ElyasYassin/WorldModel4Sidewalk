# Held-out check of --curb-aware and --robust (mock kinematics, realistic profile, odometry drift + camera latency):
# 24 fresh tiles (seed 400, never looked at), perception code frozen at the commit that added this script.
# 4 arms x 2 strategies x 24 tiles. Runs beside the Isaac jobs, so few workers.
#   powershell -ExecutionPolicy Bypass -File scripts\curb_aware_final_eval.ps1
param([string]$LogDir = "D:/go2wm/outputs/overnight_logs", [int]$Workers = 6)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force $LogDir | Out-Null
$py = "C:\Users\Elyas\Anaconda3\envs\IsaacLab\python.exe"
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null
$all = "$(0..23 -join ' ')"
$base = "scripts/run_mission.py --tiles outputs/defect_tiles_final3/tiles.json --use $all --policies walkover approach --pose-errors --workers $Workers"
$steps = @(
  @("C1_final3_tiles", "outputs/defect_tiles_final3/tiles.json", "scripts/defect_tiles.py --n 24 --seed 400 --out outputs/defect_tiles_final3"),
  @("C2_final3_base",   "outputs/missions_final3_base/comparison.md",   "$base --out outputs/missions_final3_base"),
  @("C3_final3_curb",   "outputs/missions_final3_curb/comparison.md",   "$base --curb-aware --out outputs/missions_final3_curb"),
  @("C4_final3_robust", "outputs/missions_final3_robust/comparison.md", "$base --robust --out outputs/missions_final3_robust"),
  @("C5_final3_robust_curb", "outputs/missions_final3_robust_curb/comparison.md", "$base --robust --curb-aware --out outputs/missions_final3_robust_curb")
)
foreach ($s in $steps) {
  $name, $done, $cmd = $s
  if (Test-Path "D:/go2wm/$done") { Write-Output "[final3] $(Get-Date -Format s) skip $name (done)"; continue }
  Write-Output "[final3] $(Get-Date -Format s) start $name"
  cmd /c "`"$py`" $cmd > `"$LogDir\$name.log`" 2>&1"
  Write-Output "[final3] $(Get-Date -Format s) end $name (exit $LASTEXITCODE)"
}
[W.P]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
Write-Output "[final3] $(Get-Date -Format s) all done"
