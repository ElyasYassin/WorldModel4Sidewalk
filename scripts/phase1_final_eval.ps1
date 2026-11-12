# Phase 1 final evaluation with the frozen perception (2026-11-12). The seed-100 tiles were used to diagnose
# failures (now a second dev set), so the final numbers come from 24 fresh tiles (seed 200) not looked at before.
#   powershell -ExecutionPolicy Bypass -File scripts\phase1_final_eval.ps1
param([string]$LogDir = "D:/go2wm/outputs/overnight_logs")
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force $LogDir | Out-Null
$py = "C:\Users\Elyas\Anaconda3\envs\IsaacLab\python.exe"
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null
$all = "$(0..23 -join ' ')"
$pol = "--policies walkover slow approach --workers 10"
$steps = @(
  @("F1_final_tiles",   "outputs/defect_tiles_final/tiles.json", "scripts/defect_tiles.py --n 24 --seed 200 --out outputs/defect_tiles_final"),
  @("F2_missions_final", "outputs/missions_final/comparison.md", "scripts/run_mission.py --tiles outputs/defect_tiles_final/tiles.json --use $all $pol --out outputs/missions_final"),
  @("F3_reports_final", "outputs/missions_final/reports/approach_tile23.html", "scripts/inspection_report.py --missions outputs/missions_final --tiles outputs/defect_tiles_final/tiles.json"),
  @("F4_missions_dev",  "outputs/missions_dev_v2/comparison.md", "scripts/run_mission.py --use 0 1 2 3 4 5 $pol --out outputs/missions_dev_v2"),
  @("F5_missions_dev2", "outputs/missions_dev2_v2/comparison.md", "scripts/run_mission.py --tiles outputs/defect_tiles_eval/tiles.json --use $all $pol --out outputs/missions_dev2_v2"),
  @("F6_missions_final_pess", "outputs/missions_final_pess/comparison.md", "scripts/run_mission.py --tiles outputs/defect_tiles_final/tiles.json --use $all $pol --profile pessimistic --out outputs/missions_final_pess")
)
foreach ($s in $steps) {
  $name, $done, $cmd = $s
  if (Test-Path "D:/go2wm/$done") { Write-Output "[final] $(Get-Date -Format s) skip $name (done)"; continue }
  Write-Output "[final] $(Get-Date -Format s) start $name"
  cmd /c "`"$py`" $cmd > `"$LogDir\$name.log`" 2>&1"
  Write-Output "[final] $(Get-Date -Format s) end $name (exit $LASTEXITCODE)"
}
[W.P]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
Write-Output "[final] $(Get-Date -Format s) all done"
