# Phase 1 final evaluation v2 (perception frozen after the multi-view / hole-wall rules, 2026-11-12): fresh seed-300 tiles.
# failures (now a second dev set), so the final numbers come from 24 fresh tiles (seed 300) not looked at before.
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
  @("F1_final_tiles",   "outputs/defect_tiles_final2/tiles.json", "scripts/defect_tiles.py --n 24 --seed 300 --out outputs/defect_tiles_final2"),
  @("F2_missions_final2", "outputs/missions_final2/comparison.md", "scripts/run_mission.py --tiles outputs/defect_tiles_final2/tiles.json --use $all $pol --out outputs/missions_final2"),
  @("F3_reports_final", "outputs/missions_final2/reports/approach_tile23.html", "scripts/inspection_report.py --missions outputs/missions_final2 --tiles outputs/defect_tiles_final2/tiles.json"),
  @("F6_missions_final2_pess", "outputs/missions_final2_pess/comparison.md", "scripts/run_mission.py --tiles outputs/defect_tiles_final2/tiles.json --use $all $pol --profile pessimistic --out outputs/missions_final2_pess"),
  # informative only (seed 200 motivated the v2 rules): v2 on the first final set
  @("F7_missions_final_v2code", "outputs/missions_final_v2code/comparison.md", "scripts/run_mission.py --tiles outputs/defect_tiles_final/tiles.json --use $all $pol --out outputs/missions_final_v2code")
)
foreach ($s in $steps) {
  $name, $done, $cmd = $s
  if (Test-Path "D:/go2wm/$done") { Write-Output "[final2] $(Get-Date -Format s) skip $name (done)"; continue }
  Write-Output "[final2] $(Get-Date -Format s) start $name"
  cmd /c "`"$py`" $cmd > `"$LogDir\$name.log`" 2>&1"
  Write-Output "[final2] $(Get-Date -Format s) end $name (exit $LASTEXITCODE)"
}
[W.P]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
Write-Output "[final2] $(Get-Date -Format s) all done"
