# Phase 1 overnight queue (offline, no Isaac: works with the broken 610.88 driver).
# Steps run in order; each one resumes / skips work already on disk, so the queue can simply be restarted.
# Keeps Windows awake only while it runs (no power settings are changed).
#   powershell -ExecutionPolicy Bypass -File scripts\phase1_overnight.ps1
# 2026-11-11 v2: flying-pixel wrap-around fix and detector / memory fixes -> sensor studies rerun, reports added.
param([string]$LogDir = "D:/go2wm/outputs/overnight_logs")
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force $LogDir | Out-Null
$py = "C:\Users\Elyas\Anaconda3\envs\IsaacLab\python.exe"

Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$eval = "$(0..23 -join ' ')"
$steps = @(
  @("1_crack_study",     "outputs/crack_study/crack_study.md", "scripts/crack_measurement_study.py --trials 8 --out outputs/crack_study"),
  @("2_missions_dev",    "outputs/missions/comparison.md", "scripts/run_mission.py --use 0 1 2 3 4 5 --policies walkover slow approach --workers 10 --out outputs/missions"),
  @("2b_reports_dev",    "outputs/missions/reports/approach_tile5.html", "scripts/inspection_report.py --missions outputs/missions"),
  @("3_eval_tiles",      "outputs/defect_tiles_eval/tiles.json", "scripts/defect_tiles.py --n 24 --seed 100 --out outputs/defect_tiles_eval"),
  @("4_missions_eval",   "outputs/missions_eval/comparison.md", "scripts/run_mission.py --tiles outputs/defect_tiles_eval/tiles.json --use $eval --policies walkover slow approach --workers 10 --out outputs/missions_eval"),
  @("5_sensor_errors",   "outputs/sensor_errors_v2/sensor_error_study.md", "scripts/sensor_error_study.py --episodes 12 --out outputs/sensor_errors_v2"),
  @("6_sensor_ablation", "outputs/sensor_errors_ablation_v2/sensor_error_study.md", "scripts/sensor_error_study.py --episodes 8 --profiles realistic realistic-calib realistic-scale realistic-odom realistic-stereo realistic-lidar --out outputs/sensor_errors_ablation_v2"),
  @("7_near_far",        "outputs/near_far_steps_v2/near_far_steps.md", "scripts/near_far_step_study.py --trials 60 --out outputs/near_far_steps_v2"),
  @("8_missions_pess",   "outputs/missions_eval_pess/comparison.md", "scripts/run_mission.py --tiles outputs/defect_tiles_eval/tiles.json --use $eval --policies walkover slow approach --profile pessimistic --workers 10 --out outputs/missions_eval_pess"),
  @("9_crack_study_24",  "outputs/crack_study_24/crack_study.md", "scripts/crack_measurement_study.py --trials 24 --seed-offset 100 --out outputs/crack_study_24")
)
foreach ($s in $steps) {
  $name, $done, $cmd = $s
  if (Test-Path "D:/go2wm/$done") { Write-Output "[queue] $(Get-Date -Format s) skip $name (done)"; continue }
  Write-Output "[queue] $(Get-Date -Format s) start $name"
  cmd /c "`"$py`" $cmd > `"$LogDir\$name.log`" 2>&1"
  Write-Output "[queue] $(Get-Date -Format s) end $name (exit $LASTEXITCODE)"
}
[W.P]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
Write-Output "[queue] $(Get-Date -Format s) all done"
