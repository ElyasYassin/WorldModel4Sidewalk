# Phase 1 overnight queue (offline, no Isaac: works with the broken 610.88 driver).
# Steps run in order; each one resumes / skips work already on disk, so the queue can simply be restarted.
# Keeps Windows awake only while it runs (no power settings are changed).
#   powershell -ExecutionPolicy Bypass -File scripts\phase1_overnight.ps1
param([string]$LogDir = "D:/go2wm/outputs/overnight_logs")
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force $LogDir | Out-Null
$py = "C:\Users\Elyas\Anaconda3\envs\IsaacLab\python.exe"

Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$steps = @(
  # 1. crack measurement, corrected skeleton estimator (~25 min)
  @("1_crack_study",    "outputs/crack_study/crack_study.md", "scripts/crack_measurement_study.py --trials 8 --out outputs/crack_study"),
  # 2. strategy comparison on the 6 development tiles (~45 min)
  @("2_missions_dev",   "outputs/missions/comparison.md", "scripts/run_mission.py --use 0 1 2 3 4 5 --policies walkover slow approach --out outputs/missions"),
  # 3. 24 new held-out defect tiles (different seed, ~2 min)
  @("3_eval_tiles",     "outputs/defect_tiles_eval/tiles.json", "scripts/defect_tiles.py --n 24 --seed 100 --out outputs/defect_tiles_eval"),
  # 4. strategy comparison on held-out tiles, realistic sensor errors (~3 h)
  @("4_missions_eval",  "outputs/missions_eval/comparison.md", "scripts/run_mission.py --tiles outputs/defect_tiles_eval/tiles.json --use $(0..23 -join ' ') --policies walkover slow approach --out outputs/missions_eval"),
  # 5. same, pessimistic sensor errors (robustness, ~3 h)
  @("5_missions_pess",  "outputs/missions_eval_pess/comparison.md", "scripts/run_mission.py --tiles outputs/defect_tiles_eval/tiles.json --use $(0..23 -join ' ') --policies walkover slow approach --profile pessimistic --out outputs/missions_eval_pess"),
  # 6. crack study with 3x the trials, for tighter numbers (~75 min)
  @("6_crack_study_24", "outputs/crack_study_24/crack_study.md", "scripts/crack_measurement_study.py --trials 24 --out outputs/crack_study_24")
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
