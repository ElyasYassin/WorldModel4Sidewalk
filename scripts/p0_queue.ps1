# P0 Isaac job queue (2026-11-09). Sequential: one Isaac process at a time (memory).
# Each step saves incrementally, so a killed step loses at most one episode.
# Usage: powershell -File scripts\p0_queue.ps1 -LogDir D:\go2wm\logs\p0 [-Phase data|tune|final]
param([string]$LogDir = "D:\go2wm\logs\p0", [string]$Phase = "data_tune")
New-Item -ItemType Directory -Force $LogDir | Out-Null
$ck = "outputs/state_go2_mlp/best.pt"
$data = @(
  @("collect_final_test", "scripts\collect.py --backend isaac --episodes 60 --seed 30000 --role final_test --out data/go2_final_test"),
  @("collect_cmdpattern", "scripts\collect.py --backend isaac --episodes 80 --seed 40000 --commands heldout_patterns --scene-family open --no-goal-stop --role cmdpattern_test --out data/go2_cmdpattern")
)
$tune = @()
foreach ($c in Get-ChildItem configs\tuning\conv_*.yaml | Sort-Object Name) {
  $n = $c.BaseName
  $tune += ,@("tune_$n", "scripts\run_navigation.py --backend isaac --planner conventional --config configs/tuning/$n.yaml --per-family 17 --out outputs/p0_tuning/$n")
}
# E3f confirmatory run on frozen final-test scenes 17-33 (configs/eval_protocol.yaml).
$fin = "--per-family 17 --first-index 17 --final-test"
$final = @(
  @("e3f_conv_default", "scripts\run_navigation.py --backend isaac --planner conventional --config configs/planner_isaac.yaml $fin --out outputs/e3f/conventional_default"),
  @("e3f_conv_tuned",   "scripts\run_navigation.py --backend isaac --planner conventional --config configs/planner_isaac_conv_tuned.yaml $fin --out outputs/e3f/conventional_tuned"),
  @("e3f_kin",          "scripts\run_navigation.py --backend isaac --planner kinematic_mpc --config configs/planner_isaac_m15_all.yaml $fin --out outputs/e3f/kinematic_mpc"),
  @("e3f_learned",      "scripts\run_navigation.py --backend isaac --planner learned_mpc --checkpoint $ck --config configs/planner_isaac_m15_all.yaml $fin --out outputs/e3f/learned_mpc")
)
$steps = @()
if ($Phase -match "data")  { $steps += $data }
if ($Phase -match "tune")  { $steps += $tune }
if ($Phase -match "final") { $steps += $final }
foreach ($s in $steps) {
  $name, $cmd = $s
  $free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory / 1MB, 1)
  Write-Output "[queue] $(Get-Date -Format s) start $name (free commit ${free} GB)"
  cmd /c "scripts\isaac_python.bat $cmd > `"$LogDir\$name.log`" 2>&1"
  Write-Output "[queue] $(Get-Date -Format s) end $name exit=$LASTEXITCODE"
}
Write-Output "[queue] all done"
