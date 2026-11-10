# Sequential Isaac job queue (2026-11-04 overnight). Each step saves results incrementally,
# so a killed step loses at most one episode. Usage: powershell -File scripts\overnight_queue.ps1 <log_dir>
param([string]$LogDir = ".")
$ck = "outputs/state_go2_mlp/best.pt"
$steps = @(
  # e3b_learned finished 2026-11-04 (outputs/navigation_ext_learned)
  @("e3c_kin",     "scripts\run_navigation.py --backend isaac --planner kinematic_mpc --checkpoint $ck --config configs/planner_isaac_db.yaml --per-family 17 --out outputs/navigation_db_kin"),
  @("e3c_learned", "scripts\run_navigation.py --backend isaac --planner learned_mpc --checkpoint $ck --config configs/planner_isaac_db.yaml --per-family 17 --out outputs/navigation_db_learned"),
  @("collect_v2",  "scripts\collect.py --backend isaac --episodes 120 --seed 60 --out data/go2_sim")
)
foreach ($s in $steps) {
  $name, $cmd = $s
  Write-Output "[queue] $(Get-Date -Format s) start $name"
  cmd /c "scripts\isaac_python.bat $cmd > `"$LogDir\q_$name.log`" 2>&1"
  Write-Output "[queue] $(Get-Date -Format s) end $name"
}
Write-Output "[queue] all done"
