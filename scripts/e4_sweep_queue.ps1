# E4 terrain sweep queue: one Isaac process per variant of configs/e4_sweep.yaml.
#   powershell -ExecutionPolicy Bypass -File scripts\e4_sweep_queue.ps1 [-Variants flat,slab_10] [-Skip curb_15]
param(
    [string[]]$Variants = @(),
    [string[]]$Skip = @(),
    [string]$LogDir = "D:\go2wm\logs\e4_sweep"
)
$ErrorActionPreference = "Continue"
$Variants = @($Variants | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$Skip = @($Skip | ForEach-Object { $_ -split "," } | Where-Object { $_ })
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force $LogDir | Out-Null
$py = "C:\Users\Elyas\anaconda3\envs\IsaacLab\python.exe"
if ($Variants.Count -eq 0) {
    $Variants = & $py -c "import yaml;print(' '.join(yaml.safe_load(open('configs/e4_sweep.yaml'))['variants']))"
    $Variants = $Variants -split " "
}
foreach ($v in $Variants) {
    if ($Skip -contains $v) { continue }
    $free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory / 1MB, 1)
    Write-Output "[queue] $(Get-Date -Format s) start $v (free commit $free GB)"
    cmd /c "scripts\isaac_python.bat scripts\isaac_terrain_sweep.py --variant $v > $LogDir\$v.log 2>&1"
    Write-Output "[queue] $(Get-Date -Format s) end $v exit=$LASTEXITCODE"
}
Write-Output "[queue] all done"
