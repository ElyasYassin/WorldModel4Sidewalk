# Copy E4-C datasets (episodes, RGB, split manifests, derived features) to a server's data drive.
#   powershell -File scripts\server\sync_data.ps1 -User <university_username> [-Server 4dgen.ucdenver.pvt]
# Needs the CU Denver network or VPN. Prompts for your password unless an SSH key is set up.
param([Parameter(Mandatory = $true)][string]$User, [string]$Server = "4dgen.ucdenver.pvt",
      [string[]]$Datasets = @("go2_e4c_train", "go2_e4c_test"), [string]$LocalRoot = "D:\go2wm")
$ErrorActionPreference = "Stop"
$remote = "/home/storage/$User/go2wm/data"
ssh "$User@$Server" "mkdir -p $remote"
foreach ($d in $Datasets) {
    $tar = Join-Path $env:TEMP "$d.tar"
    Write-Output "[sync] packing $d"
    tar -cf $tar -C "$LocalRoot\data" $d
    Write-Output "[sync] uploading $d ($([math]::Round((Get-Item $tar).Length / 1MB)) MB)"
    scp $tar "${User}@${Server}:$remote/"
    ssh "$User@$Server" "cd $remote && tar -xf $d.tar && rm $d.tar"
    Remove-Item $tar
}
Write-Output "[sync] done -> ${Server}:$remote"
