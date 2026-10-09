$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

$env:CUDA_VISIBLE_DEVICES = ""
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:TORCH_NUM_THREADS = "1"

$outputDir = ".\reports\trial_user_images\final_mixed_dashboard"
New-Item -ItemType Directory -Force $outputDir | Out-Null

Get-ChildItem ".\input_uji_coba" -Filter "ikan*.jpg" |
    Sort-Object Name |
    ForEach-Object {
        $output = Join-Path $outputDir ($_.BaseName + "_dashboard.jpg")
        Write-Host ("Menjalankan {0}" -f $_.Name)
        & ".\uji_satu.ps1" -Image $_.FullName -Output $output
        if ($LASTEXITCODE -ne 0) {
            throw "Uji gagal untuk $($_.Name), exit code $LASTEXITCODE"
        }
    }

Write-Host "Semua dashboard tersimpan di $outputDir"
