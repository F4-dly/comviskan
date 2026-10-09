param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Image,
    [string]$Output = "",
    [string]$FishWeights = ""
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$pipeline = Join-Path $root "all_in_one_yolocomvis.py"
$imagePath = (Resolve-Path $Image -ErrorAction Stop).Path

if ([string]::IsNullOrWhiteSpace($Output)) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($imagePath)
    $Output = Join-Path $root ("reports\uji_satu\" + $stem + "_dashboard.jpg")
}

$arguments = @("--mode", "dashboard", "--image", $imagePath, "--output", $Output)
if (-not [string]::IsNullOrWhiteSpace($FishWeights)) {
    $weightsPath = (Resolve-Path $FishWeights -ErrorAction Stop).Path
    $arguments += @("--fish-weights", $weightsPath)
}

& $python $pipeline @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Dashboard: $Output"
