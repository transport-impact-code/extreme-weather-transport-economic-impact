param([string]$Config = "configs/default.json")
$ErrorActionPreference = "Stop"
foreach ($stage in @("label", "samples", "train", "analyze")) {
    python -m traffic_prediction $stage --config $Config
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $stage" }
}
