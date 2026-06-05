$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = "C:\Users\pc\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "Owner 1: building mock manifest"
$env:PYTHONPATH = Join-Path $Repo "owner1\src"
Push-Location (Join-Path $Repo "owner1")
& $Python -m owner1_pipeline.cli --config config.mock.toml
Pop-Location

Write-Host "Owner 3: generating mock model predictions"
$env:PYTHONPATH = Join-Path $Repo "owner3\src"
& $Python -m owner3_models.cli `
    --manifest (Join-Path $Repo "owner1\outputs_mock\kernel_manifest.csv") `
    --model mock-medgemma-27b `
    --output (Join-Path $Repo "owner3\outputs_mock\mock_medgemma_27b_predictions.csv")

Write-Host "Owner 2: scoring raw mock model"
$env:PYTHONPATH = Join-Path $Repo "owner2\src"
& $Python -m owner2_benchmark.cli `
    --manifest (Join-Path $Repo "owner1\outputs_mock\kernel_manifest.csv") `
    --predictions (Join-Path $Repo "owner3\outputs_mock\mock_medgemma_27b_predictions.csv") `
    --output (Join-Path $Repo "owner2\outputs_mock\owner3_raw_metrics.json")

Write-Host "Owner 5: applying defer gate"
$env:PYTHONPATH = Join-Path $Repo "owner5\src"
& $Python -m owner5_inference.cli `
    --predictions (Join-Path $Repo "owner3\outputs_mock\mock_medgemma_27b_predictions.csv") `
    --strategy defer-on-disagree `
    --output (Join-Path $Repo "owner5\outputs_mock\defer_gate_predictions.csv")

Write-Host "Owner 2: scoring Owner 5 gate"
$env:PYTHONPATH = Join-Path $Repo "owner2\src"
& $Python -m owner2_benchmark.cli `
    --manifest (Join-Path $Repo "owner1\outputs_mock\kernel_manifest.csv") `
    --predictions (Join-Path $Repo "owner5\outputs_mock\defer_gate_predictions.csv") `
    --output (Join-Path $Repo "owner2\outputs_mock\owner5_gate_metrics.json")

Write-Host "Owner 4: applying mock training fix"
$env:PYTHONPATH = Join-Path $Repo "owner4\src"
& $Python -m owner4_training.cli `
    --predictions (Join-Path $Repo "owner3\outputs_mock\mock_medgemma_27b_predictions.csv") `
    --strategy conflict-aware-lora-mock `
    --output (Join-Path $Repo "owner4\outputs_mock\mock_lora_predictions.csv")

Write-Host "Owner 2: scoring Owner 4 mock fix"
$env:PYTHONPATH = Join-Path $Repo "owner2\src"
& $Python -m owner2_benchmark.cli `
    --manifest (Join-Path $Repo "owner1\outputs_mock\kernel_manifest.csv") `
    --predictions (Join-Path $Repo "owner4\outputs_mock\mock_lora_predictions.csv") `
    --output (Join-Path $Repo "owner2\outputs_mock\owner4_lora_metrics.json")

Write-Host "Owner 6: aggregating scored metrics"
$env:PYTHONPATH = Join-Path $Repo "owner6\src"
& $Python -m owner6_rigor.cli `
    --metrics `
        (Join-Path $Repo "owner2\outputs_mock\owner3_raw_metrics.json") `
        (Join-Path $Repo "owner2\outputs_mock\owner5_gate_metrics.json") `
        (Join-Path $Repo "owner2\outputs_mock\owner4_lora_metrics.json") `
    --output (Join-Path $Repo "owner6\outputs_mock\summary.md")

Write-Host "Mock pipeline complete."
