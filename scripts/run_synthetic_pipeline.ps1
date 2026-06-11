# run_synthetic_pipeline.ps1
# End-to-end pipeline run using the full-scale synthetic dataset (2,000 studies).
# No MIMIC access required.
#
# Steps:
#   1. Owner 1  -- build manifest from synthetic data
#   2. Owner 3  -- generate mock model predictions
#   3. Owner 2  -- score raw predictions
#   4. Owner 5  -- apply defer gate
#   5. Owner 2  -- score gated predictions
#   6. Owner 4  -- apply mock training fix
#   7. Owner 2  -- score fixed predictions
#   8. Owner 6  -- aggregate summary
#
# Usage (from the project root or any location):
#   .\scripts\run_synthetic_pipeline.ps1

$ErrorActionPreference = "Stop"

$Repo   = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = "python"

# Output directories
$Owner1Out  = Join-Path $Repo "owner1\outputs_synthetic"
$Owner2Out  = Join-Path $Repo "owner2\outputs_synthetic"
$Owner3Out  = Join-Path $Repo "owner3\outputs_synthetic"
$Owner4Out  = Join-Path $Repo "owner4\outputs_synthetic"
$Owner5Out  = Join-Path $Repo "owner5\outputs_synthetic"
$Owner6Out  = Join-Path $Repo "owner6\outputs_synthetic"

foreach ($dir in @($Owner2Out, $Owner3Out, $Owner4Out, $Owner5Out, $Owner6Out)) {
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
}

# ── Step 1: Owner 1 — build manifest ─────────────────────────────────────────
Write-Host "`n[1/8] Owner 1: building synthetic manifest"
$env:PYTHONPATH = Join-Path $Repo "owner1\src"
Push-Location (Join-Path $Repo "owner1")
& $Python -m owner1_pipeline.cli --config config.synthetic.toml
Pop-Location

$Manifest = Join-Path $Owner1Out "kernel_480.csv"
if (-not (Test-Path $Manifest)) {
    Write-Error "Manifest not found at $Manifest. Owner 1 step failed."
    exit 1
}

# ── Step 2: Owner 3 — mock model predictions ──────────────────────────────────
Write-Host "`n[2/8] Owner 3: generating mock MedGemma-27B predictions"
$env:PYTHONPATH = Join-Path $Repo "owner3\src"
$RawPreds = Join-Path $Owner3Out "mock_medgemma_27b_predictions.csv"
& $Python -m owner3_models.cli `
    --manifest $Manifest `
    --model    mock-medgemma-27b `
    --output   $RawPreds

# ── Step 3: Owner 2 — score raw predictions ───────────────────────────────────
Write-Host "`n[3/8] Owner 2: scoring raw model predictions"
$env:PYTHONPATH = Join-Path $Repo "owner2\src"
$RawMetrics = Join-Path $Owner2Out "raw_metrics.json"
& $Python -m owner2_benchmark.cli `
    --manifest    $Manifest `
    --predictions $RawPreds `
    --output      $RawMetrics

# ── Step 4: Owner 5 — apply defer gate ────────────────────────────────────────
Write-Host "`n[4/8] Owner 5: applying defer-on-disagree gate"
$env:PYTHONPATH = Join-Path $Repo "owner5\src"
$GatePreds = Join-Path $Owner5Out "defer_gate_predictions.csv"
& $Python -m owner5_inference.cli `
    --predictions $RawPreds `
    --strategy    defer-on-disagree `
    --output      $GatePreds

# ── Step 5: Owner 2 — score gated predictions ────────────────────────────────
Write-Host "`n[5/8] Owner 2: scoring gated predictions"
$env:PYTHONPATH = Join-Path $Repo "owner2\src"
$GateMetrics = Join-Path $Owner2Out "gate_metrics.json"
& $Python -m owner2_benchmark.cli `
    --manifest    $Manifest `
    --predictions $GatePreds `
    --output      $GateMetrics

# ── Step 6: Owner 4 — apply mock training fix ────────────────────────────────
Write-Host "`n[6/8] Owner 4: applying conflict-aware LoRA mock fix"
$env:PYTHONPATH = Join-Path $Repo "owner4\src"
$LoRAPreds = Join-Path $Owner4Out "lora_predictions.csv"
& $Python -m owner4_training.cli `
    --predictions $RawPreds `
    --strategy    conflict-aware-lora-mock `
    --output      $LoRAPreds

# ── Step 7: Owner 2 — score fixed predictions ────────────────────────────────
Write-Host "`n[7/8] Owner 2: scoring LoRA-fixed predictions"
$env:PYTHONPATH = Join-Path $Repo "owner2\src"
$LoRAMetrics = Join-Path $Owner2Out "lora_metrics.json"
& $Python -m owner2_benchmark.cli `
    --manifest    $Manifest `
    --predictions $LoRAPreds `
    --output      $LoRAMetrics

# ── Step 8: Owner 6 — aggregate summary ──────────────────────────────────────
Write-Host "`n[8/8] Owner 6: aggregating scored metrics into summary"
$env:PYTHONPATH = Join-Path $Repo "owner6\src"
$Summary = Join-Path $Owner6Out "summary.md"
& $Python -m owner6_rigor.cli `
    --metrics $RawMetrics $GateMetrics $LoRAMetrics `
    --output  $Summary

Write-Host "`nSynthetic pipeline complete."
Write-Host "  Manifest      : $Manifest"
Write-Host "  Raw metrics   : $RawMetrics"
Write-Host "  Gate metrics  : $GateMetrics"
Write-Host "  LoRA metrics  : $LoRAMetrics"
Write-Host "  Summary       : $Summary"
