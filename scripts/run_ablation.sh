#!/usr/bin/env bash
# Full ablation study: feature tiers × model types × lookback windows.
# Results go to runs/<tiers>_<model>_lb<lookback>/
set -euo pipefail

YEARS="2020 2021 2022 2023 2024"
EPOCHS=100
PATIENCE=15
LOOKBACK=5  # Fixed lookback for ablation; separate experiments vary lookback

MODELS=("linear" "single_layer_mlp" "mlp")
DEEP_HIDDEN="256 128 64"

# ── Feature tier ablation (fixed lookback=5) ──
# Tier combos: core-only, core+form, all (core+form+derived)
declare -A TIER_LABELS
TIER_LABELS["core"]="core"
TIER_LABELS["core form"]="core_form"
TIER_LABELS["core form derived"]="all"

for tiers in "core" "core form" "core form derived"; do
    label="${TIER_LABELS[$tiers]}"
    for model in "${MODELS[@]}"; do
        outdir="runs/ablation_${label}_${model}_lb${LOOKBACK}"
        echo "========================================"
        echo "Ablation: tiers=${label} model=${model} lookback=${LOOKBACK} -> ${outdir}"
        echo "========================================"
        uv run python scripts/train.py \
            --years $YEARS \
            --model "$model" \
            --lookback "$LOOKBACK" \
            --epochs "$EPOCHS" \
            --patience "$PATIENCE" \
            --feature-tiers $tiers \
            --output-dir "$outdir"
    done

    # Deep MLP for each tier combo
    outdir="runs/ablation_${label}_deep_mlp_lb${LOOKBACK}"
    echo "========================================"
    echo "Ablation: tiers=${label} model=deep_mlp lookback=${LOOKBACK} -> ${outdir}"
    echo "========================================"
    uv run python scripts/train.py \
        --years $YEARS \
        --model mlp \
        --hidden-dims $DEEP_HIDDEN \
        --lookback "$LOOKBACK" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --feature-tiers $tiers \
        --output-dir "$outdir"
done

# ── Lookback sweep (all features, MLP only) ──
for lb in 3 5 10; do
    outdir="runs/ablation_all_mlp_lb${lb}"
    if [ -f "${outdir}/results.json" ]; then
        echo "Skipping existing: ${outdir}"
        continue
    fi
    echo "========================================"
    echo "Lookback sweep: lb=${lb} -> ${outdir}"
    echo "========================================"
    uv run python scripts/train.py \
        --years $YEARS \
        --model mlp \
        --lookback "$lb" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --feature-tiers core form derived \
        --output-dir "$outdir"
done

echo ""
echo "========================================"
echo "ALL ABLATION EXPERIMENTS COMPLETE"
echo "========================================"
