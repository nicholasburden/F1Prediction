#!/usr/bin/env bash
# Run training experiments across model types and history lookback windows.
# Results go to runs/<model>_lb<lookback>/
set -euo pipefail

YEARS="2020 2021 2022 2023 2024"
EPOCHS=100
PATIENCE=15

MODELS=("linear" "single_layer_mlp" "mlp")
LOOKBACKS=(3 5 10)

# Also run a deep MLP variant
DEEP_HIDDEN="256 128 64"

for lb in "${LOOKBACKS[@]}"; do
    for model in "${MODELS[@]}"; do
        outdir="runs/${model}_lb${lb}"
        echo "========================================"
        echo "Training: model=${model} lookback=${lb} -> ${outdir}"
        echo "========================================"
        uv run python scripts/train.py \
            --years $YEARS \
            --model "$model" \
            --lookback "$lb" \
            --epochs "$EPOCHS" \
            --patience "$PATIENCE" \
            --output-dir "$outdir"
    done

    # Deep MLP (wider + deeper)
    outdir="runs/deep_mlp_lb${lb}"
    echo "========================================"
    echo "Training: model=mlp (deep) lookback=${lb} -> ${outdir}"
    echo "========================================"
    uv run python scripts/train.py \
        --years $YEARS \
        --model mlp \
        --hidden-dims $DEEP_HIDDEN \
        --lookback "$lb" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --output-dir "$outdir"
done

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================"
echo ""

# Print comparison table
echo "Model                    | Lookback | Val MAE  | Test MAE | Test RMSE | Top-3 Acc | Params"
echo "-------------------------|----------|----------|----------|-----------|-----------|-------"
for lb in "${LOOKBACKS[@]}"; do
    for model in "${MODELS[@]}" "deep_mlp"; do
        outdir="runs/${model}_lb${lb}"
        results="${outdir}/results.json"
        if [ -f "$results" ]; then
            printf "%-24s | %8d | " "$model" "$lb"
            python3 -c "
import json
r = json.load(open('$results'))
print(f\"{r['val_mae']:8.2f} | {r['test_mae']:8.2f} | {r['test_rmse']:9.2f} | {r['test_top3_acc']:9.2f} | {r['n_params']}\")
"
        fi
    done
done
