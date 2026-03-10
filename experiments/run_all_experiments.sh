#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_FILE="$ROOT_DIR/experiments/fudge_results.csv"

BATCH_SIZES=(16 64)
EPOCH_VALUES=(1 5 10)

find_last_line() {
    local pattern="$1"
    local file="$2"
    if command -v rg >/dev/null 2>&1; then
        rg -i "$pattern" "$file" | tail -n 1 || true
    else
        grep -Ei "$pattern" "$file" | tail -n 1 || true
    fi
}

extract_first_float_from_tuple() {
    local line="$1"
    local tuple_values
    tuple_values="$(printf '%s\n' "$line" | sed -E 's/^[^:]*:[[:space:]]*\((.*)\)[[:space:]]*$/\1/')"
    printf '%s\n' "$tuple_values" | awk -F',' '
        {
            for (i = 1; i <= NF; i++) {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
                if ($i ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) {
                    print $i
                    exit
                }
            }
        }
    '
}

printf "Batch_Size,Epochs,Privacy_Score,Utility_Score,Security_Score\n" > "$RESULTS_FILE"

for batch_size in "${BATCH_SIZES[@]}"; do
    for epochs in "${EPOCH_VALUES[@]}"; do
        echo "Running experiment with batch_size=$batch_size, epochs=$epochs"
        run_log="$(mktemp)"

        if ! (
            cd "$ROOT_DIR"
            BATCH_SIZE="$batch_size" EPOCHS="$epochs" bash experiments/gather_prelim_data.sh
        ) 2>&1 | tee "$run_log"; then
            echo "Experiment failed for batch_size=$batch_size, epochs=$epochs" >&2
            rm -f "$run_log"
            exit 1
        fi

        privacy_line="$(find_last_line '^privacy score' "$run_log")"
        utility_line="$(find_last_line '^utility score' "$run_log")"
        security_line="$(find_last_line '^security score' "$run_log")"

        if [[ -z "$privacy_line" || -z "$utility_line" || -z "$security_line" ]]; then
            echo "Missing score lines in run output for batch_size=$batch_size, epochs=$epochs" >&2
            rm -f "$run_log"
            exit 1
        fi

        privacy_score="$(extract_first_float_from_tuple "$privacy_line")"
        utility_score="$(extract_first_float_from_tuple "$utility_line")"
        security_score="$(extract_first_float_from_tuple "$security_line")"

        if [[ -z "$privacy_score" || -z "$utility_score" || -z "$security_score" ]]; then
            echo "Failed to parse one or more scores for batch_size=$batch_size, epochs=$epochs" >&2
            rm -f "$run_log"
            exit 1
        fi

        printf "%s,%s,%s,%s,%s\n" \
            "$batch_size" \
            "$epochs" \
            "$privacy_score" \
            "$utility_score" \
            "$security_score" >> "$RESULTS_FILE"

        rm -f "$run_log"
    done
done

echo "Saved experiment results to $RESULTS_FILE"
