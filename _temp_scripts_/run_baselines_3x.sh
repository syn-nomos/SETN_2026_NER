#!/usr/bin/env bash
# Run ACE baseline (ep.1, all-6-embeddings) 3 times for each of 3 datasets.
# Reuses the SAME model folder (overwrites between runs) but generates a
# unique MD report per run BEFORE the next run starts overwriting.
set -e
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
mkdir -p logs/baselines_3x
mkdir -p RESULTS/GLN/BASELINE RESULTS/INLNER/BASELINE RESULTS/LEGALNERO/BASELINE

declare -A OUTDIR=(
  [gln]=RESULTS/GLN/BASELINE
  [inlner]=RESULTS/INLNER/BASELINE
  [legalnero]=RESULTS/LEGALNERO/BASELINE
)
declare -A UPNAME=(
  [gln]=GLN
  [inlner]=INLNER
  [legalnero]=LEGALNERO
)

for ds in gln inlner legalnero; do
  cfg="config/${ds}_baseline.yaml"
  name="${ds}_baseline"
  taggerdir="resources/taggers/${name}"

  for i in 1 2 3; do
    log="logs/baselines_3x/${name}_run${i}.log"
    md="${OUTDIR[$ds]}/${UPNAME[$ds]}_BASELINE_run${i}.md"

    echo "=========================================="
    echo "[$(date +%H:%M:%S)] $name run${i} (GPU $GPU)"
    echo "=========================================="
    # Wipe previous tagger state so each run starts fresh
    rm -rf "$taggerdir"
    mkdir -p "$taggerdir"

    CUDA_VISIBLE_DEVICES=$GPU python train.py --config "$cfg" 2>&1 | tee "$log"

    echo "[generate_report] -> $md"
    src_log="$taggerdir/training.log"
    [[ -f "$src_log" ]] || src_log="$log"
    python RESULTS/scripts/generate_report.py "$src_log" \
      --config "$cfg" --model "$taggerdir" -o "$md" \
      || echo "WARN: report generation failed for $name run${i}"
  done
done

echo "ALL DONE."
