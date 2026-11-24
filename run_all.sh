#!/bin/bash
# Reproduces every result of the paper from the TITAN embeddings in data/titan/ (about 15 minutes on a 64-core CPU).
#
#   ./run_all.sh            # experiments, then the paper's numbers and tables (results/paper/)
#   ./run_all.sh --data     # also rebuild data/manifest.csv and data/splits.csv first
#
# Uses the active Python environment (conda / venv); with uv: `uv run ./run_all.sh`. Override with PYTHON=/path/to/python.
# Per-script output goes to logs/<script>.log.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTHONWARNINGS=ignore
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # the chord package, without requiring `pip install -e .`
mkdir -p logs

run() {
    echo "[$(date '+%F %T')] $1"
    "$PY" "scripts/$1.py" > "logs/$1.log" 2>&1 || { echo "FAILED: $1 (see logs/$1.log)"; exit 1; }
}

if [ "${1:-}" = "--data" ]; then
    run build_manifest        # data/manifest.csv: cohorts, roles (ID / near-OOD / far-OOD), exclusions
    run make_splits           # data/splits.csv: CPTAC patient-level train / val / dev / test
fi

# 1. Hyperparameter selection on the CPTAC-dev semi-synthetic benchmark (no target cohort used)
run run_semisynth             # Step 5: benchmark scenarios and the baselines' shift-recovery error
run run_method_dev            # Step 6: estimator x n_min x r grid
run summarize_method_dev      #         rule-based choice of estimator, n_min, r
run run_iterations_dev        # Step 7: number of passes T
run run_variants_dev          # Step 7b: shared vs organ-wise vs class-wise shift

# 2. Real cohorts (source: CPTAC; targets: TCGA, PLCO, EBRAINS; control: CPTAC-test)
run run_final                 # CHORD vs no correction: transductive, held-out patients, what CHORD selects
run run_detectors             # ten post-hoc detectors x corrections, bootstrap CIs
run run_mean_of_ten           # mean over the ten detectors (Table 2), CPTAC-dev AUROC per detector
run run_fpr95_ci              # FPR95 and its bootstrap CIs (Table 2)
run run_sensitivity           # ablations and sensitivity (Table 3)
run run_ci_refit              # bootstrap with CHORD refitted on every resample
run run_failure_analysis      # absent-class failure case (Discussion)

# 3. Paper numbers and tables
run make_paper_tables         # results/paper/numbers.tex, results/paper/tables/*.tex
echo "[$(date '+%F %T')] done"
