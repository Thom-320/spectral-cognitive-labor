#!/usr/bin/env bash
set -euo pipefail

# Ejecuta el pipeline reproducible completo sin menu interactivo.
# Uso: ./scripts/run_all.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export MPLBACKEND=Agg

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
elif [[ -d "venv" ]]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
else
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
  pip install -r requirements.txt
fi

run_step() {
  local label="$1"
  local script="$2"
  printf '\n[%s] %s\n' "$label" "$script"
  python "$script"
}

run_step "1/8 Grafo base" "src/00_spectral_grid.py"
run_step "2/8 Diada ejemplo" "src/01_single_dyad_analysis.py"
run_step "3/8 Analisis primario" "src/02_full_comparison.py"
run_step "4/8 Dinamica temporal" "src/04_temporal_dynamics.py"
run_step "5/8 Contraejemplo rectangular" "src/05_counterexample_P6xP8.py"
run_step "6/8 Robustez de particiones" "src/06_partition_robustness.py"
run_step "7/8 Metricas informacionales" "src/07_entropy_analysis.py"
run_step "8/8 Prediccion temprana" "src/08_early_prediction.py"

printf '\nPipeline completo ejecutado correctamente.\n'
