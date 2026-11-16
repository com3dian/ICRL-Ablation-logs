#!/usr/bin/env bash
# Submit from the repository root:
#   sbatch jobs/sciworld_luna_normal.sbatch
# Small connectivity test (makes model API calls):
#   SAMPLES=1 ROUNDS=1 BOOTSTRAP_ATTEMPTS=0 MAX_ENV_STEPS=3 sbatch jobs/sciworld_luna_normal.sbatch
# Continue an interrupted run with its saved configuration:
#   RESUME_DIR=ablation/outputs/sciworld/gpt6luna/numeric/<run-id> sbatch jobs/sciworld_luna_normal.sbatch
#SBATCH --job-name=sciworld-luna-normal
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-sciworld-luna-%j.out
#SBATCH --error=slurm-sciworld-luna-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_DIR"
if [[ ! -f ablation/run.py ]]; then
    echo "Submit from the repository root or set REPO_DIR to its absolute path." >&2
    exit 1
fi

ENV_PREFIX="${ENV_PREFIX:-$PWD/.conda/ablation}"
if [[ ! -x "$ENV_PREFIX/bin/python" || ! -x "$ENV_PREFIX/bin/java" ]]; then
    echo "Missing Python/Java environment at $ENV_PREFIX; follow the README setup first." >&2
    exit 1
fi
export PATH="$ENV_PREFIX/bin:$PATH"
export JAVA_HOME="$ENV_PREFIX"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
# The full baseline keeps 29 simulator JVMs alive at once.
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--XX:ActiveProcessorCount=1 -Xmx1024m}"
# Setup has already downloaded MiniLM; avoid network refreshes on compute nodes.
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/icrl-matplotlib-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"

if [[ -n "${RESUME_DIR:-}" ]]; then
    # Check that this job cannot accidentally resume another experiment condition.
    "$ENV_PREFIX/bin/python" - "$RESUME_DIR" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads((Path(sys.argv[1]) / "config.json").read_text())
expected = {"task": "sciworld", "provider": "cliproxy", "model_alias": "gpt6luna", "feedback": "numeric"}
if any(config.get(key) != value for key, value in expected.items()) or config.get("reward_polarity", "normal") != "normal":
    raise SystemExit("RESUME_DIR must be a normal numeric ScienceWorld gpt6luna Cliproxy run")
PY
    run_args=(--resume "$RESUME_DIR" --sciworld-resume-mode "${RESUME_MODE:-auto}")
else
    run_args=(
        --task sciworld --provider cliproxy --model gpt6luna
        --feedback numeric --reward-polarity normal
        --samples "${SAMPLES:-29}" --rounds "${ROUNDS:-40}"
        --bootstrap-attempts "${BOOTSTRAP_ATTEMPTS:-2}"
        --max-env-steps "${MAX_ENV_STEPS:-15}"
        --max-tokens "${MAX_TOKENS:-1000}"
    )
fi

# Check local prerequisites without printing credentials or calling the API.
"$ENV_PREFIX/bin/python" - <<'PY'
import os
from pathlib import Path

import nltk
from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env", override=False)
if not os.getenv("CLIPROXY_API_KEY"):
    raise SystemExit("CLIPROXY_API_KEY is missing; set it in the private root .env")
for resource in ("corpora/wordnet.zip", "tokenizers/punkt_tab/english/"):
    nltk.data.find(resource)
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu", local_files_only=True)
print("Local prerequisites ready; credentials were not printed.", flush=True)
PY

printf 'Job %s on %s; repository: %s\n' "${SLURM_JOB_ID:-local}" "$(hostname)" "$PWD"
printf 'Command: python -u -m ablation.run'
printf ' %q' "${run_args[@]}"
printf '\n'
exec "$ENV_PREFIX/bin/python" -u -m ablation.run "${run_args[@]}"
