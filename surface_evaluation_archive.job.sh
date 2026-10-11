#!/usr/bin/env bash
#
# Slurm directives (adjust to your cluster)
#SBATCH --job-name=surf_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --output=surf_eval-%x-%j.out
#SBATCH --error=surf_eval-%x-%j.err
#SBATCH --qos=longrunning
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=a12007396@unet.univie.ac.at
# Array over lines in processed.txt; set on submission (recommended)
## SBATCH --array=1-1

set -euo pipefail

# ----------------------------
# FreeSurfer (for mris_euler_number)
# ----------------------------
module purge || true
module load FreeSurfer 2>/dev/null || true
if [[ -n "${FREESURFER_HOME:-}" && -d "$FREESURFER_HOME" ]]; then
  HAVE_FS=1
else
  echo "WARN: FREESURFER_HOME not set after module load — Euler characteristic will be skipped." >&2
  HAVE_FS=0
fi

# ----------------------------
# User-configurable (hardcoded)
# ----------------------------
# Path to your Singularity/Apptainer image (.sif)
CONTAINER_IMG="/home/cir/swolff/surface_reconstruction/vox2cortex.sif"

# Path to your dataset root (contains sub-XXXX/ses-YYYYMMDD/surf/)
ROOT="/project/fetal/2025_FetalSurface_swolff/2024_FETAL_CONTROL_JT"

# Path to processed.txt listing cases (one per line):
# "sub-XXXX ses-YYYYMMDD" or "sub-XXXX/ses-YYYYMMDD"
PROCESSED="/project/fetal/2025_FetalSurface_swolff/2024_FETAL_CONTROL_JT/processed.txt"

# Path to your repo root (contains evaluate_mesh_pair.py and eval_metrics.py)
REPO="/home/cir/swolff/surface_reconstruction/Vox2Cortex/vox2organ"

# Output CSV (absolute path recommended)
OUT_CSV="/home/cir/swolff/surface_reconstruction/dataset_scores.csv"

# Evaluation options
N_POINTS=10000              # samples per mesh
DEVICE="cuda"                # "cuda" or "cpu"
SELF_INTERSECTIONS="false"   # "true" or "false"

# Container runner and Python entrypoint
# If singularity/apptainer aren’t in PATH, hardcode RUN_CTN to its full path.
RUN_CTN="${RUN_CTN:-singularity}"   # e.g., "/usr/local/bin/singularity" or "apptainer"
PYTHON_BIN="python"                 # interpreter inside the conda env

# Conda inside the container
# Adjust these to match your image layout and env name
CONDA_SH="/opt/conda/etc/profile.d/conda.sh"  # typical location; change if different
CONDA_ENV="torch_1_10"                         # the env you need to activate

# ----------------------------
# Validation
# ----------------------------
if [[ ! -f "${CONTAINER_IMG}" ]]; then
  echo "ERROR: CONTAINER_IMG not found: ${CONTAINER_IMG}" >&2; exit 1
fi
if [[ ! -d "${ROOT}" ]]; then
  echo "ERROR: ROOT dir not found: ${ROOT}" >&2; exit 1
fi
if [[ ! -f "${PROCESSED}" ]]; then
  echo "ERROR: PROCESSED not found: ${PROCESSED}" >&2; exit 1
fi
if [[ ! -f "${REPO}/evaluate_mesh_pair.py" ]]; then
  echo "ERROR: evaluator not found: ${REPO}/evaluate_mesh_pair.py" >&2; exit 1
fi
if ! command -v "${RUN_CTN}" >/dev/null 2>&1; then
  echo "ERROR: Container runner not found: ${RUN_CTN}" >&2; exit 1
fi

# ----------------------------
# Arrays and container flags
# ----------------------------
# Predeclare arrays to be safe with 'set -u'
declare -a NV_FLAG
declare -a binds

# Enable GPU passthrough if using CUDA
NV_FLAG=()
if [[ "${DEVICE}" == "cuda" ]]; then
  NV_FLAG=(--nv)
fi

# Ensure output dir exists and create a lock file path
OUT_CSV_ABS="$(readlink -f "${OUT_CSV}")"
OUT_DIR="$(dirname "${OUT_CSV_ABS}")"
mkdir -p "${OUT_DIR}"
LOCK_FILE="${OUT_CSV_ABS}.lock"

# Bind mounts: keep host paths identical inside container
binds=(
  -B "${ROOT}:${ROOT}"
  -B "${REPO}:${REPO}"
  -B "${OUT_DIR}:${OUT_DIR}"
)

# Make repo importable inside the container via env
export SINGULARITYENV_PYTHONPATH="${REPO}:${PYTHONPATH:-}"

# Temporary workspace for CSV fragments
TMP_DIR="$(mktemp -d -t surf_eval_${SLURM_JOB_ID:-noj}_XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "Starting sequential evaluation of cases in ${PROCESSED}"

# ----------------------------
# Processing loop (sequential)
# ----------------------------
while IFS= read -r line || [[ -n "$line" ]]; do
  # Skip empty/comment lines
  [[ -z "${line// /}" ]] && continue
  [[ "${line}" =~ ^# ]] && continue

  # Accept "sub-XXXX ses-YYYYMMDD" or "sub-XXXX/ses-YYYYMMDD"
  TOK1="$(echo "$line" | awk '{print $1}')"
  TOK2="$(echo "$line" | awk '{print $2}')"
  if [[ -n "${TOK2}" ]]; then
    SUB="${TOK1}"
    SES="${TOK2}"
  else
    SUB="$(echo "${TOK1}" | cut -d'/' -f1)"
    SES="$(echo "${TOK1}" | cut -d'/' -f2)"
  fi

  if [[ -z "${SUB}" || -z "${SES}" ]]; then
    echo "WARN: could not parse line: ${line}"
    continue
  fi

  SURF_DIR="${ROOT}/${SUB}/${SES}/surf"
  if [[ ! -d "${SURF_DIR}" ]]; then
    echo "WARN: surf dir missing: ${SURF_DIR}"
    continue
  fi

  echo "Case: ${SUB} ${SES} — evaluating L/R × pial/white"

  for H in L R; do
    [[ "$H" == "L" ]] && _side="lh" || _side="rh"
    for MAT in pial white; do
      GT="${SURF_DIR}/${SES}.${H}.${MAT}.native.surf.ply"
      PRED="${SURF_DIR}/${SES}.${H}.${MAT}.fsa6.ply"
      LABEL="${SUB}_${SES}_${H}_${MAT}"

      if [[ ! -f "${GT}" ]]; then
        echo "  Missing GT:   ${GT} (skip)"
        continue
      fi
      if [[ ! -f "${PRED}" ]]; then
        echo "  Missing Pred: ${PRED} (skip)"
        continue
      fi

      # Euler characteristic (FreeSurfer mris_euler_number on native FS surface)
      FS_SURF="${SURF_DIR}/${_side}.${MAT}"
      if [[ "${HAVE_FS}" == "1" ]]; then
        if [[ -f "${FS_SURF}" ]]; then
          euler_out="$(mris_euler_number "${FS_SURF}" 2>&1)"
          echo "  Euler(${_side}.${MAT}): ${euler_out}"
        else
          echo "  Euler(${_side}.${MAT}): FS surface not found: ${FS_SURF} — skipping"
        fi
      fi

      echo "  Evaluating ${LABEL}"
      TMP_CSV="${TMP_DIR}/${LABEL}.csv"

      # Optional flag for self intersections (as a string we can append safely)
      EXTRA_FLAG=""
      if [[ "${SELF_INTERSECTIONS}" == "true" ]]; then
        EXTRA_FLAG="--self_intersections"
      fi

      # Build the inner command to run inside the container:
      # source conda, activate env, run the evaluator
      INNER_CMD="source '${CONDA_SH}' && conda activate '${CONDA_ENV}' && \
PYTHONPATH='${REPO}:${PYTHONPATH:-}' ${PYTHON_BIN} '${REPO}/evaluate_mesh_pair.py' \
  --pred '${PRED}' \
  --gt '${GT}' \
  --labels '${LABEL}' \
  --n_points '${N_POINTS}' \
  --device '${DEVICE}' \
  ${EXTRA_FLAG} \
  --out_csv '${TMP_CSV}'"

      # Execute inside the container via bash -lc so 'conda activate' works
      "${RUN_CTN}" exec "${NV_FLAG[@]-}" "${binds[@]-}" "${CONTAINER_IMG}" \
        /bin/bash -lc "${INNER_CMD}"

      # Append to the global CSV with context columns using a lock
      (
        flock -x 200
        if [[ ! -s "${OUT_CSV_ABS}" ]]; then
          echo "Subject,Session,Hemisphere,Surface,PredPath,GTPath,$(head -n1 "${TMP_CSV}")" > "${OUT_CSV_ABS}"
        fi
        tail -n +2 "${TMP_CSV}" | awk -v s="${SUB}" -v e="${SES}" -v h="${H}" -v m="${MAT}" -v pp="${PRED}" -v gp="${GT}" \
          'BEGIN{FS=OFS=","} {print s,e,h,m,pp,gp,$0}' >> "${OUT_CSV_ABS}"
      ) 200>"${LOCK_FILE}"

    done
  done

done < "${PROCESSED}"

echo "All cases processed. CSV: ${OUT_CSV_ABS}"
