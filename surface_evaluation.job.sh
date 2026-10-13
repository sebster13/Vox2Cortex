#!/usr/bin/env bash

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

set -euo pipefail


# -------------------------------------------------------------------------
# Evaluation mode
# -------------------------------------------------------------------------
# Supported values:
#   gt-only
#   pair
#
# gt-only:
#   Evaluates only ground-truth surfaces using the general quality metrics.
#
# pair:
#   Evaluates GT quality, prediction quality, and prediction-versus-GT
#   metrics.
EVAL_MODE="gt-only"


# -------------------------------------------------------------------------
# Container and repository
# -------------------------------------------------------------------------
CONTAINER_IMG="/home/cir/swolff/surface_reconstruction/vox2cortex.sif"

REPO="/home/cir/swolff/surface_reconstruction/Vox2Cortex/vox2organ"

QUALITY_CONFIG="${REPO}/quality_config.json"

RUN_CTN="${RUN_CTN:-singularity}"

PYTHON_BIN="python"

CONDA_SH="/opt/conda/etc/profile.d/conda.sh"

CONDA_ENV="torch_1_10"


# -------------------------------------------------------------------------
# Subject/session list
# -------------------------------------------------------------------------
PROCESSED="/project/fetal/2025_FetalSurface_swolff/2024_FETAL_CONTROL_JT/processed.txt"


# -------------------------------------------------------------------------
# Flexible mesh roots and naming templates
# -------------------------------------------------------------------------
# GT and predictions may have different roots and naming conventions.
#
# The templates can contain:
#   {ROOT}
#   {SUB}
#   {SES}
#   {H}          L or R
#   {HEMI}       L or R
#   {FS_HEMI}    lh or rh
#   {SIDE}       lh or rh
#   {SURFACE}    pial or white
#   {MAT}        pial or white

GT_ROOT="/project/fetal/2025_FetalSurface_swolff/2024_FETAL_CONTROL_JT"

PRED_ROOT="/project/fetal/2025_FetalSurface_swolff/2024_FETAL_CONTROL_JT"

GT_MESH_TEMPLATE="{ROOT}/{SUB}/{SES}/surf/{SES}.{H}.{SURFACE}.fsa6.reg.ply"

PRED_MESH_TEMPLATE="{ROOT}/{SUB}/{SES}/surf/{SES}.{H}.{SURFACE}.fsa6.ply"


# -------------------------------------------------------------------------
# Surfaces to evaluate
# -------------------------------------------------------------------------
HEMISPHERES=(L R)

SURFACES=(pial white)


# -------------------------------------------------------------------------
# Optional segmentation inputs for Dice/IoU
# -------------------------------------------------------------------------
# To use these:
#
# 1. Set USE_SEGMENTATION_FILES="true".
# 2. Change quality_config.json:
#
#      "overlap": {
#        "mode": "segmentation_files",
#        ...
#      }
#
# 3. Configure the templates below.
#
# Prediction and GT segmentations must be in the same voxel grid.
USE_SEGMENTATION_FILES="false"

GT_SEG_ROOT="${GT_ROOT}"

PRED_SEG_ROOT="${PRED_ROOT}"

REFERENCE_IMAGE_ROOT="${GT_ROOT}"

GT_SEG_TEMPLATE="{ROOT}/{SUB}/{SES}/seg/{SES}.{H}.{SURFACE}.gt.nii.gz"

PRED_SEG_TEMPLATE="{ROOT}/{SUB}/{SES}/seg/{SES}.{H}.{SURFACE}.pred.nii.gz"

# Leave empty if no separate reference image is needed.
REFERENCE_IMAGE_TEMPLATE=""

# Leave a label empty to treat all nonzero voxels as foreground.
GT_SEG_LABEL_PIAL=""
GT_SEG_LABEL_WHITE=""

PRED_SEG_LABEL_PIAL=""
PRED_SEG_LABEL_WHITE=""


# -------------------------------------------------------------------------
# Distance evaluation options
# -------------------------------------------------------------------------
N_POINTS=10000

SEED=42

DEVICE="cuda"


# -------------------------------------------------------------------------
# Output
# -------------------------------------------------------------------------
OUT_CSV="/home/cir/swolff/surface_reconstruction/dataset_scores.csv"

# If true, an existing output CSV is deleted before the job starts.
RESET_OUTPUT="false"

# If true, failure of one surface does not terminate evaluation of all
# remaining cases.
CONTINUE_ON_ERROR="true"


# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------
if [[ "${EVAL_MODE}" != "gt-only" && "${EVAL_MODE}" != "pair" ]]; then
  echo "ERROR: EVAL_MODE must be 'gt-only' or 'pair'." >&2
  exit 1
fi

if [[ ! -f "${CONTAINER_IMG}" ]]; then
  echo "ERROR: Container image not found: ${CONTAINER_IMG}" >&2
  exit 1
fi

if [[ ! -d "${REPO}" ]]; then
  echo "ERROR: Repository not found: ${REPO}" >&2
  exit 1
fi

if [[ ! -f "${REPO}/evaluate_mesh_pair.py" ]]; then
  echo "ERROR: Evaluator not found: ${REPO}/evaluate_mesh_pair.py" >&2
  exit 1
fi

if [[ ! -f "${QUALITY_CONFIG}" ]]; then
  echo "ERROR: Quality configuration not found: ${QUALITY_CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${PROCESSED}" ]]; then
  echo "ERROR: Processed list not found: ${PROCESSED}" >&2
  exit 1
fi

if [[ ! -d "${GT_ROOT}" ]]; then
  echo "ERROR: GT root does not exist: ${GT_ROOT}" >&2
  exit 1
fi

if [[ "${EVAL_MODE}" == "pair" && ! -d "${PRED_ROOT}" ]]; then
  echo "ERROR: Prediction root does not exist: ${PRED_ROOT}" >&2
  exit 1
fi

if ! command -v "${RUN_CTN}" >/dev/null 2>&1; then
  echo "ERROR: Container runner not found: ${RUN_CTN}" >&2
  exit 1
fi


# -------------------------------------------------------------------------
# Path-template rendering
# -------------------------------------------------------------------------
render_template() {
  local template="$1"
  local root="$2"
  local sub="$3"
  local ses="$4"
  local hemi="$5"
  local surface="$6"
  local fs_hemi

  if [[ "${hemi}" == "L" ]]; then
    fs_hemi="lh"
  else
    fs_hemi="rh"
  fi

  local path="${template}"

  path="${path//\{ROOT\}/${root}}"
  path="${path//\{SUB\}/${sub}}"
  path="${path//\{SES\}/${ses}}"
  path="${path//\{H\}/${hemi}}"
  path="${path//\{HEMI\}/${hemi}}"
  path="${path//\{FS_HEMI\}/${fs_hemi}}"
  path="${path//\{SIDE\}/${fs_hemi}}"
  path="${path//\{SURFACE\}/${surface}}"
  path="${path//\{MAT\}/${surface}}"

  printf '%s' "${path}"
}


# -------------------------------------------------------------------------
# Container bind mounts
# -------------------------------------------------------------------------
declare -a NV_FLAG
declare -a BINDS
declare -A BOUND_PATHS

NV_FLAG=()

if [[ "${DEVICE}" == "cuda" ]]; then
  NV_FLAG=(--nv)
fi

add_bind() {
  local path="$1"

  if [[ -z "${path}" ]]; then
    return
  fi

  if [[ ! -d "${path}" ]]; then
    echo "WARN: Bind directory does not exist: ${path}" >&2
    return
  fi

  if [[ -z "${BOUND_PATHS[${path}]+x}" ]]; then
    BINDS+=(
      -B "${path}:${path}"
    )
    BOUND_PATHS["${path}"]=1
  fi
}


# -------------------------------------------------------------------------
# Output and temporary workspace
# -------------------------------------------------------------------------
OUT_DIR="$(dirname "${OUT_CSV}")"
mkdir -p "${OUT_DIR}"

OUT_CSV_ABS="$(readlink -m "${OUT_CSV}")"
LOCK_FILE="${OUT_CSV_ABS}.lock"

if [[ "${RESET_OUTPUT}" == "true" ]]; then
  rm -f "${OUT_CSV_ABS}" "${LOCK_FILE}"
fi

TMP_DIR="$(mktemp -d -t "surf_eval_${SLURM_JOB_ID:-noj}_XXXXXX")"

trap 'rm -rf "${TMP_DIR}"' EXIT


# Bind all potentially required host paths.
BINDS=()

add_bind "${GT_ROOT}"
add_bind "${REPO}"
add_bind "${OUT_DIR}"
add_bind "$(dirname "${QUALITY_CONFIG}")"

if [[ "${EVAL_MODE}" == "pair" ]]; then
  add_bind "${PRED_ROOT}"
fi

if [[ "${USE_SEGMENTATION_FILES}" == "true" ]]; then
  add_bind "${GT_SEG_ROOT}"
  add_bind "${PRED_SEG_ROOT}"
  add_bind "${REFERENCE_IMAGE_ROOT}"
fi


# -------------------------------------------------------------------------
# CSV appending
# -------------------------------------------------------------------------
append_csv_safely() {
  local temporary_csv="$1"

  if [[ ! -s "${temporary_csv}" ]]; then
    echo "ERROR: Temporary CSV is missing or empty: ${temporary_csv}" >&2
    return 1
  fi

  (
    flock -x 200

    local new_header
    local existing_header

    new_header="$(head -n 1 "${temporary_csv}" | tr -d '\r')"

    if [[ ! -s "${OUT_CSV_ABS}" ]]; then
      cat "${temporary_csv}" > "${OUT_CSV_ABS}"
    else
      existing_header="$(head -n 1 "${OUT_CSV_ABS}" | tr -d '\r')"

      if [[ "${existing_header}" != "${new_header}" ]]; then
        echo "ERROR: Output CSV schema mismatch." >&2
        echo "Existing header:" >&2
        echo "${existing_header}" >&2
        echo "New header:" >&2
        echo "${new_header}" >&2
        echo "Use a new OUT_CSV or set RESET_OUTPUT=true." >&2
        return 1
      fi

      tail -n +2 "${temporary_csv}" >> "${OUT_CSV_ABS}"
    fi
  ) 200>"${LOCK_FILE}"
}


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------
echo "Evaluation mode: ${EVAL_MODE}"
echo "Cases:           ${PROCESSED}"
echo "Configuration:   ${QUALITY_CONFIG}"
echo "Output:          ${OUT_CSV_ABS}"


while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
  # Remove leading and trailing whitespace.
  line="${raw_line#"${raw_line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"

  # Skip blank lines and full-line comments.
  [[ -z "${line}" ]] && continue
  [[ "${line}" == \#* ]] && continue

  # Remove an optional trailing comment.
  line="${line%%#*}"
  line="${line%"${line##*[![:space:]]}"}"

  [[ -z "${line}" ]] && continue

  token1=""
  token2=""

  read -r token1 token2 _ <<< "${line}"

  if [[ -n "${token2}" ]]; then
    SUB="${token1}"
    SES="${token2}"
  else
    SUB=""
    SES=""

    IFS='/' read -r SUB SES _ <<< "${token1}"
  fi

  if [[ -z "${SUB}" || -z "${SES}" ]]; then
    echo "WARN: Could not parse processed-list line: ${raw_line}" >&2
    continue
  fi

  echo "Case: ${SUB} ${SES}"

  for H in "${HEMISPHERES[@]}"; do
    for SURFACE in "${SURFACES[@]}"; do
      LABEL="${SUB}_${SES}_${H}_${SURFACE}"

      GT_PATH="$(
        render_template \
          "${GT_MESH_TEMPLATE}" \
          "${GT_ROOT}" \
          "${SUB}" \
          "${SES}" \
          "${H}" \
          "${SURFACE}"
      )"

      if [[ ! -f "${GT_PATH}" ]]; then
        echo "  Missing GT: ${GT_PATH} — skipping" >&2
        continue
      fi

      PRED_PATH=""

      if [[ "${EVAL_MODE}" == "pair" ]]; then
        PRED_PATH="$(
          render_template \
            "${PRED_MESH_TEMPLATE}" \
            "${PRED_ROOT}" \
            "${SUB}" \
            "${SES}" \
            "${H}" \
            "${SURFACE}"
        )"

        if [[ ! -f "${PRED_PATH}" ]]; then
          echo "  Missing prediction: ${PRED_PATH} — skipping" >&2
          continue
        fi
      fi

      echo "  Evaluating ${LABEL}"

      TMP_CSV="${TMP_DIR}/${LABEL}.csv"

      declare -a PYTHON_ARGS

      PYTHON_ARGS=(
        "${PYTHON_BIN}"
        "${REPO}/evaluate_mesh_pair.py"
        --mode "${EVAL_MODE}"
        --gt "${GT_PATH}"
        --config "${QUALITY_CONFIG}"
        --subject "${SUB}"
        --session "${SES}"
        --hemisphere "${H}"
        --surface "${SURFACE}"
        --label "${LABEL}"
        --n_points "${N_POINTS}"
        --seed "${SEED}"
        --device "${DEVICE}"
        --out_csv "${TMP_CSV}"
      )

      if [[ "${EVAL_MODE}" == "pair" ]]; then
        PYTHON_ARGS+=(
          --pred "${PRED_PATH}"
        )
      fi

      if [[ "${EVAL_MODE}" == "pair" && "${USE_SEGMENTATION_FILES}" == "true" ]]; then
        GT_SEG_PATH="$(
          render_template \
            "${GT_SEG_TEMPLATE}" \
            "${GT_SEG_ROOT}" \
            "${SUB}" \
            "${SES}" \
            "${H}" \
            "${SURFACE}"
        )"

        PRED_SEG_PATH="$(
          render_template \
            "${PRED_SEG_TEMPLATE}" \
            "${PRED_SEG_ROOT}" \
            "${SUB}" \
            "${SES}" \
            "${H}" \
            "${SURFACE}"
        )"

        PYTHON_ARGS+=(
          --gt_seg "${GT_SEG_PATH}"
          --pred_seg "${PRED_SEG_PATH}"
        )

        if [[ "${SURFACE}" == "pial" ]]; then
          GT_SEG_LABEL="${GT_SEG_LABEL_PIAL}"
          PRED_SEG_LABEL="${PRED_SEG_LABEL_PIAL}"
        else
          GT_SEG_LABEL="${GT_SEG_LABEL_WHITE}"
          PRED_SEG_LABEL="${PRED_SEG_LABEL_WHITE}"
        fi

        if [[ -n "${GT_SEG_LABEL}" ]]; then
          PYTHON_ARGS+=(
            --gt_seg_label "${GT_SEG_LABEL}"
          )
        fi

        if [[ -n "${PRED_SEG_LABEL}" ]]; then
          PYTHON_ARGS+=(
            --pred_seg_label "${PRED_SEG_LABEL}"
          )
        fi

        if [[ -n "${REFERENCE_IMAGE_TEMPLATE}" ]]; then
          REFERENCE_IMAGE_PATH="$(
            render_template \
              "${REFERENCE_IMAGE_TEMPLATE}" \
              "${REFERENCE_IMAGE_ROOT}" \
              "${SUB}" \
              "${SES}" \
              "${H}" \
              "${SURFACE}"
          )"

          PYTHON_ARGS+=(
            --reference_image "${REFERENCE_IMAGE_PATH}"
          )
        fi
      fi

      # Quote all Python arguments safely before passing them through
      # /bin/bash -lc inside the container.
      printf -v PYTHON_COMMAND '%q ' "${PYTHON_ARGS[@]}"

      printf -v QUOTED_CONDA_SH '%q' "${CONDA_SH}"
      printf -v QUOTED_CONDA_ENV '%q' "${CONDA_ENV}"

      CONTAINER_PYTHONPATH="${REPO}:${PYTHONPATH:-}"
      printf -v QUOTED_PYTHONPATH '%q' "${CONTAINER_PYTHONPATH}"

      INNER_COMMAND="source ${QUOTED_CONDA_SH} && \
conda activate ${QUOTED_CONDA_ENV} && \
export PYTHONPATH=${QUOTED_PYTHONPATH} && \
${PYTHON_COMMAND}"

      if ! "${RUN_CTN}" exec \
        "${NV_FLAG[@]}" \
        "${BINDS[@]}" \
        "${CONTAINER_IMG}" \
        /bin/bash -lc "${INNER_COMMAND}"; then

        echo "ERROR: Evaluation failed for ${LABEL}" >&2

        if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
          continue
        fi

        exit 1
      fi

      if ! append_csv_safely "${TMP_CSV}"; then
        echo "ERROR: Could not append result for ${LABEL}" >&2

        if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
          continue
        fi

        exit 1
      fi
    done
  done

done < "${PROCESSED}"


echo "All available cases processed."
echo "CSV: ${OUT_CSV_ABS}"