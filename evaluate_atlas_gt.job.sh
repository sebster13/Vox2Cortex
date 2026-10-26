#!/usr/bin/env bash

#SBATCH --job-name=atlas_gt_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --output=atlas_gt_eval-%x-%j.out
#SBATCH --error=atlas_gt_eval-%x-%j.err
#SBATCH --qos=longrunning
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=a12007396@unet.univie.ac.at

set -euo pipefail


# =========================================================================
# Container and repository configuration
# =========================================================================

CONTAINER_IMG="/home/cir/swolff/surface_reconstruction/vox2cortex.sif"

REPO="/home/cir/swolff/surface_reconstruction/Vox2Cortex/vox2organ"

QUALITY_CONFIG="${REPO}/quality_config.json"

RUN_CTN="${RUN_CTN:-singularity}"

PYTHON_BIN="python"

CONDA_SH="/opt/conda/etc/profile.d/conda.sh"

CONDA_ENV="torch_1_10"


# =========================================================================
# Atlas input configuration
# =========================================================================

# This remains the same dataset root used by the previous evaluation.
GT_ROOT="/project/fetal/2025_FetalSurface_swolff/2024_FETAL_CONTROL_JT"

# Atlas folders are located below GT_ROOT/atlases.
ATLAS_ROOT="${GT_ROOT}/atlases"

# One atlas folder per line, for example:
#
# CRL_ATLAS2017_GW21
# CRL_ATLAS2017_GW22
# ...
# CRL_ATLAS2017_GW38
#
# Empty lines and lines beginning with # are ignored.
ATLAS_PROCESSED="${ATLAS_ROOT}/atlases_folders.txt"

# The atlas filenames contain identifiers such as:
#
# GW21 -> 104021
# GW22 -> 104022
# GW38 -> 104038
#
# The fixed portion is therefore 1040 and the gestational week is appended.
ATLAS_FILE_ID_PREFIX="1040"

# Name of the directory containing the surfaces inside each atlas folder.
ATLAS_SURFACE_DIRECTORY="SURFACES"


# =========================================================================
# Output configuration
# =========================================================================

# Use a separate output from the subject/session reconstruction evaluation.
OUT_CSV="/home/cir/swolff/surface_reconstruction/atlas_gt_quality.csv"

# If true, remove an existing output file before starting.
#
# This is useful after changing output columns or metric definitions.
RESET_OUTPUT="false"

# If true, failure to evaluate one surface does not terminate the whole job.
CONTINUE_ON_ERROR="true"


# =========================================================================
# Evaluation metadata
# =========================================================================

# The evaluator requires metadata fields named Subject and Session. For
# atlas evaluation, they are populated as follows:
#
# Subject = complete atlas folder name, e.g. CRL_ATLAS2017_GW21
# Session = gestational-week identifier, e.g. GW21
#
# Hemisphere and Surface continue to contain:
#
# Hemisphere = L or R
# Surface    = white or pial


# =========================================================================
# Validation
# =========================================================================

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

if [[ ! -f "${REPO}/utils/eval_metrics.py" ]]; then
  echo "ERROR: Metric file not found: ${REPO}/utils/eval_metrics.py" >&2
  exit 1
fi

if [[ ! -f "${QUALITY_CONFIG}" ]]; then
  echo "ERROR: Quality configuration not found: ${QUALITY_CONFIG}" >&2
  exit 1
fi

if [[ ! -d "${GT_ROOT}" ]]; then
  echo "ERROR: GT root does not exist: ${GT_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${ATLAS_ROOT}" ]]; then
  echo "ERROR: Atlas root does not exist: ${ATLAS_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${ATLAS_PROCESSED}" ]]; then
  echo "ERROR: Atlas list does not exist: ${ATLAS_PROCESSED}" >&2
  exit 1
fi

if ! command -v "${RUN_CTN}" >/dev/null 2>&1; then
  echo "ERROR: Container runner not found: ${RUN_CTN}" >&2
  exit 1
fi


# =========================================================================
# Atlas filename mapping
# =========================================================================

atlas_surface_stem() {
  local hemisphere="$1"
  local surface="$2"

  case "${hemisphere}:${surface}" in
    "L:white")
      printf '%s' "INNERCORTEXLEFT"
      ;;

    "R:white")
      printf '%s' "INNERCORTEXRIGHT"
      ;;

    "L:pial")
      printf '%s' "OUTERCORTEXLEFT"
      ;;

    "R:pial")
      printf '%s' "OUTERCORTEXRIGHT"
      ;;

    *)
      echo "ERROR: Unsupported hemisphere/surface combination: ${hemisphere}:${surface}" >&2
      return 1
      ;;
  esac
}


atlas_mesh_path() {
  local atlas_folder="$1"
  local gestational_week="$2"
  local hemisphere="$3"
  local surface="$4"

  local surface_stem
  local atlas_file_id

  surface_stem="$(
    atlas_surface_stem \
      "${hemisphere}" \
      "${surface}"
  )"

  atlas_file_id="${ATLAS_FILE_ID_PREFIX}${gestational_week}"

  printf '%s/%s/%s/%s_HR_%s.vtk' \
    "${ATLAS_ROOT}" \
    "${atlas_folder}" \
    "${ATLAS_SURFACE_DIRECTORY}" \
    "${surface_stem}" \
    "${atlas_file_id}"
}


# =========================================================================
# Output and temporary workspace
# =========================================================================

OUT_DIR="$(dirname "${OUT_CSV}")"
mkdir -p "${OUT_DIR}"

OUT_CSV_ABS="$(readlink -m "${OUT_CSV}")"
LOCK_FILE="${OUT_CSV_ABS}.lock"

if [[ "${RESET_OUTPUT}" == "true" ]]; then
  rm -f \
    "${OUT_CSV_ABS}" \
    "${LOCK_FILE}"
fi

TMP_DIR="$(
  mktemp -d \
    -t "atlas_gt_eval_${SLURM_JOB_ID:-noj}_XXXXXX"
)"

trap 'rm -rf "${TMP_DIR}"' EXIT


# =========================================================================
# Container bind mounts
# =========================================================================

declare -a BINDS
declare -A BOUND_PATHS

BINDS=()

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

add_bind "${GT_ROOT}"
add_bind "${REPO}"
add_bind "${OUT_DIR}"
add_bind "$(dirname "${QUALITY_CONFIG}")"


# =========================================================================
# Safe CSV aggregation
# =========================================================================

append_csv_safely() {
  local temporary_csv="$1"

  if [[ ! -s "${temporary_csv}" ]]; then
    echo "ERROR: Temporary CSV is missing or empty: ${temporary_csv}" >&2
    return 1
  fi

  local new_header
  new_header="$(
    head -n 1 "${temporary_csv}" |
      tr -d '\r'
  )"

  (
    flock -x 200

    if [[ ! -s "${OUT_CSV_ABS}" ]]; then
      cat "${temporary_csv}" > "${OUT_CSV_ABS}"
    else
      existing_header="$(
        head -n 1 "${OUT_CSV_ABS}" |
          tr -d '\r'
      )"

      if [[ "${existing_header}" != "${new_header}" ]]; then
        echo "ERROR: Output CSV schema mismatch." >&2
        echo "Existing output header:" >&2
        echo "${existing_header}" >&2
        echo "Current evaluator header:" >&2
        echo "${new_header}" >&2
        echo "Use a new OUT_CSV or set RESET_OUTPUT=true." >&2
        exit 1
      fi

      tail -n +2 "${temporary_csv}" >> "${OUT_CSV_ABS}"
    fi
  ) 200>"${LOCK_FILE}"
}


# =========================================================================
# Evaluation counters
# =========================================================================

ATLAS_COUNT=0
SURFACE_COUNT=0
MISSING_COUNT=0
FAILED_COUNT=0


# =========================================================================
# Atlas GT-only evaluation
# =========================================================================

echo "Starting atlas ground-truth quality evaluation"
echo "Atlas root:       ${ATLAS_ROOT}"
echo "Atlas list:       ${ATLAS_PROCESSED}"
echo "Quality config:   ${QUALITY_CONFIG}"
echo "Output CSV:       ${OUT_CSV_ABS}"
echo


while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
  # Remove a potential Windows carriage return.
  raw_line="${raw_line%$'\r'}"

  # Remove leading whitespace.
  line="${raw_line#"${raw_line%%[![:space:]]*}"}"

  # Remove trailing whitespace.
  line="${line%"${line##*[![:space:]]}"}"

  # Skip empty lines.
  [[ -z "${line}" ]] && continue

  # Skip full-line comments.
  [[ "${line}" == \#* ]] && continue

  # Remove an optional trailing comment.
  line="${line%%#*}"

  # Trim again after removing the comment.
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"

  [[ -z "${line}" ]] && continue

  # Only the first whitespace-separated item is treated as the folder.
  atlas_entry=""
  read -r atlas_entry _ <<< "${line}"

  # Permit either:
  #
  # CRL_ATLAS2017_GW21
  #
  # or:
  #
  # /some/path/CRL_ATLAS2017_GW21/
  atlas_entry="${atlas_entry%/}"
  ATLAS_FOLDER="$(basename "${atlas_entry}")"

  # Extract the gestational week and ensure that it lies between 21 and 38.
  if [[ "${ATLAS_FOLDER}" =~ ^CRL_ATLAS2017_GW(2[1-9]|3[0-8])$ ]]; then
    GW="${BASH_REMATCH[1]}"
  else
    echo "WARN: Atlas folder name does not match CRL_ATLAS2017_GW21-38:" >&2
    echo "      ${ATLAS_FOLDER}" >&2
    echo "      Skipping this entry." >&2
    continue
  fi

  ATLAS_DIRECTORY="${ATLAS_ROOT}/${ATLAS_FOLDER}"
  SURFACES_DIRECTORY="${ATLAS_DIRECTORY}/${ATLAS_SURFACE_DIRECTORY}"

  if [[ ! -d "${ATLAS_DIRECTORY}" ]]; then
    echo "WARN: Atlas directory not found: ${ATLAS_DIRECTORY}" >&2
    continue
  fi

  if [[ ! -d "${SURFACES_DIRECTORY}" ]]; then
    echo "WARN: Atlas SURFACES directory not found: ${SURFACES_DIRECTORY}" >&2
    continue
  fi

  ATLAS_COUNT=$((ATLAS_COUNT + 1))

  echo "Atlas: ${ATLAS_FOLDER}"
  echo "  Gestational week: ${GW}"
  echo "  Surface directory: ${SURFACES_DIRECTORY}"

  for HEMISPHERE in L R; do
    for SURFACE in white pial; do
      MESH_PATH="$(
        atlas_mesh_path \
          "${ATLAS_FOLDER}" \
          "${GW}" \
          "${HEMISPHERE}" \
          "${SURFACE}"
      )"

      LABEL="${ATLAS_FOLDER}_${HEMISPHERE}_${SURFACE}"

      if [[ ! -f "${MESH_PATH}" ]]; then
        echo "  Missing ${HEMISPHERE} ${SURFACE}: ${MESH_PATH}" >&2
        MISSING_COUNT=$((MISSING_COUNT + 1))
        continue
      fi

      echo "  Evaluating ${LABEL}"
      echo "    Mesh: ${MESH_PATH}"

      TMP_CSV="${TMP_DIR}/${LABEL}.csv"

      declare -a PYTHON_ARGS

      PYTHON_ARGS=(
        "${PYTHON_BIN}"
        "${REPO}/evaluate_mesh_pair.py"
        --mode "gt-only"
        --gt "${MESH_PATH}"
        --config "${QUALITY_CONFIG}"
        --subject "${ATLAS_FOLDER}"
        --session "GW${GW}"
        --hemisphere "${HEMISPHERE}"
        --surface "${SURFACE}"
        --label "${LABEL}"
        --out_csv "${TMP_CSV}"
      )

      # Quote every Python argument before sending the command through
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
        "${BINDS[@]}" \
        "${CONTAINER_IMG}" \
        /bin/bash -lc "${INNER_COMMAND}"; then

        echo "ERROR: Evaluation failed for ${LABEL}" >&2
        FAILED_COUNT=$((FAILED_COUNT + 1))

        if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
          continue
        fi

        exit 1
      fi

      if ! append_csv_safely "${TMP_CSV}"; then
        echo "ERROR: Could not append result for ${LABEL}" >&2
        FAILED_COUNT=$((FAILED_COUNT + 1))

        if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
          continue
        fi

        exit 1
      fi

      SURFACE_COUNT=$((SURFACE_COUNT + 1))
    done
  done

  echo
done < "${ATLAS_PROCESSED}"


# =========================================================================
# Summary
# =========================================================================

echo "Atlas evaluation completed."
echo "Atlas folders processed:   ${ATLAS_COUNT}"
echo "Surfaces evaluated:        ${SURFACE_COUNT}"
echo "Missing surface files:     ${MISSING_COUNT}"
echo "Failed evaluations:        ${FAILED_COUNT}"
echo "Output CSV:                ${OUT_CSV_ABS}"