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

# Add the complete path to vtk2ply.py here.
VTK2PLY_SCRIPT="/home/cir/swolff/surface_reconstruction/Vox2Cortex/vtk2ply.py"

RUN_CTN="${RUN_CTN:-singularity}"

# Evaluation Python environment
PYTHON_BIN="python"
CONDA_SH="/opt/conda/etc/profile.d/conda.sh"
CONDA_ENV="torch_1_10"

# Conversion Python environment.
#
# These may be the same as the evaluation environment if PyVista is installed
# in torch_1_10. Otherwise, set CONVERTER_CONDA_ENV to the Conda environment
# containing PyVista.
CONVERTER_PYTHON_BIN="python"
CONVERTER_CONDA_ENV="${CONDA_ENV}"


# =========================================================================
# Atlas input configuration
# =========================================================================

# Existing dataset root.
GT_ROOT="/project/fetal/2025_FetalSurface_swolff/2024_FETAL_CONTROL_JT"

# Atlas folders are below:
#
# ${GT_ROOT}/atlases/CRL_ATLAS2017_GW21
# ...
# ${GT_ROOT}/atlases/CRL_ATLAS2017_GW38
ATLAS_ROOT="${GT_ROOT}/atlases"

# One atlas folder per line, for example:
#
# CRL_ATLAS2017_GW21
# CRL_ATLAS2017_GW22
# ...
# CRL_ATLAS2017_GW38
ATLAS_PROCESSED="${ATLAS_ROOT}/atlases_folders.txt"

# Name of the directory containing the surface meshes within each atlas.
ATLAS_SURFACE_DIRECTORY="SURFACES"

# Atlas filenames use:
#
# GW21 -> 104021
# GW22 -> 104022
# ...
# GW38 -> 104038
ATLAS_FILE_ID_PREFIX="1040"


# =========================================================================
# PLY conversion behavior
# =========================================================================

# Converted files are stored next to their source VTK files using the same
# basename:
#
# INNERCORTEXLEFT_HR_104021.vtk
# INNERCORTEXLEFT_HR_104021.ply
#
# Existing nonempty PLY files are reused.
#
# If false, the modification time of the VTK is ignored and an existing PLY
# is always reused.
#
# If true, the PLY is regenerated when the VTK is newer than the PLY.
RECONVERT_IF_VTK_NEWER="false"


# =========================================================================
# Output configuration
# =========================================================================

OUT_CSV="/home/cir/swolff/surface_reconstruction/atlas_CRL_gt_quality.csv"

# Remove an existing output CSV before starting.
RESET_OUTPUT="false"

# Continue with other surfaces if one conversion or evaluation fails.
CONTINUE_ON_ERROR="true"


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

if [[ ! -f "${VTK2PLY_SCRIPT}" ]]; then
  echo "ERROR: VTK-to-PLY converter not found: ${VTK2PLY_SCRIPT}" >&2
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
# Atlas filename construction
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
      echo \
        "ERROR: Unsupported hemisphere/surface combination: ${hemisphere}:${surface}" \
        >&2
      return 1
      ;;
  esac
}


atlas_vtk_path() {
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


# The atlas bind needs write access because converted PLY files are written
# next to their VTK files. Singularity bind mounts are normally writable
# unless the container or filesystem is explicitly configured otherwise.
add_bind "${GT_ROOT}"
add_bind "${REPO}"
add_bind "${OUT_DIR}"
add_bind "$(dirname "${QUALITY_CONFIG}")"
add_bind "$(dirname "${VTK2PLY_SCRIPT}")"


# =========================================================================
# Generic containerized Python runner
# =========================================================================

run_python_in_container() {
  local conda_environment="$1"
  shift

  # All remaining arguments form the Python command and are quoted
  # independently before being passed through bash -lc.
  local quoted_command
  local quoted_conda_sh
  local quoted_conda_environment
  local quoted_pythonpath
  local container_pythonpath
  local inner_command

  printf -v quoted_command '%q ' "$@"
  printf -v quoted_conda_sh '%q' "${CONDA_SH}"
  printf -v quoted_conda_environment '%q' "${conda_environment}"

  container_pythonpath="${REPO}:${PYTHONPATH:-}"
  printf -v quoted_pythonpath '%q' "${container_pythonpath}"

  inner_command="source ${quoted_conda_sh} && \
conda activate ${quoted_conda_environment} && \
export PYTHONPATH=${quoted_pythonpath} && \
${quoted_command}"

  "${RUN_CTN}" exec \
    "${BINDS[@]}" \
    "${CONTAINER_IMG}" \
    /bin/bash -lc "${inner_command}"
}


# =========================================================================
# VTK-to-PLY conversion
# =========================================================================

ensure_atlas_ply_exists() {
  local vtk_path="$1"
  local ply_path="$2"

  local conversion_required="false"

  if [[ ! -f "${vtk_path}" ]]; then
    echo "ERROR: Source VTK does not exist: ${vtk_path}" >&2
    return 1
  fi

  # A nonempty PLY is reusable by default.
  if [[ -s "${ply_path}" ]]; then
    if [[ "${RECONVERT_IF_VTK_NEWER}" == "true" && "${vtk_path}" -nt "${ply_path}" ]]; then
      echo "    Existing PLY is older than VTK; reconversion required."
      conversion_required="true"
    else
      echo "    Reusing existing PLY: ${ply_path}"
      return 0
    fi
  else
    conversion_required="true"
  fi

  if [[ "${conversion_required}" != "true" ]]; then
    return 0
  fi

  local conversion_lock="${ply_path}.conversion.lock"

  # The conversion occurs under a lock so that two simultaneous jobs cannot
  # create or replace the same PLY at the same time.
  (
    flock -x 201

    # Check again after obtaining the lock. Another process may have created
    # the PLY while this process was waiting.
    if [[ -s "${ply_path}" ]]; then
      if [[ "${RECONVERT_IF_VTK_NEWER}" == "true" && "${vtk_path}" -nt "${ply_path}" ]]; then
        echo "    Locked PLY is older than VTK; regenerating."
      else
        echo "    PLY was created by another process: ${ply_path}"
        exit 0
      fi
    fi

    local ply_directory
    local ply_basename
    local temporary_ply

    ply_directory="$(dirname "${ply_path}")"
    ply_basename="$(basename "${ply_path}" .ply)"

    # The temporary file retains the .ply suffix so that PyVista selects the
    # correct writer from the filename.
    temporary_ply="${ply_directory}/.${ply_basename}.conversion.${SLURM_JOB_ID:-noj}.$$.${RANDOM}.ply"

    rm -f "${temporary_ply}"

    cleanup_conversion_file() {
      rm -f "${temporary_ply}"
    }

    trap cleanup_conversion_file EXIT

    echo "    Converting VTK to PLY"
    echo "      Input:  ${vtk_path}"
    echo "      Output: ${ply_path}"

    # Calls the converter exactly as:
    #
    # python vtk2ply.py input.vtk output.ply
    if ! run_python_in_container \
      "${CONVERTER_CONDA_ENV}" \
      "${CONVERTER_PYTHON_BIN}" \
      "${VTK2PLY_SCRIPT}" \
      "${vtk_path}" \
      "${temporary_ply}"; then

      echo "ERROR: VTK-to-PLY conversion failed: ${vtk_path}" >&2
      exit 1
    fi

    if [[ ! -s "${temporary_ply}" ]]; then
      echo \
        "ERROR: Converter completed but did not create a nonempty PLY: ${temporary_ply}" \
        >&2
      exit 1
    fi

    # Atomic replacement within the same filesystem. The evaluator never sees
    # a partially written final PLY.
    mv -f \
      "${temporary_ply}" \
      "${ply_path}"

    trap - EXIT

    if [[ ! -s "${ply_path}" ]]; then
      echo \
        "ERROR: Final converted PLY is missing or empty: ${ply_path}" \
        >&2
      exit 1
    fi

    echo "    Created PLY: ${ply_path}"
  ) 201>"${conversion_lock}"
}


# =========================================================================
# Safe CSV aggregation
# =========================================================================

append_csv_safely() {
  local temporary_csv="$1"

  if [[ ! -s "${temporary_csv}" ]]; then
    echo \
      "ERROR: Temporary CSV is missing or empty: ${temporary_csv}" \
      >&2
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
      local existing_header

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
        echo \
          "Use a new OUT_CSV or set RESET_OUTPUT=true." \
          >&2
        exit 1
      fi

      tail -n +2 "${temporary_csv}" >> "${OUT_CSV_ABS}"
    fi
  ) 200>"${LOCK_FILE}"
}


# =========================================================================
# Counters
# =========================================================================

ATLAS_COUNT=0
SURFACE_COUNT=0
MISSING_VTK_COUNT=0
CONVERSION_FAILED_COUNT=0
EVALUATION_FAILED_COUNT=0


# =========================================================================
# Atlas GT-only evaluation
# =========================================================================

echo "Starting atlas ground-truth quality evaluation"
echo "Atlas root:       ${ATLAS_ROOT}"
echo "Atlas list:       ${ATLAS_PROCESSED}"
echo "VTK converter:    ${VTK2PLY_SCRIPT}"
echo "Quality config:   ${QUALITY_CONFIG}"
echo "Output CSV:       ${OUT_CSV_ABS}"
echo


while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
  # Remove a possible Windows carriage return.
  raw_line="${raw_line%$'\r'}"

  # Trim leading whitespace.
  line="${raw_line#"${raw_line%%[![:space:]]*}"}"

  # Trim trailing whitespace.
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

  # Only the first whitespace-separated value is used.
  atlas_entry=""
  read -r atlas_entry _ <<< "${line}"

  # Permit either a bare folder name or a full path.
  atlas_entry="${atlas_entry%/}"
  ATLAS_FOLDER="$(basename "${atlas_entry}")"

  # Extract GW21 to GW38.
  if [[ "${ATLAS_FOLDER}" =~ ^CRL_ATLAS2017_GW(2[1-9]|3[0-8])$ ]]; then
    GW="${BASH_REMATCH[1]}"
  else
    echo \
      "WARN: Atlas folder does not match CRL_ATLAS2017_GW21-38: ${ATLAS_FOLDER}" \
      >&2
    continue
  fi

  ATLAS_DIRECTORY="${ATLAS_ROOT}/${ATLAS_FOLDER}"
  SURFACES_DIRECTORY="${ATLAS_DIRECTORY}/${ATLAS_SURFACE_DIRECTORY}"

  if [[ ! -d "${ATLAS_DIRECTORY}" ]]; then
    echo \
      "WARN: Atlas directory not found: ${ATLAS_DIRECTORY}" \
      >&2
    continue
  fi

  if [[ ! -d "${SURFACES_DIRECTORY}" ]]; then
    echo \
      "WARN: Atlas SURFACES directory not found: ${SURFACES_DIRECTORY}" \
      >&2
    continue
  fi

  ATLAS_COUNT=$((ATLAS_COUNT + 1))

  echo "Atlas: ${ATLAS_FOLDER}"
  echo "  Gestational week: ${GW}"
  echo "  Surface directory: ${SURFACES_DIRECTORY}"

  for HEMISPHERE in L R; do
    for SURFACE in white pial; do
      VTK_PATH="$(
        atlas_vtk_path \
          "${ATLAS_FOLDER}" \
          "${GW}" \
          "${HEMISPHERE}" \
          "${SURFACE}"
      )"

      # The converted PLY is stored beside the VTK using the same basename.
      PLY_PATH="${VTK_PATH%.vtk}.ply"

      LABEL="${ATLAS_FOLDER}_${HEMISPHERE}_${SURFACE}"

      if [[ ! -f "${VTK_PATH}" ]]; then
        echo \
          "  Missing ${HEMISPHERE} ${SURFACE} VTK: ${VTK_PATH}" \
          >&2

        MISSING_VTK_COUNT=$((MISSING_VTK_COUNT + 1))
        continue
      fi

      echo "  Preparing ${LABEL}"

      if ! ensure_atlas_ply_exists \
        "${VTK_PATH}" \
        "${PLY_PATH}"; then

        echo \
          "ERROR: Could not prepare PLY for ${LABEL}" \
          >&2

        CONVERSION_FAILED_COUNT=$(
          (CONVERSION_FAILED_COUNT + 1)
        )

        if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
          continue
        fi

        exit 1
      fi

      echo "  Evaluating ${LABEL}"
      echo "    PLY: ${PLY_PATH}"

      TMP_CSV="${TMP_DIR}/${LABEL}.csv"

      # Prevent a failed evaluator from leaving a stale temporary result.
      rm -f "${TMP_CSV}"

      declare -a EVALUATOR_ARGS

      EVALUATOR_ARGS=(
        "${PYTHON_BIN}"
        "${REPO}/evaluate_mesh_pair.py"
        --mode "gt-only"
        --gt "${PLY_PATH}"
        --config "${QUALITY_CONFIG}"
        --subject "${ATLAS_FOLDER}"
        --session "GW${GW}"
        --hemisphere "${HEMISPHERE}"
        --surface "${SURFACE}"
        --label "${LABEL}"
        --out_csv "${TMP_CSV}"
      )

      if ! run_python_in_container \
        "${CONDA_ENV}" \
        "${EVALUATOR_ARGS[@]}"; then

        echo \
          "ERROR: Evaluation failed for ${LABEL}" \
          >&2

        EVALUATION_FAILED_COUNT=$(
          (EVALUATION_FAILED_COUNT + 1)
        )

        if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
          continue
        fi

        exit 1
      fi

      if ! append_csv_safely "${TMP_CSV}"; then
        echo \
          "ERROR: Could not append result for ${LABEL}" \
          >&2

        EVALUATION_FAILED_COUNT=$(
          (EVALUATION_FAILED_COUNT + 1)
        )

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
echo "Atlas folders processed:       ${ATLAS_COUNT}"
echo "Surfaces evaluated:            ${SURFACE_COUNT}"
echo "Missing source VTK files:      ${MISSING_VTK_COUNT}"
echo "Failed VTK-to-PLY conversions: ${CONVERSION_FAILED_COUNT}"
echo "Failed evaluations:            ${EVALUATION_FAILED_COUNT}"
echo "Output CSV:                    ${OUT_CSV_ABS}"