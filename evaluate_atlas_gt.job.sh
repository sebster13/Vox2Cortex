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

# Container used for cortical-surface metric evaluation.
EVALUATION_CONTAINER_IMG="/home/cir/swolff/surface_reconstruction/vox2cortex.sif"

# Separate container used only for VTK-to-PLY conversion with PyVista.
CONVERSION_CONTAINER_IMG="/home/cir/swolff/surface_reconstruction/surfaceprocessing.sif"

REPO="/home/cir/swolff/surface_reconstruction/Vox2Cortex/vox2organ"

QUALITY_CONFIG="${REPO}/quality_config.json"

# Add the complete path to vtk2ply.py here.
VTK2PLY_SCRIPT="/FULL/PATH/TO/vtk2ply.py"

RUN_CTN="${RUN_CTN:-singularity}"


# =========================================================================
# Evaluation Python environment in vox2cortex.sif
# =========================================================================

EVALUATION_PYTHON_BIN="python"

EVALUATION_CONDA_SH="/opt/conda/etc/profile.d/conda.sh"

EVALUATION_CONDA_ENV="torch_1_10"


# =========================================================================
# Conversion Python environment in surfaceprocessing.sif
# =========================================================================

# Set this to the Python executable in surfaceprocessing.sif.
#
# If Python is directly available in PATH, leave this as:
#
#   python
#
# Otherwise, use an absolute path such as:
#
#   /opt/conda/bin/python
CONVERSION_PYTHON_BIN="python"

# Most processing containers already expose their intended Python environment
# without requiring "conda activate". The default is therefore false.
#
# Set to true only if surfaceprocessing.sif requires explicit Conda
# activation.
CONVERSION_USE_CONDA="false"

# These are used only if CONVERSION_USE_CONDA="true".
CONVERSION_CONDA_SH="/opt/conda/etc/profile.d/conda.sh"
CONVERSION_CONDA_ENV="surfaceprocessing"


# =========================================================================
# Atlas input configuration
# =========================================================================

GT_ROOT="/project/fetal/2025_FetalSurface_swolff/2024_FETAL_CONTROL_JT"

ATLAS_ROOT="${GT_ROOT}/atlases"

# One atlas folder per line:
#
# CRL_ATLAS2017_GW21
# CRL_ATLAS2017_GW22
# ...
# CRL_ATLAS2017_GW38
ATLAS_PROCESSED="${ATLAS_ROOT}/processed_atlases.txt"

ATLAS_SURFACE_DIRECTORY="SURFACES"

# GW21 -> 104021
# GW22 -> 104022
# ...
# GW38 -> 104038
ATLAS_FILE_ID_PREFIX="1040"


# =========================================================================
# PLY conversion behavior
# =========================================================================

# Existing nonempty PLY files are reused.
#
# If true, an existing PLY is regenerated when its source VTK is newer.
RECONVERT_IF_VTK_NEWER="false"


# =========================================================================
# Output configuration
# =========================================================================

OUT_CSV="/home/cir/swolff/surface_reconstruction/atlas_gt_quality.csv"

# Remove an existing result CSV before starting.
RESET_OUTPUT="false"

# Continue evaluating remaining surfaces after an individual failure.
CONTINUE_ON_ERROR="true"


# =========================================================================
# Validation
# =========================================================================

if [[ ! -f "${EVALUATION_CONTAINER_IMG}" ]]; then
  echo "ERROR: Evaluation container not found:" >&2
  echo "       ${EVALUATION_CONTAINER_IMG}" >&2
  exit 1
fi

if [[ ! -f "${CONVERSION_CONTAINER_IMG}" ]]; then
  echo "ERROR: Conversion container not found:" >&2
  echo "       ${CONVERSION_CONTAINER_IMG}" >&2
  exit 1
fi

if [[ ! -d "${REPO}" ]]; then
  echo "ERROR: Repository not found: ${REPO}" >&2
  exit 1
fi

if [[ ! -f "${REPO}/evaluate_mesh_pair.py" ]]; then
  echo "ERROR: Evaluator not found:" >&2
  echo "       ${REPO}/evaluate_mesh_pair.py" >&2
  exit 1
fi

if [[ ! -f "${REPO}/utils/eval_metrics.py" ]]; then
  echo "ERROR: Metric file not found:" >&2
  echo "       ${REPO}/utils/eval_metrics.py" >&2
  exit 1
fi

if [[ ! -f "${QUALITY_CONFIG}" ]]; then
  echo "ERROR: Quality configuration not found:" >&2
  echo "       ${QUALITY_CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${VTK2PLY_SCRIPT}" ]]; then
  echo "ERROR: VTK-to-PLY converter not found:" >&2
  echo "       ${VTK2PLY_SCRIPT}" >&2
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
  echo "ERROR: Atlas list does not exist:" >&2
  echo "       ${ATLAS_PROCESSED}" >&2
  exit 1
fi

if ! command -v "${RUN_CTN}" >/dev/null 2>&1; then
  echo "ERROR: Container runner not found: ${RUN_CTN}" >&2
  exit 1
fi

if [[ "${CONVERSION_USE_CONDA}" != "true" && "${CONVERSION_USE_CONDA}" != "false" ]]; then
  echo "ERROR: CONVERSION_USE_CONDA must be true or false." >&2
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


# GT_ROOT must be writable in the conversion container because PLY files are
# stored next to the original VTK files.
add_bind "${GT_ROOT}"

# Required by the evaluation container.
add_bind "${REPO}"
add_bind "${OUT_DIR}"
add_bind "$(dirname "${QUALITY_CONFIG}")"

# Required by the conversion container.
add_bind "$(dirname "${VTK2PLY_SCRIPT}")"


# =========================================================================
# Evaluation-container runner
# =========================================================================

run_evaluation_python() {
  local quoted_command
  local quoted_conda_sh
  local quoted_conda_environment
  local quoted_pythonpath
  local container_pythonpath
  local inner_command

  # All function arguments constitute the command being run.
  printf -v quoted_command '%q ' "$@"

  printf -v quoted_conda_sh \
    '%q' \
    "${EVALUATION_CONDA_SH}"

  printf -v quoted_conda_environment \
    '%q' \
    "${EVALUATION_CONDA_ENV}"

  container_pythonpath="${REPO}:${PYTHONPATH:-}"

  printf -v quoted_pythonpath \
    '%q' \
    "${container_pythonpath}"

  inner_command="source ${quoted_conda_sh} && \
conda activate ${quoted_conda_environment} && \
export PYTHONPATH=${quoted_pythonpath} && \
${quoted_command}"

  "${RUN_CTN}" exec \
    "${BINDS[@]}" \
    "${EVALUATION_CONTAINER_IMG}" \
    /bin/bash -lc "${inner_command}"
}


# =========================================================================
# Conversion-container runner
# =========================================================================

run_conversion_python() {
  local quoted_command
  local inner_command

  # All function arguments constitute the converter command.
  printf -v quoted_command '%q ' "$@"

  if [[ "${CONVERSION_USE_CONDA}" == "true" ]]; then
    local quoted_conda_sh
    local quoted_conda_environment

    printf -v quoted_conda_sh \
      '%q' \
      "${CONVERSION_CONDA_SH}"

    printf -v quoted_conda_environment \
      '%q' \
      "${CONVERSION_CONDA_ENV}"

    inner_command="source ${quoted_conda_sh} && \
conda activate ${quoted_conda_environment} && \
${quoted_command}"
  else
    inner_command="${quoted_command}"
  fi

  "${RUN_CTN}" exec \
    "${BINDS[@]}" \
    "${CONVERSION_CONTAINER_IMG}" \
    /bin/bash -lc "${inner_command}"
}


# =========================================================================
# Conversion-environment validation
# =========================================================================

validate_conversion_environment() {
  echo "Checking conversion environment in:"
  echo "  ${CONVERSION_CONTAINER_IMG}"

  if ! run_conversion_python \
    "${CONVERSION_PYTHON_BIN}" \
    -c \
    "import sys; import pyvista as pv; print('Python:', sys.executable); print('PyVista:', pv.__version__)"; then

    echo "ERROR: PyVista could not be imported in the conversion container." >&2
    echo "       Container: ${CONVERSION_CONTAINER_IMG}" >&2
    echo "       Python:    ${CONVERSION_PYTHON_BIN}" >&2
    echo "       Conda:     ${CONVERSION_USE_CONDA}" >&2

    if [[ "${CONVERSION_USE_CONDA}" == "true" ]]; then
      echo "       Environment: ${CONVERSION_CONDA_ENV}" >&2
    fi

    exit 1
  fi

  echo
}


# =========================================================================
# VTK-to-PLY conversion
# =========================================================================

ensure_atlas_ply_exists() {
  local vtk_path="$1"
  local ply_path="$2"

  if [[ ! -f "${vtk_path}" ]]; then
    echo "ERROR: Source VTK does not exist: ${vtk_path}" >&2
    return 1
  fi

  # Reuse an existing nonempty PLY unless explicit reconversion is needed.
  if [[ -s "${ply_path}" ]]; then
    if [[ "${RECONVERT_IF_VTK_NEWER}" == "true" && "${vtk_path}" -nt "${ply_path}" ]]; then
      echo "    Existing PLY is older than VTK; reconversion required."
    else
      echo "    Reusing existing PLY: ${ply_path}"
      return 0
    fi
  fi

  local conversion_lock="${ply_path}.conversion.lock"

  # Serializes conversion if multiple jobs encounter the same VTK.
  (
    flock -x 201

    # Check again after obtaining the lock. Another process may have
    # completed the conversion while this process was waiting.
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

    # Keep the .ply suffix so that PyVista selects the correct output writer.
    temporary_ply="${ply_directory}/.${ply_basename}.conversion.${SLURM_JOB_ID:-noj}.$$.${RANDOM}.ply"

    rm -f "${temporary_ply}"

    cleanup_conversion_file() {
      rm -f "${temporary_ply}"
    }

    trap cleanup_conversion_file EXIT

    echo "    Converting VTK to PLY with surfaceprocessing.sif"
    echo "      Input:     ${vtk_path}"
    echo "      Temporary: ${temporary_ply}"
    echo "      Final:     ${ply_path}"

    # This executes inside surfaceprocessing.sif:
    #
    # python vtk2ply.py input.vtk temporary_output.ply
    if ! run_conversion_python \
      "${CONVERSION_PYTHON_BIN}" \
      "${VTK2PLY_SCRIPT}" \
      "${vtk_path}" \
      "${temporary_ply}"; then

      echo "ERROR: VTK-to-PLY conversion failed:" >&2
      echo "       ${vtk_path}" >&2
      exit 1
    fi

    if [[ ! -s "${temporary_ply}" ]]; then
      echo "ERROR: Converter did not create a nonempty PLY:" >&2
      echo "       ${temporary_ply}" >&2
      exit 1
    fi

    # The temporary and final files are in the same directory/filesystem, so
    # mv provides atomic publication of the completed file.
    mv -f \
      "${temporary_ply}" \
      "${ply_path}"

    trap - EXIT

    if [[ ! -s "${ply_path}" ]]; then
      echo "ERROR: Final converted PLY is missing or empty:" >&2
      echo "       ${ply_path}" >&2
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
    echo "ERROR: Temporary CSV is missing or empty:" >&2
    echo "       ${temporary_csv}" >&2
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
        echo "Use a new OUT_CSV or set RESET_OUTPUT=true." >&2
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
# Validate PyVista before processing all atlases
# =========================================================================

validate_conversion_environment


# =========================================================================
# Atlas GT-only evaluation
# =========================================================================

echo "Starting atlas ground-truth quality evaluation"
echo "Atlas root:          ${ATLAS_ROOT}"
echo "Atlas list:          ${ATLAS_PROCESSED}"
echo "Conversion image:    ${CONVERSION_CONTAINER_IMG}"
echo "Evaluation image:    ${EVALUATION_CONTAINER_IMG}"
echo "VTK converter:       ${VTK2PLY_SCRIPT}"
echo "Quality config:      ${QUALITY_CONFIG}"
echo "Output CSV:          ${OUT_CSV_ABS}"
echo


while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
  # Remove a possible Windows carriage return.
  raw_line="${raw_line%$'\r'}"

  # Trim leading and trailing whitespace.
  line="${raw_line#"${raw_line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"

  # Skip empty lines and full-line comments.
  [[ -z "${line}" ]] && continue
  [[ "${line}" == \#* ]] && continue

  # Remove an optional trailing comment, then trim again.
  line="${line%%#*}"
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"

  [[ -z "${line}" ]] && continue

  atlas_entry=""
  read -r atlas_entry _ <<< "${line}"

  # Accept either a bare directory name or a full path.
  atlas_entry="${atlas_entry%/}"
  ATLAS_FOLDER="$(basename "${atlas_entry}")"

  # Extract gestational week 21 through 38.
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
    echo "WARN: Atlas directory not found:" >&2
    echo "      ${ATLAS_DIRECTORY}" >&2
    continue
  fi

  if [[ ! -d "${SURFACES_DIRECTORY}" ]]; then
    echo "WARN: Atlas SURFACES directory not found:" >&2
    echo "      ${SURFACES_DIRECTORY}" >&2
    continue
  fi

  ATLAS_COUNT=$((ATLAS_COUNT + 1))

  echo "Atlas: ${ATLAS_FOLDER}"
  echo "  Gestational week:  ${GW}"
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

      # Converted PLY is saved beside the source VTK with the same basename.
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

        echo "ERROR: Could not prepare PLY for ${LABEL}" >&2

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

      # Prevent a previous failed result from being reused.
      rm -f "${TMP_CSV}"

      declare -a EVALUATOR_ARGS

      EVALUATOR_ARGS=(
        "${EVALUATION_PYTHON_BIN}"
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

      if ! run_evaluation_python \
        "${EVALUATOR_ARGS[@]}"; then

        echo "ERROR: Evaluation failed for ${LABEL}" >&2

        EVALUATION_FAILED_COUNT=$(
          (EVALUATION_FAILED_COUNT + 1)
        )

        if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
          continue
        fi

        exit 1
      fi

      if ! append_csv_safely "${TMP_CSV}"; then
        echo "ERROR: Could not append result for ${LABEL}" >&2

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