#!/usr/bin/env bash

#SBATCH --job-name=sub_atlas_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --output=sub_atlas_eval-%x-%j.out
#SBATCH --error=sub_atlas_eval-%x-%j.err
#SBATCH --qos=longrunning
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=a12007396@unet.univie.ac.at

set -euo pipefail


# =========================================================================
# Containers and repository
# =========================================================================

# Container used for metrics.
EVALUATION_CONTAINER_IMG="/home/cir/swolff/surface_reconstruction/vox2cortex.sif"

# Container used for PyVista VTK-to-PLY conversion.
CONVERSION_CONTAINER_IMG="/home/cir/swolff/surface_reconstruction/surfaceprocessing.sif"

REPO="/home/cir/swolff/surface_reconstruction/Vox2Cortex/vox2organ"

QUALITY_CONFIG="${REPO}/quality_config.json"

# Set the complete path to vtk2ply.py.
VTK2PLY_SCRIPT="/home/cir/swolff/surface_reconstruction/Vox2Cortex/vtk2ply.py"

RUN_CTN="${RUN_CTN:-singularity}"


# =========================================================================
# Evaluation environment in vox2cortex.sif
# =========================================================================

EVALUATION_PYTHON_BIN="python"

EVALUATION_CONDA_SH="/opt/conda/etc/profile.d/conda.sh"

EVALUATION_CONDA_ENV="torch_1_10"


# =========================================================================
# Conversion environment in surfaceprocessing.sif
# =========================================================================

# surfaceprocessing.sif is assumed to expose a working PyVista installation
# through this interpreter.
CONVERSION_PYTHON_BIN="python"

# Set to true only if the conversion container requires explicit Conda
# activation.
CONVERSION_USE_CONDA="false"

CONVERSION_CONDA_SH="/opt/conda/etc/profile.d/conda.sh"

CONVERSION_CONDA_ENV="surfaceprocessing"


# =========================================================================
# Subject input
# =========================================================================

ROOT="/project/fetal/2025_FetalSurface_swolff/2024_FETAL_CONTROL_JT"

# Only these subjects/sessions are evaluated.
#
# Supported line forms:
#
# sub-18 ses-20191107
# sub-18/ses-20191107
PROCESSED="${ROOT}/processed.txt"

# Metadata CSV containing at least:
#
# subject,date_trans,age_trans
# sub-100000,20120110,28
#
# An unnamed Pandas index column is permitted and ignored.
METADATA_CSV="${ROOT}/dataset_age.csv"

# Subject surface template.
#
# Available placeholders:
#
# {ROOT}
# {SUB}
# {SES}
# {H}
# {HEMI}
# {FS_HEMI}
# {SIDE}
# {SURFACE}
# {MAT}
#
# Example:
#
# /project/.../sub-18/ses-20191107/surf/
# ses-20191107.R.pial.fsa6.reg.ply
PRED_MESH_TEMPLATE="{ROOT}/{SUB}/{SES}/surf/{SES}.{H}.{SURFACE}.native.surf.ply"


# =========================================================================
# Atlas input
# =========================================================================

ATLAS_ROOT="${ROOT}/atlases"

ATLAS_SURFACE_DIRECTORY="SURFACES"

# GW21 -> 104021
# GW38 -> 104038
ATLAS_FILE_ID_PREFIX="1040"

MIN_ATLAS_GW=21
MAX_ATLAS_GW=38


# =========================================================================
# Conversion behavior
# =========================================================================

# If false, any existing nonempty atlas PLY is reused.
#
# If true, an atlas PLY is regenerated when its VTK source is newer.
RECONVERT_IF_VTK_NEWER="false"


# =========================================================================
# Evaluation options
# =========================================================================

N_POINTS=10000

SEED=42

DEVICE="cuda"

HEMISPHERES=(L R)

SURFACES=(white pial)


# =========================================================================
# Output
# =========================================================================

OUT_CSV="/home/cir/swolff/surface_reconstruction/subject_atlas_scores.csv"

RESET_OUTPUT="false"

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
  echo "ERROR: Metrics file not found:" >&2
  echo "       ${REPO}/utils/eval_metrics.py" >&2
  exit 1
fi

if [[ ! -f "${QUALITY_CONFIG}" ]]; then
  echo "ERROR: Quality configuration not found:" >&2
  echo "       ${QUALITY_CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${VTK2PLY_SCRIPT}" ]]; then
  echo "ERROR: VTK-to-PLY script not found:" >&2
  echo "       ${VTK2PLY_SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${ROOT}" ]]; then
  echo "ERROR: Dataset root not found: ${ROOT}" >&2
  exit 1
fi

if [[ ! -d "${ATLAS_ROOT}" ]]; then
  echo "ERROR: Atlas root not found: ${ATLAS_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${PROCESSED}" ]]; then
  echo "ERROR: Processed list not found: ${PROCESSED}" >&2
  exit 1
fi

if [[ ! -f "${METADATA_CSV}" ]]; then
  echo "ERROR: Subject metadata CSV not found:" >&2
  echo "       ${METADATA_CSV}" >&2
  exit 1
fi

if ! command -v "${RUN_CTN}" >/dev/null 2>&1; then
  echo "ERROR: Container runner not found: ${RUN_CTN}" >&2
  exit 1
fi

if [[ "${CONVERSION_USE_CONDA}" != "true" &&
      "${CONVERSION_USE_CONDA}" != "false" ]]; then
  echo "ERROR: CONVERSION_USE_CONDA must be true or false." >&2
  exit 1
fi


# =========================================================================
# Temporary workspace and output setup
# =========================================================================

OUT_DIR="$(dirname "${OUT_CSV}")"
mkdir -p "${OUT_DIR}"

OUT_CSV_ABS="$(readlink -m "${OUT_CSV}")"
OUT_LOCK_FILE="${OUT_CSV_ABS}.lock"

if [[ "${RESET_OUTPUT}" == "true" ]]; then
  rm -f \
    "${OUT_CSV_ABS}" \
    "${OUT_LOCK_FILE}"
fi

TMP_DIR="$(
  mktemp -d \
    -t "subject_atlas_eval_${SLURM_JOB_ID:-noj}_XXXXXX"
)"

trap 'rm -rf "${TMP_DIR}"' EXIT


# =========================================================================
# Build a normalized subject/session-to-GW lookup table
# =========================================================================

# The input CSV may contain more subjects than processed.txt. That is
# intentional. The Python helper reads all metadata but the Bash loop later
# looks up only processed.txt cases.
#
# Output format:
#
# sub-100000<TAB>ses-20120110<TAB>28<TAB>28
#
# Columns:
#   subject
#   normalized session
#   original gestational age
#   clamped atlas gestational week
GW_LOOKUP_TSV="${TMP_DIR}/gw_lookup.tsv"

python3 - \
  "${METADATA_CSV}" \
  "${GW_LOOKUP_TSV}" \
  "${MIN_ATLAS_GW}" \
  "${MAX_ATLAS_GW}" <<'PY'
import csv
import math
import sys
from pathlib import Path


metadata_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
minimum_atlas_gw = int(sys.argv[3])
maximum_atlas_gw = int(sys.argv[4])


def normalize_subject(value):
    value = str(value).strip()

    if not value:
        raise ValueError("Empty subject value")

    return value


def normalize_date(value):
    value = str(value).strip()

    if value.lower().startswith("ses-"):
        value = value[4:]

    # CSV parsers or spreadsheet exports may represent the date as
    # 20120110.0. Remove only a terminal decimal-zero suffix.
    if value.endswith(".0"):
        value = value[:-2]

    if not value.isdigit():
        raise ValueError(
            "date_trans must contain a YYYYMMDD-style integer; got {!r}"
            .format(value)
        )

    return value


def normalize_age(value):
    value = str(value).strip()

    if not value:
        raise ValueError("Empty age_trans value")

    age = float(value)

    if not math.isfinite(age):
        raise ValueError(
            "Non-finite age_trans value: {!r}".format(value)
        )

    # The requested matching policy is exact integer GW with clamping only.
    if not age.is_integer():
        raise ValueError(
            "age_trans must be an integer gestational week for exact "
            "matching; got {!r}".format(value)
        )

    return int(age)


records = {}

with metadata_path.open("r", newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle)

    if reader.fieldnames is None:
        raise RuntimeError("Metadata CSV has no header")

    normalized_fields = {
        str(name).strip(): name
        for name in reader.fieldnames
        if name is not None
    }

    required = ["subject", "date_trans", "age_trans"]

    missing = [
        name
        for name in required
        if name not in normalized_fields
    ]

    if missing:
        raise RuntimeError(
            "Metadata CSV is missing required columns: {}. "
            "Available columns: {}"
            .format(
                ", ".join(missing),
                ", ".join(normalized_fields),
            )
        )

    for line_number, row in enumerate(reader, start=2):
        try:
            subject = normalize_subject(
                row[normalized_fields["subject"]]
            )

            date = normalize_date(
                row[normalized_fields["date_trans"]]
            )

            subject_gw = normalize_age(
                row[normalized_fields["age_trans"]]
            )
        except Exception as exc:
            raise RuntimeError(
                "Invalid metadata at CSV line {}: {}"
                .format(line_number, exc)
            ) from exc

        atlas_gw = max(
            minimum_atlas_gw,
            min(maximum_atlas_gw, subject_gw),
        )

        session = "ses-{}".format(date)
        key = (subject, session)
        value = (subject_gw, atlas_gw)

        if key in records and records[key] != value:
            raise RuntimeError(
                "Conflicting duplicate metadata for {}/{}: {} versus {}"
                .format(
                    subject,
                    session,
                    records[key],
                    value,
                )
            )

        records[key] = value


with output_path.open("w", encoding="utf-8") as handle:
    for (subject, session), (subject_gw, atlas_gw) in sorted(
        records.items()
    ):
        handle.write(
            "{}\t{}\t{}\t{}\n".format(
                subject,
                session,
                subject_gw,
                atlas_gw,
            )
        )
PY


# Load the normalized table into associative arrays.
declare -A SUBJECT_GW_BY_CASE
declare -A ATLAS_GW_BY_CASE

while IFS=$'\t' read -r map_sub map_ses map_subject_gw map_atlas_gw; do
  [[ -z "${map_sub}" || -z "${map_ses}" ]] && continue

  map_key="${map_sub}|${map_ses}"

  SUBJECT_GW_BY_CASE["${map_key}"]="${map_subject_gw}"
  ATLAS_GW_BY_CASE["${map_key}"]="${map_atlas_gw}"
done < "${GW_LOOKUP_TSV}"


# =========================================================================
# Path construction
# =========================================================================

render_subject_template() {
  local template="$1"
  local root="$2"
  local sub="$3"
  local ses="$4"
  local hemisphere="$5"
  local surface="$6"

  local fs_hemisphere

  if [[ "${hemisphere}" == "L" ]]; then
    fs_hemisphere="lh"
  else
    fs_hemisphere="rh"
  fi

  local path="${template}"

  path="${path//\{ROOT\}/${root}}"
  path="${path//\{SUB\}/${sub}}"
  path="${path//\{SES\}/${ses}}"
  path="${path//\{H\}/${hemisphere}}"
  path="${path//\{HEMI\}/${hemisphere}}"
  path="${path//\{FS_HEMI\}/${fs_hemisphere}}"
  path="${path//\{SIDE\}/${fs_hemisphere}}"
  path="${path//\{SURFACE\}/${surface}}"
  path="${path//\{MAT\}/${surface}}"

  printf '%s' "${path}"
}


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
  local atlas_gw="$1"
  local hemisphere="$2"
  local surface="$3"

  local atlas_folder
  local atlas_file_id
  local surface_stem

  atlas_folder="CRL_ATLAS2017_GW${atlas_gw}"
  atlas_file_id="${ATLAS_FILE_ID_PREFIX}${atlas_gw}"

  surface_stem="$(
    atlas_surface_stem \
      "${hemisphere}" \
      "${surface}"
  )"

  printf '%s/%s/%s/%s_HR_%s.vtk' \
    "${ATLAS_ROOT}" \
    "${atlas_folder}" \
    "${ATLAS_SURFACE_DIRECTORY}" \
    "${surface_stem}" \
    "${atlas_file_id}"
}


# =========================================================================
# Bind mounts
# =========================================================================

declare -a BINDS
declare -a NV_FLAG
declare -A BOUND_PATHS

BINDS=()
NV_FLAG=()

if [[ "${DEVICE}" == "cuda" ]]; then
  NV_FLAG=(--nv)
fi


add_bind() {
  local path="$1"

  [[ -z "${path}" ]] && return

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


# ROOT must be writable in the conversion container because the converted
# atlas PLY files are saved beside their source VTK files.
add_bind "${ROOT}"
add_bind "${REPO}"
add_bind "${OUT_DIR}"
add_bind "$(dirname "${QUALITY_CONFIG}")"
add_bind "$(dirname "${VTK2PLY_SCRIPT}")"


# =========================================================================
# Container runners
# =========================================================================

run_evaluation_python() {
  local quoted_command
  local quoted_conda_sh
  local quoted_conda_environment
  local quoted_pythonpath
  local container_pythonpath
  local inner_command

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
    "${NV_FLAG[@]}" \
    "${BINDS[@]}" \
    "${EVALUATION_CONTAINER_IMG}" \
    /bin/bash -lc "${inner_command}"
}


run_conversion_python() {
  local quoted_command
  local inner_command

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
# Verify conversion environment
# =========================================================================

echo "Checking PyVista in conversion container..."

if ! run_conversion_python \
  "${CONVERSION_PYTHON_BIN}" \
  -c \
  "import sys; import pyvista as pv; print('Python:', sys.executable); print('PyVista:', pv.__version__)"; then

  echo "ERROR: PyVista is unavailable in the conversion container." >&2
  echo "Container: ${CONVERSION_CONTAINER_IMG}" >&2
  exit 1
fi

echo


# =========================================================================
# Ensure atlas PLY exists
# =========================================================================

ensure_atlas_ply_exists() {
  local vtk_path="$1"
  local ply_path="$2"

  if [[ ! -f "${vtk_path}" ]]; then
    echo "ERROR: Atlas VTK not found: ${vtk_path}" >&2
    return 1
  fi

  if [[ -s "${ply_path}" ]]; then
    if [[ "${RECONVERT_IF_VTK_NEWER}" == "true" &&
          "${vtk_path}" -nt "${ply_path}" ]]; then
      echo "    Existing atlas PLY is older than VTK; reconverting."
    else
      echo "    Reusing atlas PLY: ${ply_path}"
      return 0
    fi
  fi

  local conversion_lock="${ply_path}.conversion.lock"

  (
    flock -x 201

    # Check again after acquiring the lock.
    if [[ -s "${ply_path}" ]]; then
      if [[ "${RECONVERT_IF_VTK_NEWER}" == "true" &&
            "${vtk_path}" -nt "${ply_path}" ]]; then
        echo "    Locked PLY is older than VTK; reconverting."
      else
        echo "    Atlas PLY was created by another process."
        exit 0
      fi
    fi

    local ply_directory
    local ply_basename
    local temporary_ply

    ply_directory="$(dirname "${ply_path}")"
    ply_basename="$(basename "${ply_path}" .ply)"

    temporary_ply="${ply_directory}/.${ply_basename}.conversion.${SLURM_JOB_ID:-noj}.$$.${RANDOM}.ply"

    rm -f "${temporary_ply}"

    cleanup_conversion() {
      rm -f "${temporary_ply}"
    }

    trap cleanup_conversion EXIT

    echo "    Converting atlas VTK to PLY"
    echo "      VTK: ${vtk_path}"
    echo "      PLY: ${ply_path}"

    if ! run_conversion_python \
      "${CONVERSION_PYTHON_BIN}" \
      "${VTK2PLY_SCRIPT}" \
      "${vtk_path}" \
      "${temporary_ply}"; then

      echo "ERROR: Atlas conversion failed: ${vtk_path}" >&2
      exit 1
    fi

    if [[ ! -s "${temporary_ply}" ]]; then
      echo "ERROR: Converter did not create a nonempty PLY:" >&2
      echo "       ${temporary_ply}" >&2
      exit 1
    fi

    mv -f \
      "${temporary_ply}" \
      "${ply_path}"

    trap - EXIT

    if [[ ! -s "${ply_path}" ]]; then
      echo "ERROR: Final atlas PLY is missing or empty:" >&2
      echo "       ${ply_path}" >&2
      exit 1
    fi
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
        echo "Existing header:" >&2
        echo "${existing_header}" >&2
        echo "New header:" >&2
        echo "${new_header}" >&2
        echo "Use a new output file or set RESET_OUTPUT=true." >&2
        exit 1
      fi

      tail -n +2 "${temporary_csv}" >> "${OUT_CSV_ABS}"
    fi
  ) 200>"${OUT_LOCK_FILE}"
}


# =========================================================================
# Counters
# =========================================================================

CASE_COUNT=0
SURFACE_COUNT=0
MISSING_METADATA_COUNT=0
MISSING_PREDICTION_COUNT=0
MISSING_ATLAS_COUNT=0
CONVERSION_FAILURE_COUNT=0
EVALUATION_FAILURE_COUNT=0


# =========================================================================
# Main subject loop
# =========================================================================

echo "Starting subject-to-age-matched-atlas evaluation"
echo "Processed list:       ${PROCESSED}"
echo "Metadata CSV:         ${METADATA_CSV}"
echo "Atlas root:           ${ATLAS_ROOT}"
echo "Evaluation container: ${EVALUATION_CONTAINER_IMG}"
echo "Conversion container: ${CONVERSION_CONTAINER_IMG}"
echo "Output CSV:           ${OUT_CSV_ABS}"
echo


while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
  raw_line="${raw_line%$'\r'}"

  line="${raw_line#"${raw_line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"

  [[ -z "${line}" ]] && continue
  [[ "${line}" == \#* ]] && continue

  line="${line%%#*}"
  line="${line#"${line%%[![:space:]]*}"}"
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
    echo "WARN: Could not parse processed-list line:" >&2
    echo "      ${raw_line}" >&2
    continue
  fi

  CASE_KEY="${SUB}|${SES}"

  if [[ -z "${SUBJECT_GW_BY_CASE[${CASE_KEY}]+x}" ]]; then
    echo "WARN: No metadata CSV entry for ${SUB}/${SES}; skipping." >&2

    MISSING_METADATA_COUNT=$(
      (MISSING_METADATA_COUNT + 1)
    )
    continue
  fi

  SUBJECT_GW="${SUBJECT_GW_BY_CASE[${CASE_KEY}]}"
  ATLAS_GW="${ATLAS_GW_BY_CASE[${CASE_KEY}]}"
  ATLAS_FOLDER="CRL_ATLAS2017_GW${ATLAS_GW}"

  ATLAS_DIRECTORY="${ATLAS_ROOT}/${ATLAS_FOLDER}"

  if [[ ! -d "${ATLAS_DIRECTORY}" ]]; then
    echo "WARN: Matched atlas directory does not exist:" >&2
    echo "      ${ATLAS_DIRECTORY}" >&2

    MISSING_ATLAS_COUNT=$(
      (MISSING_ATLAS_COUNT + 1)
    )
    continue
  fi

  CASE_COUNT=$((CASE_COUNT + 1))

  echo "Case: ${SUB} ${SES}"
  echo "  Subject GW: ${SUBJECT_GW}"
  echo "  Atlas GW:   ${ATLAS_GW}"
  echo "  Atlas:      ${ATLAS_FOLDER}"

  for HEMISPHERE in "${HEMISPHERES[@]}"; do
    for SURFACE in "${SURFACES[@]}"; do
      PRED_PATH="$(
        render_subject_template \
          "${PRED_MESH_TEMPLATE}" \
          "${ROOT}" \
          "${SUB}" \
          "${SES}" \
          "${HEMISPHERE}" \
          "${SURFACE}"
      )"

      if [[ ! -f "${PRED_PATH}" ]]; then
        echo "  Missing subject surface: ${PRED_PATH}" >&2

        MISSING_PREDICTION_COUNT=$(
          (MISSING_PREDICTION_COUNT + 1)
        )
        continue
      fi

      ATLAS_VTK_PATH="$(
        atlas_vtk_path \
          "${ATLAS_GW}" \
          "${HEMISPHERE}" \
          "${SURFACE}"
      )"

      ATLAS_PLY_PATH="${ATLAS_VTK_PATH%.vtk}.ply"

      LABEL="${SUB}_${SES}_${HEMISPHERE}_${SURFACE}_atlasGW${ATLAS_GW}"

      if [[ ! -f "${ATLAS_VTK_PATH}" &&
            ! -s "${ATLAS_PLY_PATH}" ]]; then
        echo "  Missing atlas source and converted surface:" >&2
        echo "    VTK: ${ATLAS_VTK_PATH}" >&2
        echo "    PLY: ${ATLAS_PLY_PATH}" >&2

        MISSING_ATLAS_COUNT=$(
          (MISSING_ATLAS_COUNT + 1)
        )
        continue
      fi

      # If a valid PLY exists, it can be used even when the original VTK is
      # no longer present. Otherwise, conversion requires the VTK.
      if [[ ! -s "${ATLAS_PLY_PATH}" ]]; then
        if ! ensure_atlas_ply_exists \
          "${ATLAS_VTK_PATH}" \
          "${ATLAS_PLY_PATH}"; then

          echo "ERROR: Could not prepare atlas PLY for ${LABEL}" >&2

          CONVERSION_FAILURE_COUNT=$(
            (CONVERSION_FAILURE_COUNT + 1)
          )

          if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
            continue
          fi

          exit 1
        fi
      else
        if [[ "${RECONVERT_IF_VTK_NEWER}" == "true" &&
              -f "${ATLAS_VTK_PATH}" &&
              "${ATLAS_VTK_PATH}" -nt "${ATLAS_PLY_PATH}" ]]; then

          if ! ensure_atlas_ply_exists \
            "${ATLAS_VTK_PATH}" \
            "${ATLAS_PLY_PATH}"; then

            echo "ERROR: Atlas reconversion failed for ${LABEL}" >&2

            CONVERSION_FAILURE_COUNT=$(
              (CONVERSION_FAILURE_COUNT + 1)
            )

            if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
              continue
            fi

            exit 1
          fi
        else
          echo "  Reusing atlas PLY: ${ATLAS_PLY_PATH}"
        fi
      fi

      TMP_CSV="${TMP_DIR}/${LABEL}.csv"
      rm -f "${TMP_CSV}"

      declare -a EVALUATOR_ARGS

      EVALUATOR_ARGS=(
        "${EVALUATION_PYTHON_BIN}"
        "${REPO}/evaluate_mesh_pair.py"
        --mode "pair"
        --pred "${PRED_PATH}"
        --gt "${ATLAS_PLY_PATH}"
        --skip_gt_qa
        --config "${QUALITY_CONFIG}"
        --subject "${SUB}"
        --session "${SES}"
        --subject_gw "${SUBJECT_GW}"
        --atlas_gw "${ATLAS_GW}"
        --atlas_folder "${ATLAS_FOLDER}"
        --hemisphere "${HEMISPHERE}"
        --surface "${SURFACE}"
        --label "${LABEL}"
        --n_points "${N_POINTS}"
        --seed "${SEED}"
        --device "${DEVICE}"
        --out_csv "${TMP_CSV}"
      )

      echo "  Evaluating ${HEMISPHERE} ${SURFACE}"
      echo "    Subject: ${PRED_PATH}"
      echo "    Atlas:   ${ATLAS_PLY_PATH}"

      if ! run_evaluation_python \
        "${EVALUATOR_ARGS[@]}"; then

        echo "ERROR: Evaluation failed for ${LABEL}" >&2

        EVALUATION_FAILURE_COUNT=$(
          (EVALUATION_FAILURE_COUNT + 1)
        )

        if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
          continue
        fi

        exit 1
      fi

      if ! append_csv_safely "${TMP_CSV}"; then
        echo "ERROR: Could not append output for ${LABEL}" >&2

        EVALUATION_FAILURE_COUNT=$(
          (EVALUATION_FAILURE_COUNT + 1)
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
done < "${PROCESSED}"


# =========================================================================
# Summary
# =========================================================================

echo "Subject-to-atlas evaluation completed."
echo "Cases with valid metadata:       ${CASE_COUNT}"
echo "Surfaces evaluated:              ${SURFACE_COUNT}"
echo "Cases missing metadata:          ${MISSING_METADATA_COUNT}"
echo "Missing subject surfaces:        ${MISSING_PREDICTION_COUNT}"
echo "Missing atlas surfaces/folders:  ${MISSING_ATLAS_COUNT}"
echo "Atlas conversion failures:       ${CONVERSION_FAILURE_COUNT}"
echo "Evaluation failures:             ${EVALUATION_FAILURE_COUNT}"
echo "Output CSV:                      ${OUT_CSV_ABS}"