#!/usr/bin/env bash
#
# Slurm directives
#SBATCH --job-name=surf_proc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --output=surf_proc-%x-%j.out
#SBATCH --error=surf_proc-%x-%j.err
#SBATCH --qos=longrunning
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=a12007396@unet.univie.ac.at

set -Euo pipefail

export LC_ALL=C
export LANG=C
export NO_COLOR=1
export TERM=dumb

# ---------------------------
# Config (EDIT THESE)
# ---------------------------
DATASET_DIR="${DATASET_DIR:-/project/fetal/2025_FetalSurface_swolff/2024_FETAL_CONTROL_JT}"

CONTAINER_IMG="${CONTAINER_IMG:-/home/cir/swolff/surface_reconstruction/surfaceprocessing.sif}"

BIND_PATHS="${BIND_PATHS:-}"

ATLAS_DIR="${ATLAS_DIR:-/project/fetal/2025_FetalSurface_swolff/2024_FETAL_CONTROL_JT/atlases}"

# CSV file mapping subjects to gestational age
GA_CSV="${GA_CSV:-${DATASET_DIR}/final_dataset.csv}"

# Valid atlas age range
ATLAS_AGE_MIN="${ATLAS_AGE_MIN:-21}"
ATLAS_AGE_MAX="${ATLAS_AGE_MAX:-37}"

# Padding dimensions
PAD_X="${PAD_X:-105}"
PAD_Y="${PAD_Y:-119}"
PAD_Z="${PAD_Z:-100}"

# Parallelism (simple, per-session)
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"

# ---------------------------
# Environment loading (host NiftyReg)
# ---------------------------
module purge || true
module load NiftyReg

# Determine container runtime (Apptainer preferred, fallback to Singularity)
if command -v apptainer >/dev/null 2>&1; then
  RUN_CTN="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  RUN_CTN="singularity"
else
  echo "ERROR: Neither apptainer nor singularity found in PATH." >&2
  exit 1
fi

if [[ ! -f "$CONTAINER_IMG" ]]; then
  echo "ERROR: Container image not found: $CONTAINER_IMG" >&2
  exit 1
fi

# ---------------------------
# Helpers
# ---------------------------
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }
err() { echo "[$(ts)] ERROR: $*" >&2; }

append_unique() {
  local line="$1" file="$2"
  grep -Fxq "$line" "$file" 2>/dev/null || echo "$line" >> "$file"
}

# Build bind arguments for the container
build_binds() {
  local -a args=()
  if [[ -n "${DATASET_DIR:-}" ]]; then
    args+=("--bind" "${DATASET_DIR}:${DATASET_DIR}")
  fi
  if [[ -n "${BIND_PATHS:-}" ]]; then
    for p in $BIND_PATHS; do
      args+=("--bind" "${p}:${p}")
    done
  fi
  echo "${args[@]}"
}

# Look up gestational week from CSV for a given subject and session.
# CSV is semicolon-delimited with header:
#   subject;fetal_mri_date;ga_at_mri;fetal_sex;tesla;quality;neuro;FASD;GE;test
# ses_id is "ses-YYYYMMDD", fetal_mri_date is "YYYY-MM-DD"
get_atlas_age() {
  local sub_id="$1"   # e.g. sub-101
  local ses_id="$2"   # e.g. ses-20190211

  if [[ ! -f "$GA_CSV" ]]; then
    err "GA CSV not found: $GA_CSV"
    return 1
  fi

  # Convert ses-YYYYMMDD -> YYYY-MM-DD for matching against fetal_mri_date
  local ses_num="${ses_id#ses-}"  # 20190211
  local ses_date="${ses_num:0:4}-${ses_num:4:2}-${ses_num:6:2}"  # 2019-02-11

  # Find the line matching both subject and date
  local ga_raw
  ga_raw="$(awk -F';' -v s="$sub_id" -v d="$ses_date" \
    'NR>1 && $1 == s && $2 == d {print $3; exit}' "$GA_CSV")"

  if [[ -z "$ga_raw" ]]; then
    err "No GA entry found for $sub_id / $ses_date in $GA_CSV"
    return 1
  fi

  # Extract integer weeks (part before '+')
  local ga_weeks="${ga_raw%%+*}"
  ga_weeks="${ga_weeks// /}"

  if ! [[ "$ga_weeks" =~ ^[0-9]+$ ]]; then
    err "Invalid GA value '$ga_raw' for $sub_id / $ses_date"
    return 1
  fi

  # Clamp to valid atlas range
  if (( ga_weeks < ATLAS_AGE_MIN )); then
    log "GA $ga_weeks for $sub_id/$ses_id below min ($ATLAS_AGE_MIN), clamping."
    ga_weeks="$ATLAS_AGE_MIN"
  elif (( ga_weeks > ATLAS_AGE_MAX )); then
    log "GA $ga_weeks for $sub_id/$ses_id above max ($ATLAS_AGE_MAX), clamping."
    ga_weeks="$ATLAS_AGE_MAX"
  fi

  echo "$ga_weeks"
}

# Run a Python command inside the container
run_container_python() {
  local -a binds
  IFS=' ' read -r -a binds <<< "$(build_binds)"

  "$RUN_CTN" exec \
    "${binds[@]}" \
    "$CONTAINER_IMG" \
    python "$@"
}

# Apply the inverse affine transform to a PLY mesh (inside the container)
apply_affine_to_mesh() {
  local aff_txt="$1"
  local in_ply="$2"
  local out_ply="$3"

  local -a binds
  IFS=' ' read -r -a binds <<< "$(build_binds)"

  "$RUN_CTN" exec \
    "${binds[@]}" \
    "$CONTAINER_IMG" \
    python -c "
import numpy as np, meshio, sys
A = np.loadtxt('${aff_txt}')
A = np.linalg.inv(A)
mesh = meshio.read('${in_ply}')
pts = mesh.points
pts_h = np.c_[pts, np.ones(len(pts))]
pts2 = (pts_h @ A.T)[:, :3]
mesh.points = pts2
meshio.write('${out_ply}', mesh)
print('Wrote:', '${out_ply}')
"
}

# ---------------------------
# Per-session processing
# ---------------------------
process_session() {
  local ses_dir="$1"
  local sub_id ses_id
  sub_id="$(basename "$(dirname "$ses_dir")")"
  ses_id="$(basename "$ses_dir")"
  local surf_dir="$ses_dir/surf"
  local svr_dir="$ses_dir/nesvor_svr"

  # Skip if already processed
  if grep -Fxq "$sub_id/$ses_id" "$CACHE_OK"; then
    log "Skip (cached): $sub_id/$ses_id"
    return 0
  fi

  log "Processing $sub_id/$ses_id"

  # ---------------------------
  # 1. Determine atlas age from CSV
  # ---------------------------
  local ATLAS_AGE
  ATLAS_AGE="$(get_atlas_age "$sub_id" "$ses_id")" || {
    append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
    return 0
  }
  log "Atlas age for $sub_id/$ses_id: GW${ATLAS_AGE}"

  # Validate atlas directory exists
  local atlas_subj_dir="${ATLAS_DIR}/CRL_ATLAS2017_GW${ATLAS_AGE}"
  if [[ ! -d "$atlas_subj_dir" ]]; then
    err "Atlas directory not found: $atlas_subj_dir"
    append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
    return 0
  fi

  # ---------------------------
  # 2. Check required input files
  # ---------------------------
  local INNII="${svr_dir}/svr_1.0mm.nii.gz"
  if [[ ! -s "$INNII" ]]; then
    err "Missing input NIfTI: $INNII"
    append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
    return 0
  fi

  if [[ ! -d "$surf_dir" ]]; then
    err "Missing surf directory: $surf_dir"
    append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
    return 0
  fi

  # Check that all four fsa6 PLY surfaces exist
  local hemi stype
  for hemi in L R; do
    for stype in white pial; do
      local ply_check="$surf_dir/${ses_id}.${hemi}.${stype}.fsa6.ply"
      if [[ ! -s "$ply_check" ]]; then
        err "Missing required surface: $ply_check"
        append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
        return 0
      fi
    done
  done

  # ---------------------------
  # 3a. Padding SVR volume
  # ---------------------------
  local FLO="${svr_dir}/svr_1mm_padded.nii.gz"
  if [[ ! -s "$FLO" ]]; then
    log "Padding SVR: $INNII -> $FLO"
    run_container_python \
      /home/cir/swolff/surface_reconstruction/Vox2Cortex/padding.py \
      "$INNII" "$FLO" "$PAD_X" "$PAD_Y" "$PAD_Z"
    if [[ ! -s "$FLO" ]]; then
      err "Padding failed, output not found: $FLO"
      append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
      return 0
    fi
  else
    log "Reusing padded SVR volume: $FLO"
  fi

  # ---------------------------
  # 3b. Padding segmentation
  # ---------------------------
  local SEG_INNII="${svr_dir}/svr_1mm-mask-brain_bounti-19.nii.gz"
  local FLO_SEG="${svr_dir}/svr_1mm-mask-brain_bounti-19_padded.nii.gz"

  if [[ ! -s "$SEG_INNII" ]]; then
    err "Missing segmentation input: $SEG_INNII"
    append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
    return 0
  fi

  if [[ ! -s "$FLO_SEG" ]]; then
    log "Padding segmentation: $SEG_INNII -> $FLO_SEG"
    run_container_python \
      /home/cir/swolff/surface_reconstruction/Vox2Cortex/padding.py \
      "$SEG_INNII" "$FLO_SEG" "$PAD_X" "$PAD_Y" "$PAD_Z"
    if [[ ! -s "$FLO_SEG" ]]; then
      err "Padding segmentation failed, output not found: $FLO_SEG"
      append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
      return 0
    fi
  else
    log "Reusing padded segmentation: $FLO_SEG"
  fi

  # ---------------------------
  # 4. Rigid registration (reg_aladin)
  # ---------------------------
  local REF="${atlas_subj_dir}/SR_1040${ATLAS_AGE}_1mm.nii.gz"
  local RES="${svr_dir}/svr_1mm_padded_reg.nii.gz"
  local AFF="${ses_dir}/niftyregAffine.txt"

  if [[ ! -s "$REF" ]]; then
    err "Atlas reference not found: $REF"
    append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
    return 0
  fi

  if [[ ! -s "$AFF" ]]; then
    log "Running reg_aladin: $FLO -> $REF"
    reg_aladin -ref "$REF" -flo "$FLO" -res "$RES" -aff "$AFF" -rigOnly -voff
    if [[ ! -s "$AFF" ]]; then
      err "reg_aladin failed, affine not produced: $AFF"
      append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
      return 0
    fi
  else
    log "Reusing affine: $AFF"
  fi

  # ---------------------------
  # 5. Resample segmentation with affine
  # ---------------------------
  local REF_SEG="${atlas_subj_dir}/SEGMENTATIONS/SEG_CRL_1040${ATLAS_AGE}_1mm.nii.gz"
  local RES_SEG="${svr_dir}/svr_1mm-mask-brain_bounti-19_padded_reg.nii.gz"

  if [[ ! -s "$RES_SEG" ]]; then
    if [[ ! -s "$REF_SEG" ]]; then
      err "Atlas segmentation reference not found: $REF_SEG"
      append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
      return 0
    fi
    log "Resampling segmentation: $FLO_SEG -> $RES_SEG"
    reg_resample -ref "$REF_SEG" -flo "$FLO_SEG" -trans "$AFF" -res "$RES_SEG" -inter 0 -voff
    if [[ ! -s "$RES_SEG" ]]; then
      err "reg_resample failed, output not found: $RES_SEG"
      append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
      return 0
    fi
  else
    log "Reusing resampled segmentation: $RES_SEG"
  fi

  # ---------------------------
  # 6. Apply inverse affine to all four surfaces
  # ---------------------------
  for hemi in L R; do
    for stype in white pial; do
      local in_ply="$surf_dir/${ses_id}.${hemi}.${stype}.fsa6.ply"
      local out_ply="$surf_dir/${ses_id}.${hemi}.${stype}.fsa6.reg.ply"

      if [[ ! -s "$out_ply" ]]; then
        log "Applying affine to surface: $in_ply -> $out_ply"
        apply_affine_to_mesh "$AFF" "$in_ply" "$out_ply"
        if [[ ! -s "$out_ply" ]]; then
          err "Affine transform failed for: $out_ply"
          append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
          return 0
        fi
      else
        log "Reusing registered surface: $out_ply"
      fi
    done
  done

  append_unique "$sub_id/$ses_id" "$CACHE_OK"
  log "Processed OK: $sub_id/$ses_id"
  return 0
}

# ---------------------------
# Initialization
# ---------------------------
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "ERROR: DATASET_DIR not found: $DATASET_DIR" >&2
  exit 1
fi

PREV_PROCESSED="$DATASET_DIR/processed.txt"
CACHE_OK="$DATASET_DIR/processed_reg.txt"
CACHE_FAIL="$DATASET_DIR/processed_reg_failed.txt"
touch "$CACHE_OK" "$CACHE_FAIL"

if [[ ! -s "$PREV_PROCESSED" ]]; then
  echo "ERROR: No previous processed.txt found or it is empty: $PREV_PROCESSED" >&2
  exit 1
fi

log "DATASET_DIR=$DATASET_DIR"
log "ATLAS_DIR=$ATLAS_DIR"
log "GA_CSV=$GA_CSV"
log "CONTAINER_IMG=$CONTAINER_IMG"
log "Runtime=$RUN_CTN  N_JOBS=$N_JOBS"
log "Reading sessions from $PREV_PROCESSED..."

# Build session list from processed.txt entries (sub-XXX/ses-XXXXXXXX per line)
mapfile -t SESSIONS < <(while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  ses_path="${DATASET_DIR}/${line}"
  if [[ -d "$ses_path" ]]; then
    echo "$ses_path"
  else
    echo "[WARNING] Directory not found, skipping: $ses_path" >&2
  fi
done < "$PREV_PROCESSED" | sort)

log "Found ${#SESSIONS[@]} sessions to process"

# ---------------------------
# Main loop
# ---------------------------
if [[ "$N_JOBS" -gt 1 ]]; then
  sem_count=0
  for ses in "${SESSIONS[@]}"; do
    process_session "$ses" &
    ((sem_count++))
    if (( sem_count % N_JOBS == 0 )); then
      wait -n
    fi
  done
  wait
else
  for ses in "${SESSIONS[@]}"; do
    process_session "$ses"
  done
fi

log "All done. See:"
log "  OK:     $CACHE_OK"
log "  FAILED: $CACHE_FAIL"