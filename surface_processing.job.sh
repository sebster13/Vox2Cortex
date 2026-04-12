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
# Optional: uncomment and set a partition/account if required by your cluster
# #SBATCH --partition=standard
# If you want to use multiple CPU cores within this job, also set:
# #SBATCH --cpus-per-task=4

# Surface processing pipeline:
# - Requires: module environment with FreeSurfer, Python3 for surface_conversion.py
# - Caching: processed.txt and processed_failed.txt in dataset root
# - Continues on failure for each session (logs and moves on)

set -Euo pipefail

export LC_ALL=C
export LANG=C
export NO_COLOR=1
export TERM=dumb
# ---------------------------
# Config (EDIT THESE)
# ---------------------------
# Dataset root (contains sub-*/ses-* hierarchy) — independent of the .sif location
DATASET_DIR="${DATASET_DIR:-/project/fetal/2025_FetalSurface_swolff/2024_FETAL_CONTROL_JT}"
# DATASET_DIR="${DATASET_DIR:-/home/cir/swolff/surface_reconstruction}"

# Path to your Singularity/Apptainer image (.sif) with Python tools (any location)
CONTAINER_IMG="${CONTAINER_IMG:-/home/cir/swolff/surface_reconstruction/surfaceprocessing.sif}"

# If your data span multiple unrelated roots and you want them visible in the container,
# list them space-separated here (optional). Each will be bound to the same path inside.
# Example: BIND_PATHS="/project /scratch"
BIND_PATHS="${BIND_PATHS:-}"

# Atlas configuration (host FreeSurfer usage)
ATLAS_AGE="${ATLAS_AGE:-39}"
ATLAS_DIR="${ATLAS_DIR:-/home/cir/swolff/surface_reconstruction}"
TRG_ATLAS_SUBJ="${TRG_ATLAS_SUBJ:-atlases/${ATLAS_AGE}_fsa6}"  # must exist under SUBJECTS_DIR

# Reuse options
REUSE_INFLATE="${REUSE_INFLATE:-1}"   # 1 = reuse existing inflated/sphere files if present
REUSE_SPHERE="${REUSE_SPHERE:-1}"     # 1 = reuse sphere if present

# ---------------------------
# Evaluation config (mesh pair scoring, uses separate container)
# ---------------------------
EVAL_CONTAINER_IMG="${EVAL_CONTAINER_IMG:-/home/cir/swolff/surface_reconstruction/vox2cortex.sif}"
REPO="${REPO:-/home/cir/swolff/surface_reconstruction/Vox2Cortex/vox2organ}"
OUT_CSV="${OUT_CSV:-/home/cir/swolff/surface_reconstruction/dataset_scores.csv}"
N_POINTS="${N_POINTS:-10000}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
SELF_INTERSECTIONS="${SELF_INTERSECTIONS:-false}"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-torch_1_10}"

# Parallelism (simple, per-session)
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"

# ---------------------------
# Environment loading (host FreeSurfer)
# ---------------------------
module purge || true
module load FreeSurfer

if [[ -z "${FREESURFER_HOME:-}" || ! -d "$FREESURFER_HOME" ]]; then
  echo "ERROR: FREESURFER_HOME not set after module load. Check your module configuration." >&2
  exit 1
fi

# SUBJECTS_DIR points to dataset root so FreeSurfer sees sub-XXX/ses-YYYYMMDD
export SUBJECTS_DIR="${SUBJECTS_DIR:-$DATASET_DIR}"

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

# Safe, idempotent append (avoid duplicates)
append_unique() {
  local line="$1" file="$2"
  grep -Fxq "$line" "$file" 2>/dev/null || echo "$line" >> "$file"
}

# Build bind arguments for the container from a list of host paths
build_bind_args() {
  local -a args=()
  # Always bind the dataset root for safety
  if [[ -n "$DATASET_DIR" ]]; then
    args+=("--bind" "${DATASET_DIR}:${DATASET_DIR}")
  fi
  # Optional extra binds from BIND_PATHS (space-separated)
  if [[ -n "$BIND_PATHS" ]]; then
    for p in $BIND_PATHS; do
      args+=("--bind" "${p}:${p}")
    done
  fi
  echo "${args[@]}"
}

bash

# Containerized Python conversion: VTK -> GIfTI with explicit --outbase
vtk_to_gii() {
  local in_vtk="$1"    # absolute path to input .vtk
  local out_gii="$2"   # absolute path to desired output .gii
  local outbase="${out_gii%.gii}"

  local in_dir out_dir
  in_dir="$(dirname "$in_vtk")"
  out_dir="$(dirname "$out_gii")"

  # Build bind list: dataset root (if set), input dir, output dir, plus optional extra binds
  local -a binds
  binds=()
  # Always bind DATASET_DIR if set (safe default)
  if [[ -n "${DATASET_DIR:-}" ]]; then
    binds+=("--bind" "${DATASET_DIR}:${DATASET_DIR}")
  fi
  # Optional extras from BIND_PATHS
  if [[ -n "${BIND_PATHS:-}" ]]; then
    for p in $BIND_PATHS; do
      binds+=("--bind" "${p}:${p}")
    done
  fi
  # Ensure input/output dirs are visible
  binds+=("--bind" "${in_dir}:${in_dir}")
  if [[ "$out_dir" != "$in_dir" ]]; then
    binds+=("--bind" "${out_dir}:${out_dir}")
  fi

  "$RUN_CTN" exec \
    "${binds[@]}" \
    "$CONTAINER_IMG" \
    python /home/cir/swolff/surface_reconstruction/surface_conversion.py "$in_vtk" --gii --outbase "$outbase"
}

# Containerized Python conversion: VTK -> PLY
vtk_to_ply() {
  local out_vtk="$1"    # absolute path to input .vtk
  local out_ply="$2"   # absolute path to desired output .ply

  local in_dir out_dir
  in_dir="$(dirname "$out_vtk")"
  out_dir="$(dirname "$out_ply")"

  # Build bind list: dataset root (if set), input dir, output dir, plus optional extra binds
  local -a binds
  binds=()
  # Always bind DATASET_DIR if set (safe default)
  if [[ -n "${DATASET_DIR:-}" ]]; then
    binds+=("--bind" "${DATASET_DIR}:${DATASET_DIR}")
  fi
  # Optional extras from BIND_PATHS
  if [[ -n "${BIND_PATHS:-}" ]]; then
    for p in $BIND_PATHS; do
      binds+=("--bind" "${p}:${p}")
    done
  fi
  # Ensure input/output dirs are visible
  binds+=("--bind" "${in_dir}:${in_dir}")
  if [[ "$out_dir" != "$in_dir" ]]; then
    binds+=("--bind" "${out_dir}:${out_dir}")
  fi

  "$RUN_CTN" exec \
    "${binds[@]}" \
    "$CONTAINER_IMG" \
    python /home/cir/swolff/surface_reconstruction/vtk2ply.py "$out_vtk" "$out_ply"
}

# ---------------------------
# Per-session processing
# ---------------------------
process_session() {
  local ses_dir="$1"   # e.g., /.../sub-101/ses-20190211
  local sub_id ses_id
  sub_id="$(basename "$(dirname "$ses_dir")")"   # sub-101
  ses_id="$(basename "$ses_dir")"               # ses-20190211
  local surf_dir="$ses_dir/surf"
  local surface_dir="$ses_dir/surface"

  # Skip if already processed
  if grep -Fxq "$sub_id/$ses_id" "$CACHE_OK"; then
    log "Skip (cached): $sub_id/$ses_id"
    return 0
  fi

  # Ensure a surf folder exists (rename surface -> surf if needed)
  if [[ -d "$surface_dir" && ! -d "$surf_dir" ]]; then
    mv "$surface_dir" "$surf_dir"
  fi

  if [[ ! -d "$surf_dir" ]]; then
    err "Missing surface folder for $sub_id/$ses_id"
    append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
    return 0  # continue
  fi

  # Check if surf folder is empty
  if [[ -z "$(ls -A "$surf_dir")" ]]; then
    err "Empty surf folder for $sub_id/$ses_id"
    append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
    return 0
  fi

  # Required VTK files
  local req=( \
    "${ses_id}.L.pial.native.surf.vtk" \
    "${ses_id}.L.white.native.surf.vtk" \
    "${ses_id}.R.pial.native.surf.vtk" \
    "${ses_id}.R.white.native.surf.vtk" \
  )
  local missing=0
  for f in "${req[@]}"; do
    if [[ ! -s "$surf_dir/$f" ]]; then
      err "Missing required file: $surf_dir/$f"
      missing=1
    fi
  done
  if [[ $missing -ne 0 ]]; then
    append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
    return 0
  fi

  # Process four surfaces: L/R x pial/white
  local hemi side surf in_vtk out_gii out_fs
  for hemi in L R; do
    if [[ "$hemi" == "L" ]]; then side="lh"; else side="rh"; fi
    for surf in white pial; do
      in_vtk="$surf_dir/${ses_id}.${hemi}.${surf}.native.surf.vtk"
      out_gii="$surf_dir/${ses_id}.${hemi}.${surf}.native.surf.gii"
      out_fs="$surf_dir/${side}.${surf}"

      # Convert native VTK -> PLY using vtk2ply.py (preserves original geometry as-is)
      local native_ply="$surf_dir/${ses_id}.${hemi}.${surf}.native.surf.ply"
      if [[ ! -s "$native_ply" ]]; then
        log "Native VTK -> PLY (container): $in_vtk -> $native_ply"
        vtk_to_ply "$in_vtk" "$native_ply"
        if [[ ! -s "$native_ply" ]]; then
          err "Expected native PLY not found after conversion: $native_ply"
          append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
          return 0
        fi
      else
        log "Reusing native PLY: $native_ply"
      fi

      # Convert VTK -> GIfTI in the container
      if [[ ! -s "$out_gii" ]]; then
        log "Converting VTK->GIfTI (container): $in_vtk -> $out_gii"
        vtk_to_gii "$in_vtk" "$out_gii"
        if [[ ! -s "$out_gii" ]]; then
          err "Expected GIfTI not found after conversion: $out_gii"
          append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
          return 0
        fi
      else
        log "Reusing existing GIfTI: $out_gii"
      fi

      # Convert GIfTI -> FreeSurfer surface (host FreeSurfer)
      if [[ ! -s "$out_fs" ]]; then
        log "GIfTI->FS: $out_gii -> $out_fs"
        mris_convert "$out_gii" "$out_fs"
      else
        log "Reusing FS surface: $out_fs"
      fi

      # Euler characteristic (FreeSurfer mris_euler_number)
      if [[ -s "$out_fs" ]]; then
        local euler_out
        euler_out="$(mris_euler_number "$out_fs" 2>&1)"
        log "Euler($side.$surf): $euler_out"
      fi

      # White-surface-only steps
      if [[ "$surf" == "white" ]]; then
        local inflated="$surf_dir/${side}.inflated"
        local smoothwm="$surf_dir/${side}.smoothwm"
        local sphere="$surf_dir/${side}.sphere"
        local curv="$surf_dir/${side}.curv"
        local sphere_reg="$surf_dir/${side}.sphere.reg"
        local atlas_sphere="${ATLAS_DIR}/atlases/${ATLAS_AGE}/surf/${side}.sphere"
        local trg_atlas="${TRG_ATLAS_SUBJ}"

        if [[ -s "$inflated" && "$REUSE_INFLATE" == "1" ]]; then
          log "Reusing inflated: $inflated"
        else
          log "Inflating: $out_fs -> $inflated"
          mris_inflate "$out_fs" "$inflated"
        fi

        if [[ ! -s "$smoothwm" ]]; then
          log "Smoothing white -> smoothwm: $out_fs -> $smoothwm"
          mris_smooth "$out_fs" "$smoothwm"
        else
          log "Reusing smoothwm: $smoothwm"
        fi

        if [[ -s "$sphere" && "$REUSE_SPHERE" == "1" ]]; then
          log "Reusing sphere: $sphere"
        else
          log "Creating sphere: $inflated -> $sphere"
          mris_sphere "$inflated" "$sphere"
        fi

        if [[ ! -s "$curv" ]]; then
          log "Computing curvature: $out_fs"
          mris_curvature "$out_fs"
        else
          log "Reusing curvature: $curv"
        fi

        if [[ ! -s "$sphere_reg" ]]; then
          if [[ ! -s "$atlas_sphere" ]]; then
            err "Atlas sphere not found: $atlas_sphere"
            append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
            return 0
          fi
          log "Register sphere: $sphere -> atlas -> $sphere_reg"
          mris_register -1 "$sphere" "$atlas_sphere" "$sphere_reg"
        else
          log "Reusing sphere.reg: $sphere_reg"
        fi
      fi

      # Resample to fsaverage6 for both pial and white
      local out_fs_fsa6="$surf_dir/${side}.${surf}.fsa6"
      if [[ ! -s "$out_fs_fsa6" ]]; then
        local srcsubject="${sub_id}/${ses_id}"
        if [[ ! -s "$surf_dir/${side}.sphere.reg" ]]; then
          err "Missing ${side}.sphere.reg required for resampling. Ensure white processed first for $sub_id/$ses_id."
          append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
          return 0
        fi
        log "Resample to fsaverage6: ${side}.${surf} -> ${side}.${surf}.fsa6"
        mri_surf2surf \
          --sd "$DATASET_DIR" \
          --hemi "$side" \
          --srcsubject "$srcsubject" \
          --trgsubject "$trg_atlas" \
          --srcsurfreg sphere.reg \
          --trgsurfreg sphere.reg \
          --sval-xyz "$surf" \
          --tval "$out_fs_fsa6" \
          --tval-xyz "$FREESURFER_HOME/subjects/fsaverage6/mri/orig.mgz"
      else
        log "Reusing resampled: $out_fs_fsa6"
      fi

      # Convert resampled FS surface to VTK with session-style naming
      local out_vtk="$surf_dir/${ses_id}.${hemi}.${surf}.fsa6.vtk"
      if [[ ! -s "$out_vtk$" ]]; then
        log "FS -> VTK: $out_fs_fsa6 -> $out_vtk"
        mris_convert "$out_fs_fsa6" "$out_vtk"
      else
        log "Reusing VTK: $out_vtk"
      fi
      
      # Convert VTK file to PLY file
      local out_ply="$surf_dir/${ses_id}.${hemi}.${surf}.fsa6.ply"
      if [[ ! -s "$out_ply" ]]; then
        log "VTK -> PLY (container): $out_vtk -> $out_ply"
        vtk_to_ply "$out_vtk" "$out_ply"
        if [[ ! -s "$out_ply" ]]; then
          err "Expected GIfTI not found after conversion: $out_ply"
          append_unique "$sub_id/$ses_id" "$CACHE_FAIL"
          return 0
        fi
      else
        log "Reusing PLY: $out_ply"
      fi

      # Cleanup intermediate GIfTI
      if [[ -s "$out_gii" ]]; then
        rm -f "$out_gii"
      fi
    done
  done

  # ---------------------------
  # Evaluation (mesh pair scoring: native GT vs. fsa6 prediction)
  # ---------------------------
  if [[ -f "$EVAL_CONTAINER_IMG" && -f "$REPO/evaluate_mesh_pair.py" ]]; then
    local TMP_EVAL_DIR
    TMP_EVAL_DIR="$(mktemp -d -t eval_${sub_id}_${ses_id}_XXXXXX)"
    # shellcheck disable=SC2064
    trap "rm -rf '${TMP_EVAL_DIR}'" RETURN

    local OUT_CSV_ABS OUT_DIR LOCK_FILE
    OUT_CSV_ABS="$(readlink -f "${OUT_CSV}")"
    OUT_DIR="$(dirname "${OUT_CSV_ABS}")"
    mkdir -p "${OUT_DIR}"
    LOCK_FILE="${OUT_CSV_ABS}.lock"

    local -a EVAL_NV_FLAG=()
    if [[ "${EVAL_DEVICE}" == "cuda" ]]; then
      EVAL_NV_FLAG=(--nv)
    fi

    local EVAL_EXTRA_FLAG=""
    if [[ "${SELF_INTERSECTIONS}" == "true" ]]; then
      EVAL_EXTRA_FLAG="--self_intersections"
    fi

    local -a eval_binds=(
      --bind "${DATASET_DIR}:${DATASET_DIR}"
      --bind "${REPO}:${REPO}"
      --bind "${OUT_DIR}:${OUT_DIR}"
    )
    export SINGULARITYENV_PYTHONPATH="${REPO}:${PYTHONPATH:-}"

    local eval_hemi eval_side eval_surf eval_gt eval_pred eval_label eval_tmp_csv eval_inner_cmd
    for eval_hemi in L R; do
      if [[ "$eval_hemi" == "L" ]]; then eval_side="lh"; else eval_side="rh"; fi
      for eval_surf in pial white; do
        eval_gt="$surf_dir/${ses_id}.${eval_hemi}.${eval_surf}.native.surf.vtk"
        eval_pred="$surf_dir/${ses_id}.${eval_hemi}.${eval_surf}.fsa6.ply"
        eval_label="${sub_id}_${ses_id}_${eval_hemi}_${eval_surf}"

        if [[ ! -f "${eval_gt}" ]]; then
          log "Eval: Missing GT: ${eval_gt} (skip)"
          continue
        fi
        if [[ ! -f "${eval_pred}" ]]; then
          log "Eval: Missing Pred: ${eval_pred} (skip)"
          continue
        fi

        log "Evaluating ${eval_label}"
        eval_tmp_csv="${TMP_EVAL_DIR}/${eval_label}.csv"

        eval_inner_cmd="source '${CONDA_SH}' && conda activate '${CONDA_ENV}' && \
PYTHONPATH='${REPO}:${PYTHONPATH:-}' python '${REPO}/evaluate_mesh_pair.py' \
  --pred '${eval_pred}' \
  --gt '${eval_gt}' \
  --labels '${eval_label}' \
  --n_points '${N_POINTS}' \
  --device '${EVAL_DEVICE}' \
  ${EVAL_EXTRA_FLAG} \
  --out_csv '${eval_tmp_csv}'"

        "$RUN_CTN" exec "${EVAL_NV_FLAG[@]-}" "${eval_binds[@]-}" "$EVAL_CONTAINER_IMG" \
          /bin/bash -lc "${eval_inner_cmd}"

        if [[ -s "${eval_tmp_csv}" ]]; then
          (
            flock -x 200
            if [[ ! -s "${OUT_CSV_ABS}" ]]; then
              echo "Subject,Session,Hemisphere,Surface,PredPath,GTPath,$(head -n1 "${eval_tmp_csv}")" > "${OUT_CSV_ABS}"
            fi
            tail -n +2 "${eval_tmp_csv}" | awk \
              -v s="${sub_id}" -v e="${ses_id}" \
              -v h="${eval_hemi}" -v m="${eval_surf}" \
              -v pp="${eval_pred}" -v gp="${eval_gt}" \
              'BEGIN{FS=OFS=","} {print s,e,h,m,pp,gp,$0}' >> "${OUT_CSV_ABS}"
          ) 200>"${LOCK_FILE}"
        else
          log "WARN: No output CSV produced for ${eval_label}"
        fi
      done
    done
  else
    log "WARN: Skipping evaluation — EVAL_CONTAINER_IMG or evaluate_mesh_pair.py not found."
  fi

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
CACHE_OK="$DATASET_DIR/processed.txt"
CACHE_FAIL="$DATASET_DIR/processed_failed.txt"
touch "$CACHE_OK" "$CACHE_FAIL"

log "DATASET_DIR=$DATASET_DIR"
log "SUBJECTS_DIR=$SUBJECTS_DIR"
log "ATLAS_DIR=$ATLAS_DIR"
log "ATLAS_AGE=$ATLAS_AGE"
log "TRG_ATLAS_SUBJ=$TRG_ATLAS_SUBJ"
log "FREESURFER_HOME=$FREESURFER_HOME"
log "CONTAINER_IMG=$CONTAINER_IMG"
log "BIND_PATHS=$BIND_PATHS"
log "Runtime=$RUN_CTN  N_JOBS=$N_JOBS"
log "Starting session discovery..."

# Discover sessions
mapfile -t SESSIONS < <(find "$DATASET_DIR" -maxdepth 2 -mindepth 2 -type d -name "ses-*" | sort)

log "Found ${#SESSIONS[@]} sessions"

# ---------------------------
# Main loop (optional simple parallelism)
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
