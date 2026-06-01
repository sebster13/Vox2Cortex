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
ATLAS_AGE="${ATLAS_AGE:-25}" # To be adapted later on accordingly
ATLAS_DIR="${ATLAS_DIR:-/project/fetal/2025_FetalSurface_swolff/2024_FETAL_CONTROL_JT/atlases}"
TRG_ATLAS_SUBJ="${TRG_ATLAS_SUBJ:-/CRL_ATLAS2017_GW${ATLAS_AGE}}"  # must exist under SUBJECTS_DIR

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

