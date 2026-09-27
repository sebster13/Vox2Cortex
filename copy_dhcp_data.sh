#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================
# Change these paths to your actual source and destination directories
SRC_DIR="/data/rawdata_derivatives/2026_FETAL_MR"
DST_DIR="/project/fetal/2025_FetalSurface_swolff/2026_FETAL_MR"

# Log files (will be created in the directory where you run the script)
SUCCESS_LOG="copy_success.log"
ERROR_LOG="copy_errors.log"

# List of relative paths to copy for each session
REQUIRED_ITEMS=(
    "age.tsv"
    "dhcp/masks"
    "dhcp/segmentations"
    "dhcp/segmentations-data/cortical"
    "dhcp/segmentations-data/gm-posteriors"
    "dhcp/segmentations-data/labels"
    "dhcp/segmentations-data/posteriors"
    "dhcp/segmentations-data/template"
    "dhcp/segmentations-data/tissue-initial-segmentations"
    "dhcp/segmentations-data/tissue-posteriors"
    "dhcp/surfaces"
    "dhcp/T2"
    "seg_svrtk"
)

# ==========================================
# INITIALIZATION
# ==========================================
# Clear previous logs
> "$SUCCESS_LOG"
> "$ERROR_LOG"

echo "Starting copy process..."
echo "Source: $SRC_DIR"
echo "Destination: $DST_DIR"
echo "------------------------------------------"

# Ensure destination exists
mkdir -p "$DST_DIR"

# Count variables
success_count=0
error_count=0

# ==========================================
# MAIN LOOP
# ==========================================
# Loop through sub-XX folders (matching integers only)
for sub_dir in "$SRC_DIR"/sub-[0-9]*; do
    # Ensure it's a directory
    [ -d "$sub_dir" ] || continue
    sub_name=$(basename "$sub_dir")

    # Loop through ses-YYYYMMDD folders (matching 8-digit integers)
    for ses_dir in "$sub_dir"/ses-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]; do
        # Ensure it's a directory
        [ -d "$ses_dir" ] || continue
        ses_name=$(basename "$ses_dir")

        echo "Checking $sub_name/$ses_name..."

        # Phase 1: Check if all required items exist
        missing_items=()
        for item in "${REQUIRED_ITEMS[@]}"; do
            if [ ! -e "$ses_dir/$item" ]; then
                missing_items+=("$item")
            fi
        done

        # Phase 2: Copy or Log Error
        if [ ${#missing_items[@]} -eq 0 ]; then
            # All items exist. Proceed to copy.
            dest_ses_dir="$DST_DIR/$sub_name/$ses_name"
            
            for item in "${REQUIRED_ITEMS[@]}"; do
                # Get the parent directory of the item to recreate structure
                item_dir=$(dirname "$item")
                target_dir="$dest_ses_dir/$item_dir"
                
                # Recreate subfolders in destination
                mkdir -p "$target_dir"
                
                # Copy file or folder
                cp -r "$ses_dir/$item" "$target_dir/"
            done
            
            echo "SUCCESS: $sub_name/$ses_name copied."
            echo "$sub_name/$ses_name" >> "$SUCCESS_LOG"
            ((success_count++))
        else
            # Missing items found. Log error and skip.
            echo "ERROR: $sub_name/$ses_name is missing required files/folders."
            echo "--- $sub_name/$ses_name ---" >> "$ERROR_LOG"
            for missing in "${missing_items[@]}"; do
                echo "  Missing: $missing" >> "$ERROR_LOG"
            done
            echo "" >> "$ERROR_LOG" # add blank line
            ((error_count++))
        fi
    done
done

# ==========================================
# SUMMARY
# ==========================================
echo "------------------------------------------"
echo "Process Finished."
echo "Successfully copied: $success_count sessions (Logged in $SUCCESS_LOG)"
echo "Failed/Incomplete: $error_count sessions (Logged in $ERROR_LOG)"