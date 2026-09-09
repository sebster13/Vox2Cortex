#!/bin/bash

#SBATCH -J surface_reconstruction
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --qos=a100-sxm4-80gb
#SBATCH --mail-type="END,FAIL"
#SBATCH --mail-user=a12007396@unet.univie.ac.at
#SBATCH --output="slurm-%j.out"

#=========================================================================================
# YOUR COMMANDS TO EXECUTE
#=========================================================================================

echo "Job started on $(date)"
echo "Job running on node $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "----------------------------------------------------"

# Note: The home directory ($HOME) inside the container is usually the same as your
# home directory on the host machine, so ~/.wandb_secret will resolve correctly.
srun --container-writable \
     --container-image=sebastianwolff13/vox2cortex:ubuntu20.04 \
     --container-mounts=$PWD:/host_workspace \
     bash -c "
         # Exit immediately if a command exits with a non-zero status.
         set -e
         
         # --- Your commands to be executed INSIDE the container go here ---

         # 1. Load secrets from your private file
         echo 'Loading secrets from ~/.wandb_secret'
         source ~/.wandb
         
         # 2. Manually initialize Conda for this shell session
         source /opt/conda/etc/profile.d/conda.sh
         
         # 3. Activate your environment
         conda activate torch_1_10
         
         echo '--- Environment and Secret Check ---'
         if [ -n \"\$WANDB_API_KEY\" ]; then
           echo 'WANDB_API_KEY has been set successfully.'
         else
           echo 'WARNING: WANDB_API_KEY is not set.' >&2
         fi
         python --version
         echo '------------------------------------'

         # 4. Navigate to your workspace and run your script
         cd /host_workspace/Vox2Cortex/vox2organ
         
         echo 'Now running the Python script...'
    
         
         # ====> REPLACE THIS LINE WITH YOUR ACTUAL PYTHON COMMAND <====
         python3 main.py --train --test --group "V2CC" --dataset FETAL_CONTROL_JT --n_epochs 400
         
         echo '--- End of container commands ---'
     "

echo "----------------------------------------------------"
echo "Job finished on $(date)"