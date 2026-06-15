#!/bin/bash
#SBATCH --job-name=zen_garden
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --output=zen_garden_%A_%a.out
#SBATCH --error=zen_garden_%A_%a.err
#SBATCH --array=0-3

module load stack/2024-06
module load gcc/12.2.0
module load python/3.12.8
module load gurobi/13.0.0

python -m venv .venv
source .venv/bin/activate

python -m main_run_mean_variance \
    --task_id ${SLURM_ARRAY_TASK_ID}