#!/bin/bash
#SBATCH --job-name=PNC           # Name of your job
#SBATCH --output=output/slurm_outputs/%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=output/slurm_outputs/%x_%j.err             # Error file
#SBATCH --partition=CPU     
#SBATCH --cpus-per-task=64
#SBATCH --mem=600GB
#SBATCH --time=24:00:00               # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"
# Print CPU details
echo "CPU details: $(lscpu)"
echo "Memory details: $(free -h)"
echo "Disk details: $(df -h)"

# Activate the environment
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet

# Change to the project directory and set PYTHONPATH
cd /home/infres/vmorozov/code
export PYTHONPATH=$PYTHONPATH:/home/infres/vmorozov/code

# Run the Python script
python train_climate_reg.py --config configs/config_climate_reg_wind.yaml

echo "Job finished at: $(date)"