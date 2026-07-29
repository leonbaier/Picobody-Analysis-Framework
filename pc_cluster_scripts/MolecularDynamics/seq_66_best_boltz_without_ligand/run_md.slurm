#!/bin/bash
#SBATCH --job-name=picobodies_openmm_LeonB
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --time=96:00:00

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

#SBATCH --nodelist=TUZELSB-MMLAB-ALANINE

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate openmm

python run_md.py