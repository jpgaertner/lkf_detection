#!/bin/bash
#SBATCH --job-name=2094_5_6
#SBATCH --account=hhb19
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 24:00:00
#SBATCH -o output.txt
#SBATCH -e output.txt


export OMP_NUM_THREADS=1

python3 track_lkfs.py