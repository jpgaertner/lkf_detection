#!/bin/bash
#SBATCH --account=hhb19
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=lkf_tune
#SBATCH -t 10:00:00
#SBATCH -o output.txt
#SBATCH -e output.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jan.gaertner@awi.de


export OMP_NUM_THREADS=1


python3 tune_lkfs.py --slurm