#!/bin/bash -l
#SBATCH --job-name=create
#SBATCH --account=bk1377
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 8:00:00
#SBATCH --output=output.txt


export OMP_NUM_THREADS=1

python3 create_dictionary.py