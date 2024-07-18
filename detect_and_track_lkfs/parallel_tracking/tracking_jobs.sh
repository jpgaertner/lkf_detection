#!/bin/bash -l
#SBATCH --job-name=t2020
#SBATCH --account=bk1377
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --ntasks=36
#SBATCH -t 8:00:00
#SBATCH -o output.txt
#SBATCH -e output.txt


export OMP_NUM_THREADS=1

year=2020
step=10

for day in $(seq 0 $step 365); do
    python3 /work/bk1377/a270230/python_scripts/track_lkfs_chunks_${year}_${day}.py &
done

wait