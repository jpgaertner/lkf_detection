#!/bin/bash -l
#SBATCH --job-name=d2016
#SBATCH --account=bk1377
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH -t 8:00:00
#SBATCH -o output.txt
#SBATCH -e output.txt


export OMP_NUM_THREADS=1

year=2016
startdays_of_chunks=(170 214 244 286 360 )

for day in "${startdays_of_chunks[@]}"; do
    python3 /work/bk1377/a270230/python_scripts/detect_lkfs_chunks_${year}_${day}.py &
done

wait