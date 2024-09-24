#!/bin/bash -l
#SBATCH --job-name=d2020
#SBATCH --account=bk1377
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=21
#SBATCH -t 8:00:00
#SBATCH --output=output.txt


export OMP_NUM_THREADS=1

year=2020
startdays_of_chunks=(29 87 146 168 180 190 200 210 220 230 240 250 260 270 280 290 300 310 320 330 345 )

for day in "${startdays_of_chunks[@]}"; do
    python3 /work/bk1377/a270230/python_scripts/detect_lkfs_chunks_${year}_${day}.py &
done

wait