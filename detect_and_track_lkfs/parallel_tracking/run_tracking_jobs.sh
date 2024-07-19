#!/bin/bash

years=({2013..2020})
step=10
res="4km"

# needed for selecting the number of tasks and nodes in the job script
round_up() {
    echo "($1 + 0.999) / 1" | bc
}

for year in "${years[@]}"; do

    # create the python file that is executed by the job script
    for (( day = 0; day <= 365; day += $step )); do
    
        # select dataset for the individual job 
        sed -i "10s/.*/res = '${res}'/" track_lkfs_chunks.py
        sed -i "11s/.*/year = $year/" track_lkfs_chunks.py
        sed -i "12s/.*/lower = $day/" track_lkfs_chunks.py
        sed -i "13s/.*/step = $step/" track_lkfs_chunks.py

        # save as new file
        cp track_lkfs_chunks.py /work/bk1377/a270230/python_scripts/track_lkfs_chunks_${year}_${day}.py
    done
    
    # adjust job script and submit it
    sed -i "2s/.*/#SBATCH --job-name=t${year}/" tracking_jobs.sh
    sed -i "14s/.*/year=${year}/" tracking_jobs.sh
    sed -i "15s/.*/step=${step}/" tracking_jobs.sh
    ntasks=$(round_up 365/$step)
    sed -i "6s/.*/#SBATCH --ntasks=$ntasks/" tracking_jobs.sh
    sbatch tracking_jobs.sh
done
