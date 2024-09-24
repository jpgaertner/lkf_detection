#!/bin/bash

years=(2097)
step=10
res="1km"

# needed for selecting the number of tasks and nodes in the job script
round_up() {
    echo "($1 + 0.999) / 1" | bc
}

for year in "${years[@]}"; do

    # create the python file that is executed by the job script
    for (( day = 0; day <= 365; day += $step )); do
    
        # select dataset for the individual job 
        sed -i "s|\(res = \).*|\1'$res'|" track_lkfs_chunks.py
        sed -i "s|\(year = \).*|\1$year|" track_lkfs_chunks.py
        sed -i "s|\(day = \).*|\1$day|" track_lkfs_chunks.py
        sed -i "s|\(step = \).*|\1$step|" track_lkfs_chunks.py

        # save as new file
        cp track_lkfs_chunks.py /work/bk1377/a270230/python_scripts/track_lkfs_chunks_${year}_${day}.py
    done
    
    # adjust job script and submit it
    sed -i "s|\(#SBATCH --job-name=t\).*|\1$year|" tracking_jobs.sh
    sed -i "s|\(year=\).*|\1$year|" tracking_jobs.sh
    sed -i "s|\(step=\).*|\1$step|" tracking_jobs.sh
    ntasks=$(round_up 365/$step)
    sed -i "s|\(#SBATCH --ntasks=\).*|\1$ntasks|" tracking_jobs.sh
    sbatch tracking_jobs.sh
done
