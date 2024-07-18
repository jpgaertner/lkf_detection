#!/bin/bash

years=({2093..2100})
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
        sed -i "8s/.*/res = '${res}'/" detect_lkfs_chunks.py
        sed -i "13s/.*/year = $year/" detect_lkfs_chunks.py
        sed -i "14s/.*/lower = $day/" detect_lkfs_chunks.py
        sed -i "15s/.*/step = $step/" detect_lkfs_chunks.py

        # save as new file
        cp detect_lkfs_chunks.py /work/bk1377/a270230/python_scripts/detect_lkfs_chunks_${year}_${day}.py
    done
    
    # adjust job script and submit it
    sed -i "2s/.*/#SBATCH --job-name=d${year}/" detection_jobs.sh
    sed -i "14s/.*/year=${year}/" detection_jobs.sh
    sed -i "15s/.*/step=${step}/" detection_jobs.sh
    ntasks=$(round_up 365/$step)
    nodes=$(round_up $ntasks*0.1)
    sed -i "5s/.*/#SBATCH --nodes=$nodes/" detection_jobs.sh
    sed -i "6s/.*/#SBATCH --ntasks=$ntasks/" detection_jobs.sh
    sbatch detection_jobs.sh
done
