#!/bin/bash

years=({2013..2020})
step=10
res="1km"

# needed for selecting the number of tasks in the job script
round_up() {
    echo "($1 + 0.999) / 1" | bc
}

for year in "${years[@]}"; do


    ### find the days that still need to be detected ###

    path_detected_lkfs="/work/bk1377/a270230/lkfs/${res}/${year}_${res}"
    
    # if the lkf folder does not exist, no lkfs have been detected
    if [ ! -d "$path_detected_lkfs" ]; then
        days_to_detect=($(seq 1 365))
        
    # if the lkf folder does exist, check for the days that are missing
    else
        # load the file names of the detected lkfs
        lkf_files=($(ls "$path_detected_lkfs" | sort))
        lkf_files=("${lkf_files[@]/.ipynb_checkpoints}")

        # read which days have been detected
        detected_days=()
        for lkf_file in "${lkf_files[@]}"; do
            detected_day=$(echo "$lkf_file" | cut -d'.' -f1 | cut -d'_' -f4)
            detected_days+=("$((10#$detected_day))")
        done

        # create list of the days that still need to be detected
        days_all=$(seq 1 365)
        days_to_detect=()
        for day in $days_all; do
            if [[ ! " ${detected_days[@]} " =~ " $day " ]]; then
                days_to_detect+=("$day")
            fi
        done
    fi

    
    ### update the python file that detects the individual chunks of days ###

    # convert the bash array to a python list format and update it in the python file
    days_to_detect_python="[$(IFS=,; echo "${days_to_detect[*]}")]"
    sed -i "s|^\(days_to_detect = np.array(\).*|\1$days_to_detect_python)|" detect_lkfs_chunks.py

    # copy other variables into the python file
    sed -i "s|^\(res = \).*|\1'$res'|" detect_lkfs_chunks.py
    sed -i "s|^\(year = \).*|\1$year|" detect_lkfs_chunks.py
    sed -i "s|^\(step = \).*|\1$step|" detect_lkfs_chunks.py
    
    startdays_of_chunks=()
    for (( i = 0; i < ${#days_to_detect[@]}; i += step )); do
        startdays_of_chunks+=("${days_to_detect[i]}")
    done
    
    for startday in "${startdays_of_chunks[@]}"; do
        sed -i "s|^\(startday = \).*|\1$startday|" detect_lkfs_chunks.py
        # save as new file
        cp detect_lkfs_chunks.py /work/bk1377/a270230/python_scripts/detect_lkfs_chunks_${year}_${startday}.py
    done


    ### adjust job script and submit it ###
    
    sed -i "s|^\(#SBATCH --job-name=\).*|\1d$year|" detection_jobs.sh
    #sed -i "s|^\(#SBATCH --output=output_\).*\$|\1$year.txt|" detection_jobs.sh
    ntasks=${#startdays_of_chunks[@]}
    sed -i "s|^\(#SBATCH --ntasks=\).*|\1$ntasks|" detection_jobs.sh
    startdays_of_chunks_str=$(printf '%s ' "${startdays_of_chunks[@]}")
    sed -i "s|^\(startdays_of_chunks=\).*|\1(${startdays_of_chunks_str})|" detection_jobs.sh
    sed -i "s|^\(year=\).*|\1$year|" detection_jobs.sh

    sbatch detection_jobs.sh
done
