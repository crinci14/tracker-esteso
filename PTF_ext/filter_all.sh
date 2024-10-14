#! /bin/bash

set -eu

declare -A PCS=( [1]="Periodical" [2]="Disposable" [3]="Distance" [4]="Random" [5]="Car2Car")

#for fq in {1..10..9}
for fq in {10..10}
do
    for pc in {1..5}
    do
        echo -e "Executing filter with frequency $fq and policy $pc -> \"${PCS[$pc]}\"" 
        python3 filter_range.py -dir Masa_scenario/sim_5/3Rsu -fq $fq -pc $pc -rg 50
    done
done
