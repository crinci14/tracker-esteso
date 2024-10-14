#! /bin/bash

set -eu

declare -A PCS=( [1]="Periodical" [2]="Disposable" [3]="Distance" [4]="Random" [5]="Car2Car")

for fq in {10..10}
#for fq in {1..10..9}
do
    for pc in {1..5}
    do
        echo -e "Executing the PTF_ext with frequency $fq and policy $pc -> \"${PCS[$pc]}\"" 
        for count in {0..1}
        do
            for pears in {0..1}
            do
                for rb in {0..1}
                do
                    for tm in {0..1}
                    do
                        sum=$((count + pears + rb + tm))
                        if [ $sum -gt 2 ] || [ $sum -eq 0 ]; then # 0 or more then 2
                            echo "Sum($sum) is wrong."
                            continue 
                        fi
                        python3 tracker_ext.py -dir Masa_scenario/sim_5/3Rsu_50range -fq $fq -pc $pc -dim -count $count -pears $pears -rb $rb -tm $tm 
                    done
                done
            done
        done
    done
done
read