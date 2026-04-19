#!/bin/bash
run_simulation() {
    # Please modify the path to the jagurs executable file to match your environment.
    ./jagurs par=tsun.par > log.txt
    mkdir -p tgsfiles
    mv tgs0* tgsfiles/
}

n_sample=2200

for i in $(seq 1 $n_sample); do
    sim_dir="sample${i}"
    cd "$sim_dir" || exit 1  # error -> exit
    run_simulation
    echo $i
    cd ..
done
