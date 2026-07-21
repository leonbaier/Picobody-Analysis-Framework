#!/bin/bash

for d in seq_*; do
    [ -d "$d" ] || continue

    if [ -f "$d/run_md.slurm" ]; then
        echo "Submitting $d"

        (
            cd "$d"
            sbatch run_md.slurm
        )
    fi
done