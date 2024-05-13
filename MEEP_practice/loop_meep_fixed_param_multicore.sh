#!/bin/bash

# Loop over np values from 1 to 10
for np in {1..10}; do
    for run_num in {1..5}; do
       echo "Running simulation with np=$np"
       # Run the MEEP simulation with the current np value
       mpirun -np $np python MEEP_sim_fixed_param_parallel.py
   done
done

