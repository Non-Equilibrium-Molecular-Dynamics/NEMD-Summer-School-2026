#!/bin/bash

#SBATCH --partition=standard
#SBATCH --job-name=Example_lammps_job
#SBATCH --time=0:30:0
#SBATCH --account=tc075
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --qos=standard
#SBATCH --export=none

# Set the number of threads to 1
export OMP_NUM_THREADS=1

module load lammps

# Launch the parallel job
srun --hint=nomultithread --distribution=block:block lmp -i liquid_vapour.in -var localstress 1 -var dump 1


