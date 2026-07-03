#!/bin/bash

#SBATCH --time=0:30:0
#SBATCH --account=tc075
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-255

#SBATCH --partition=standard
#SBATCH --qos=standard

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Set the number of threads to 1
export OMP_NUM_THREADS=1

module load lammps

srun --hint=nomultithread --distribution=block:block lmp -i liquid_vapour.in -var localstress 1 -var outdir output_$SLURM_ARRAY_TASK_ID -var seed $SLURM_ARRAY_TASK_ID

