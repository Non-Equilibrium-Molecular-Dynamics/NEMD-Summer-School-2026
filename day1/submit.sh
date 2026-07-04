#!/bin/bash
# submit.sh -- submit one Day-1 LAMMPS measurement to Cirrus as a batch job.
#
#   Run it FROM INSIDE a measurement folder (density/ slip/ viscosity/
#   conductance/) so the job starts where `include ../shared_setup.lmp` resolves:
#
#     cd density     && ../submit.sh density.in
#     cd slip        && ../submit.sh slip.in -var vwall 2.0
#     cd conductance && ../submit.sh conductance.in
#
# It writes a short SLURM job for you and sends it to the queue with sbatch.
# Check progress with `squeue --me`; LAMMPS runs on a compute node and its output
# lands in slurm-<jobid>.out.  When the job has finished, run the matching
# analyser, e.g.  python analyse_density.py
set -e

if [ "$#" -lt 1 ]; then
  echo "usage: cd <measurement-folder> && ../submit.sh <input>.in [-var NAME value ...]"
  exit 1
fi

IN="$1"   # the .in file (any -var arguments follow it)

# Preflight: refuse to submit from the wrong place, so the job cannot fail late on a
# compute node (and waste class time) on a missing include.
if [ ! -f "$IN" ]; then
  echo "error: input '$IN' not found here -- cd into a measurement folder"
  echo "       (density/ slip/ viscosity/ conductance/) and run ../submit.sh from inside it."
  exit 1
fi
if [ ! -f ../shared_setup.lmp ]; then
  echo "error: ../shared_setup.lmp not found -- run this from INSIDE a measurement folder,"
  echo "       not the top level (each .in does 'include ../shared_setup.lmp')."
  exit 1
fi

# Refuse to submit from /home: Cirrus compute nodes can read /work but not /home, so a job
# launched from a /home folder passes the checks above but fails late on the compute node.
case "$PWD" in
  /work/tc075/tc075/*) ;;
  *)
    echo "error: Cirrus jobs must be submitted from /work/tc075/tc075/USERNAME, not:"
    echo "       $PWD"
    echo "       cd /work/tc075/tc075/$USER and run from the measurement folder there."
    exit 1
    ;;
esac

# Clear THIS folder's previous outputs before submitting, so the analyser can never
# read stale numbers from an earlier run while the new job is still queued or running:
# a missing file gives a clear error, stale data gives a silently wrong one.
rm -f day1_*.profile day1_*.dat day1_*.txt day1_*.png day1_*.xyz log.lammps

printf -v LMP_ARGS ' %q' "$@"   # input + any -var args, shell-quoted so spaces/globs survive

JOBID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=day1
#SBATCH --account=tc075
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --export=none

# Clean job environment (--export=none) so the Python modules you loaded for the
# analysers cannot leak in and clash with 'module load lammps'.
export OMP_NUM_THREADS=1

module load lammps
srun --hint=nomultithread --distribution=block:block lmp -in$LMP_ARGS
EOF
)
echo "Submitted job $JOBID. Wait until 'squeue --me' no longer lists it, THEN run the analyser:"
echo "    python analyse_${IN%.in}.py"
echo "(running it before the job finishes reports 'not found'.)"
