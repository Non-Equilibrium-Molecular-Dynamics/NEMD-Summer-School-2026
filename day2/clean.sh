#!/bin/bash
# clean.sh -- remove all regeneratable Day-1 output files so you can start a
# fresh run.
#
# Removes, from this top-level folder and the four measurement folders:
#   day1_*            simulation outputs + plots (profiles, .dat, params,
#                     trajectories, day1_*.png, day1_overview.png, day1_domain.png)
#   log.lammps        LAMMPS run logs
#   slurm-*.out       Cirrus/SLURM job logs
#   __pycache__/      python bytecode caches
#
# It NEVER touches source (*.in / *.py / *.lmp / *.sh) or the handout PDF.
#
#   ./clean.sh        remove the regeneratable files
#   ./clean.sh -n     dry run: just list what WOULD be removed (deletes nothing)
#
set -u
cd "$(dirname "$0")" || exit 1

dry=0
[ "${1:-}" = "-n" ] && dry=1

for d in . local_viscosity liquid_vapour; do
  [ -d "$d" ] || continue
  if [ "$dry" = 1 ]; then
    find "$d" -maxdepth 1 -type f \( -name 'day1_*' -o -name 'log.lammps' -o -name 'slurm-*.out' -o -name '.DS_Store' \) -print
    find "$d" -maxdepth 1 -type d -name '__pycache__' -print
  else
    find "$d" -maxdepth 1 -type f \( -name 'day1_*' -o -name 'log.lammps' -o -name 'slurm-*.out' -o -name '.DS_Store' \) -delete
    find "$d" -maxdepth 1 -type d -name '__pycache__' -exec rm -rf {} +
  fi
done

if [ "$dry" = 1 ]; then
  echo "(dry run -- nothing deleted)"
else
  echo "cleaned: regeneratable Day-1 files removed (source and the handout PDF kept)."
fi
