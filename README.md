# NEMD Summer School 2026 — Hands-on materials

Hands-on materials for the **NEMD for Fluids and Interfaces Summer School**, 6–7 July 2026,
University of Edinburgh. The school runs over two afternoons, each a set of small LAMMPS
simulations with short Python analysis scripts, run on Cirrus (EPCC):

- **Day 1 — Solid–Liquid Interfaces** (`day1/`) — available now
- **Day 2 — Nanoscale Hydrodynamics & Liquid–Vapour Interfaces** (`day2/`) — added closer to the event

## Day 1 — Solid–Liquid Interfaces

Day 1 measures four properties of a Lennard-Jones liquid in a solid-walled channel, each from
one LAMMPS simulation plus a short Python analysis script:

| Measurement | Folder | Runs | Analyses |
|---|---|---|---|
| Interfacial density / layering | `day1/density/` | `density.in` | `analyse_density.py` |
| Slip length | `day1/slip/` | `slip.in` | `analyse_slip.py` |
| Shear viscosity | `day1/viscosity/` | `viscosity.in` | `analyse_viscosity.py` |
| Interfacial thermal conductance | `day1/conductance/` | `conductance.in` | `analyse_conductance.py` |

Every case includes the shared channel setup with `include ../shared_setup.lmp`, and the analysis
scripts share their output-file parsers via `../lammps_io.py`.

## Getting onto Cirrus

The sessions run on **Cirrus** (EPCC). Once your Cirrus account is set up (SSH key registered on
SAFE + MFA token):

1. **Connect** (the `-X` forwards graphics so plots can open):
   ```
   ssh -X USERNAME@login.cirrus.ac.uk
   ```
2. **Work under `/work`** — compute nodes can read `/work` but not `/home`:
   ```
   cd /work/tc075/tc075/USERNAME
   ```
3. **Get the materials:**
   ```
   git clone https://github.com/Non-Equilibrium-Molecular-Dynamics/NEMD-Summer-School-2026.git
   ```
4. **Set up Python for the analysis scripts** (once per login):
   ```
   module load PrgEnv-gnu
   module load cray-python
   python -m pip install matplotlib scikit-learn
   ```

## Running a measurement

Work from inside a case folder (so `include ../shared_setup.lmp` resolves), and submit with the
provided `submit.sh`:

```
cd NEMD-Summer-School-2026/day1/density
../submit.sh density.in            # submits a SLURM job; check it with:  squeue --me
# once the job has finished:
python analyse_density.py           # writes day1_density.png
```

Repeat for `slip/`, `viscosity/`, and `conductance/`. Run the analysis scripts with `python`
(not `python3`). To explore how a measurement responds, pass a variable, e.g.
`../submit.sh slip.in -var eps_wf 0.6`.

Helper scripts: `day1/clean.sh` removes regenerated outputs; `day1/dashboard.py` shows a live 2×2
view across the four measurements; `day1/view_domain.py` writes a single snapshot for VMD/OVITO.

## Requirements

- **LAMMPS** — on Cirrus, `module load lammps` (the `submit.sh` job does this for you).
- **Python 3** with `numpy` and `matplotlib` — on Cirrus, `cray-python` plus the `pip install` above.

---

Part of the NEMD Summer School 2026, University of Edinburgh.
