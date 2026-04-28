# Stage 1 — MD Simulation

Run molecular dynamics for water (TIP4P/2005, TIP5P, SWM4-NDP) using OpenMM.
Produces DCD trajectories, PDB initial states, and OpenMM checkpoints under
`../data/simulations/<model>/`.

## Files

| File | Purpose |
|------|---------|
| `MDWater.py`         | Core MD engine: OpenMM setup, equilibration, production. |
| `molecules.py`       | Geometry definitions (TIP4P/2005, TIP5P, SWM4-NDP). |
| `MolPositions.py`    | Lattice initial positions. |
| `CreateTopo.py`      | OpenMM topology builder. |
| `runWater_tip4p2005.py`         | Drive TIP4P/2005 across temperatures. |
| `runWater_tip5p.py`             | Drive TIP5P across temperatures. |
| `runWater_swm4ndp_multitemp.py` | Drude-polarizable SWM4-NDP, −30 → +30 °C. |
| `runWater_tip4p2005_tanaka.py`  | TIP4P/2005 with Tanaka-paper conditions. |
| `runWater_tip5p_tanaka.py`      | TIP5P with Tanaka-paper conditions. |

## Usage

```bash
python runWater_tip4p2005.py    # parallel sweep over temperatures
python runWater_tip5p.py
```

SWM4-NDP needs `swm4ndp.xml` available in the working directory and typically
the OpenCL/CUDA OpenMM platform:

```bash
mkdir -p ../data/simulations/swm4ndp
cd       ../data/simulations/swm4ndp
python   ../../../1_simulate/runWater_swm4ndp_multitemp.py
```

Outputs land alongside the script's CWD; the orchestrator
(`../run_pipeline.sh`) cd's into the right run directory automatically.

→ Next: [Stage 2 — Order Parameters](../2_order_params/README.md)
