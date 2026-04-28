#!/usr/bin/env python3
"""
SWM4-NDP MD over −30 … +30 °C (10 °C steps), N = 1024, **5 saved frames** per run.

Prerequisites
-------------
1. Run from the directory where you want `dcd_*` / `inistate_*` / `system_*.xml`
   (recommended):

       mkdir -p /path/to/swm4ndp_runs && cd /path/to/swm4ndp_runs
       python /path/to/runWater_swm4ndp_multitemp.py

2. **`swm4ndp.xml`** (OpenMM Drude force field) must be on the OpenMM search path.
   MDWater loads ``ForceField('swm4ndp.xml')`` from the **current working directory**
   unless you extend the path in ``MDWater.py`` (see existing ``sys.path.append``).

3. SWM4-NDP uses a **Drude** integrator; pick a platform your OpenMM build supports
   (often **CUDA** or **OpenCL**). Edit ``PlatformName`` below if the run fails.

Output naming (matches ``run_batch_params.find_matching_pdb``)::

    dcd_swm4ndp_T-30_N1024_Run01_0.dcd
    inistate_swm4ndp_T-30_N1024_Run01.pdb
"""
from MDWater import *
from multiprocessing import cpu_count
from joblib import Parallel, delayed


def RunMD(inputs):
    run_name, T_celsius = inputs
    MDWater(
        RunName=run_name,
        Nwater=1024,
        T=T_celsius,
        water_forcefield="swm4ndp",
        # Short equilibration — increase for production science
        t_equilibrate=0.2 * nanoseconds,
        # Exactly 5 frames: 5 × t_reportinterval
        t_simulate=50 * picoseconds,
        t_reportinterval=10 * picoseconds,
        t_step=0.001 * picoseconds,
        CheckPointFileAvail=False,
        InitPositionPDB=None,
        ReportVelocity=False,
        ForceFieldChoice=None,
        PlatformName="OpenCL",  # try "CUDA" or "Reference" if this fails
    )


# −30 °C … +30 °C in 10 °C steps (same grid as TIP4P/2005 multitemp)
inputs_list = [
    (f"swm4ndp_T{t}_N1024_Run01", float(t))
    for t in range(-30, 31, 10)
]


if __name__ == "__main__":
    import sys

    # Single run: python runWater_swm4ndp_multitemp.py one swm4ndp_T-20_N1024_Run01 -20
    if len(sys.argv) >= 3 and sys.argv[1] == "one":
        RunMD((sys.argv[2], float(sys.argv[3])))
        sys.exit(0)

    num_cores = cpu_count()
    print(f"SWM4-NDP multitemp | cores = {num_cores} | runs = {len(inputs_list)}")
    Parallel(n_jobs=num_cores)(delayed(RunMD)(i) for i in inputs_list)
