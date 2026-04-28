#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIP4P/2005 Water Simulations - Tanaka Conditions
Based on: Shi & Tanaka, J. Am. Chem. Soc. 2020, 142, 2868-2875
Focus: Structure factor analysis at 1 bar, temperature series

Key parameters from Tanaka paper:
- Model: TIP4P/2005
- Pressure: 1 bar (ambient pressure)
- Schottky temperature Ts=1/2: 237.80 K (-35.35°C)
- System size: 1024 water molecules
- Temperature range: Focus on 240K region where s ≈ 0.5
"""

from MDWater import *
import sys

### The function to be run ###
def RunMD(inputs):
        
    MDWater(RunName=inputs[0], 
            Nwater=1024,                      # Tanaka's system size
            T=inputs[1], 
            water_forcefield='tip4p2005',     # TIP4P/2005 model
            t_equilibrate=5*nanoseconds,      # Equilibration time (reduced for efficiency)
            t_simulate=10*nanoseconds,        # Production run (reduced from 50ns to 10ns)
            t_reportinterval=10*picoseconds,  # Save trajectory every 10 ps
            t_step=0.001*picoseconds,         # 1 fs timestep (standard for water)
            CheckPointFileAvail=False, 
            InitPositionPDB=None, 
            ReportVelocity=False, 
            ForceFieldChoice=None,
            PlatformName='OpenCL'             # Use GPU acceleration (OpenCL or CUDA)
           )          

### The list of input parameters ###
# Tanaka's key temperatures for TIP4P/2005 (Table 1, Fig 3):
# Ts=1/2 = 237.80 K = -35.35°C (Schottky temperature, equal LFTS/DNLS mix)
# Focus on temperatures around ambient pressure structure factor measurements
# Temperature range from Tanaka Fig 3: ~240K to 300K at 1 bar

inputs_list = [
    # Temperature series at 1 bar (ambient pressure) - Tanaka conditions
    # These temperatures correspond to Figure 3 of Tanaka paper
    
    # Near Schottky temperature (s ≈ 0.5, maximum two-state fluctuations)
    ('tip4p2005_240K_1bar_Run01', -33.15),   # 240 K - key temperature in Tanaka Fig 3
    
    # Supercooled regime (LFTS-enriched)
    ('tip4p2005_250K_1bar_Run01', -23.15),   # 250 K
    ('tip4p2005_260K_1bar_Run01', -13.15),   # 260 K
    
    # Ambient temperature regime (DNLS-enriched)
    ('tip4p2005_270K_1bar_Run01', -3.15),    # 270 K
    ('tip4p2005_280K_1bar_Run01', 6.85),     # 280 K
    ('tip4p2005_290K_1bar_Run01', 16.85),    # 290 K
    ('tip4p2005_300K_1bar_Run01', 26.85),    # 300 K
]


### Run simulations ###
if __name__ == '__main__':
    
    # Option 1: Run single simulation (uncomment to test)
    RunMD(inputs_list[0])
    
    # Option 2: Run simulations sequentially (safer, recommended for debugging)
    # for inputs in inputs_list:
    #     print(f'\nStarting simulation: {inputs[0]} at T = {inputs[1]}°C')
    #     RunMD(inputs)
    
    # Option 3: Run parallel simulations (use with caution)
    # Limit parallel jobs to avoid system overload
    # from multiprocessing import cpu_count
    # num_cores = min(cpu_count(), 4)  # Limit to 4 cores maximum
    # print(f'Number of cores used = {num_cores}')
    # 
    # from joblib import Parallel, delayed
    # Parallel(n_jobs=num_cores)(delayed(RunMD)(i) for i in inputs_list)


"""
NOTES:
1. Simulation time reduced to 10ns for practical runtime (~2-4 hours per temp)
2. If you need better statistics, increase t_simulate to 20-50ns after testing
3. For structure factor analysis, 10ns should give reasonable S(k) statistics
4. To analyze: Use trajectory files (dcd_*.dcd) to compute:
   - O-O radial distribution function g(r)
   - Structure factor S(k) via Fourier transform
   - Coordination number distribution P(Nfs)
5. Key output files per simulation:
   - system_*.xml: System configuration
   - statedata_*.txt: Thermodynamic data (E, T, etc.)
   - dcd_*.dcd: Trajectory for analysis
   - inistate_*.pdb: Initial configuration
   - cp_*.chk: Checkpoint file
"""
