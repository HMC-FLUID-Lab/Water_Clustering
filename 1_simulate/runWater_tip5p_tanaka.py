#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIP5P Water Simulations - Tanaka Conditions


Key parameters from Tanaka paper:
- Model: TIP5P (note: TIP5P shows "overstructured" tendency vs real water)
- Pressure: 1 bar (ambient pressure)
- Schottky temperature Ts=1/2: 255.51 K (-17.64°C)
- System size: 1024 water molecules
- Temperature range: Focus on region around Ts=1/2
"""

from MDWater import *
import sys

### The function to be run ###
def RunMD(inputs):
        
    MDWater(RunName=inputs[0], 
            Nwater=1024,                      # Tanaka's system size
            T=inputs[1], 
            water_forcefield='tip5p',         # TIP5P model
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



inputs_list = [
    # Temperature series at 1 bar (ambient pressure) - Tanaka conditions
    # TIP5P requires higher temperatures than TIP4P/2005 due to overstructuring
    
    # Near Schottky temperature (s ≈ 0.5, maximum two-state fluctuations)
    ('tip5p_256K_1bar_Run01', -17.15),       # 256 K - near Schottky temp
    
    # Supercooled to ambient regime
    ('tip5p_260K_1bar_Run01', -13.15),       # 260 K
    ('tip5p_270K_1bar_Run01', -3.15),        # 270 K
    ('tip5p_280K_1bar_Run01', 6.85),         # 280 K
    ('tip5p_290K_1bar_Run01', 16.85),        # 290 K
    ('tip5p_300K_1bar_Run01', 26.85),        # 300 K
]


### Run simulations ###
if __name__ == '__main__':
    
    from multiprocessing import cpu_count
    num_cores = min(cpu_count(), 4)  # Limit to 4 cores maximum
    print(f'Number of cores used = {num_cores}')
    
    from joblib import Parallel, delayed
    # Parallel(n_jobs=num_cores)(delayed(RunMD)(i) for i in inputs_list)
    RunMD(inputs_list[0])  # Test with single simulation first
