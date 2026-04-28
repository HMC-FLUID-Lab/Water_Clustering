from MDWater import *
import sys

### The function to be run ###
def RunMD(inputs):
        
    MDWater(RunName=inputs[0], 
            Nwater=1024,                      # System size (can increase to 2000+ for better statistics)
            T=inputs[1], 
            water_forcefield='tip4p2005',     # TIP4P/2005 model from Tanaka paper
            t_equilibrate=10*nanoseconds,     # Longer equilibration for supercooled water
            t_simulate=50*nanoseconds,        # Long simulation for good statistics (Tanaka-style)
            t_reportinterval=10*picoseconds,  # Save trajectory every 10 ps
            t_step=0.001*picoseconds,         # 1 fs timestep
            CheckPointFileAvail=False, 
            InitPositionPDB=None, 
            ReportVelocity=False, 
            ForceFieldChoice=None,
            PlatformName='OpenCL'             # options: Reference, CPU, CUDA, OpenCL
           )          
 
### The list of input parameters ###
# Temperature range matching Tanaka's paper: supercooled to ambient conditions
# Tanaka's Schottky temperature for TIP4P/2005: Ts=1/2 = 237.80 K = -35.35°C
# Temperature range: -40°C (233 K) to +30°C (303 K)

inputs_list = [
    # Deep supercooled regime (LFTS-dominant)
    ('tip4p2005_T-40_N1024_Run01', -40.0),   # 233 K
    ('tip4p2005_T-40_N1024_Run02', -40.0),
    ('tip4p2005_T-35_N1024_Run01', -35.0),   # 238 K (near Schottky temp)
    ('tip4p2005_T-35_N1024_Run02', -35.0),
    ('tip4p2005_T-30_N1024_Run01', -30.0),   # 243 K
    ('tip4p2005_T-30_N1024_Run02', -30.0),
    
    # Moderately supercooled regime (two-state coexistence)
    ('tip4p2005_T-20_N1024_Run01', -20.0),   # 253 K
    ('tip4p2005_T-20_N1024_Run02', -20.0),
    ('tip4p2005_T-10_N1024_Run01', -10.0),   # 263 K
    ('tip4p2005_T-10_N1024_Run02', -10.0),
    
    # Near freezing and ambient conditions (DNLS-dominant)
    ('tip4p2005_T0_N1024_Run01', 0.0),       # 273 K
    ('tip4p2005_T0_N1024_Run02', 0.0),
    ('tip4p2005_T10_N1024_Run01', 10.0),     # 283 K
    ('tip4p2005_T10_N1024_Run02', 10.0),
    ('tip4p2005_T20_N1024_Run01', 20.0),     # 293 K
    ('tip4p2005_T20_N1024_Run02', 20.0),
    ('tip4p2005_T30_N1024_Run01', 30.0),     # 303 K
    ('tip4p2005_T30_N1024_Run02', 30.0),
]


### Run one single simulation ###
# RunMD(inputs_list[0])

### Run Paralell simulations ###
# number of CPUs
from multiprocessing import cpu_count
num_cores = cpu_count()
print('Number of cores used = %i' %num_cores)

# Parallel computing with joblib package
from joblib import Parallel, delayed
Parallel(n_jobs=num_cores)(delayed(RunMD)(i) for i in inputs_list)







