from MDWater import *
import sys

### The function to be run ###
def RunMD(inputs):
        
    MDWater(RunName=inputs[0], 
            Nwater=1024,                      # System size (matching TIP4P/2005 runs)
            T=inputs[1], 
            water_forcefield='tip5p',         # TIP5P model from Tanaka paper
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
# TIP5P model conditions from Tanaka's paper
# Tanaka's Schottky temperature for TIP5P: Ts=1/2 = 255.51 K = -17.64°C
# Temperature range: -25°C (248 K) to +30°C (303 K)
# Note: TIP5P is "overstructured" compared to TIP4P/2005 and real water

inputs_list = [
    # Deep supercooled regime (LFTS-dominant, below Schottky temp)
    ('tip5p_T-25_N1024_Run01', -25.0),   # 248 K
    ('tip5p_T-25_N1024_Run02', -25.0),
    ('tip5p_T-20_N1024_Run01', -20.0),   # 253 K
    ('tip5p_T-20_N1024_Run02', -20.0),
    ('tip5p_T-18_N1024_Run01', -18.0),   # 255 K (near Schottky temp)
    ('tip5p_T-18_N1024_Run02', -18.0),
    
    # Moderately supercooled regime (two-state coexistence)
    ('tip5p_T-15_N1024_Run01', -15.0),   # 258 K
    ('tip5p_T-15_N1024_Run02', -15.0),
    ('tip5p_T-10_N1024_Run01', -10.0),   # 263 K
    ('tip5p_T-10_N1024_Run02', -10.0),
    
    # Near freezing and ambient conditions (DNLS-dominant)
    ('tip5p_T0_N1024_Run01', 0.0),       # 273 K
    ('tip5p_T0_N1024_Run02', 0.0),
    ('tip5p_T10_N1024_Run01', 10.0),     # 283 K
    ('tip5p_T10_N1024_Run02', 10.0),
    ('tip5p_T20_N1024_Run01', 20.0),     # 293 K
    ('tip5p_T20_N1024_Run02', 20.0),
    ('tip5p_T30_N1024_Run01', 30.0),     # 303 K
    ('tip5p_T30_N1024_Run02', 30.0),
]


### Run one single simulation ###
# RunMD(inputs_list[0])

### Run Parallel simulations ###
# number of CPUs
from multiprocessing import cpu_count
num_cores = cpu_count()
print('Number of cores used = %i' %num_cores)

# Parallel computing with joblib package
from joblib import Parallel, delayed
Parallel(n_jobs=num_cores)(delayed(RunMD)(i) for i in inputs_list)
