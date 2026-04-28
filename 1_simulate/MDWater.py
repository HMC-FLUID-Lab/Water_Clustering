#!/usr/bin/python
"""
This script runs a simple molecular dynamics for water 
created January 21, 2020 
modified from MDDrudewater.py to consider other forcefields for water
@author: bilinzhuang
"""
import sys
sys.path.append('../../General_Parameters_and_Scripts/SimulationPythonScripts/')


from math import pi,sin,cos,asin
from numpy import sqrt, matrix, ceil, interp, sort, deg2rad, ones, mod

from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

from molecules import *
from MolPositions import *
from CreateTopo import *
import datetime

#from ComputeOrderParameters import *

def Water_v(T):
    """
    This function calculates the molecular volume of water as a function of temperature in nm^3
    """
  
    water_rho = {10:0.9997026, 15:0.9991026, 20:0.9982071, 25:0.9970479, 
                 30:0.9956502, 35:0.99403, 40:0.99221, 45:0.99022, 
                 50:0.98804, 55:0.98570, 60:0.98321, 65:0.98056, 
                 70:0.97778, 75:0.97486, 80:0.97180, 85:0.96862, 
                 90:0.96531, 95:0.96189, 100:0.95835} # in g/cm^3
    
    T_all = sort(array([x for x in water_rho.keys()]))
    rho_all = array([water_rho[i] for i in T_all])
    rho = interp(T,T_all,rho_all)
    
    water_v = 29.91507555/rho*10**-3; # in nm^3
    
    return water_v
  

def MDWater(RunName, Nwater,  T, water_forcefield,
                 t_equilibrate, t_simulate, t_reportinterval, t_step,
                 CheckPointFileAvail=False, InitPositionPDB=None, 
                 ReportVelocity = False, ForceFieldChoice=None,
                 PlatformName = 'Reference'):
    
    run_label=RunName
    
    
    # Create Topology
    Water=createWater(water_forcefield)
    
    Water.v = Water_v(T) 
    
    Sol_N = [('Water',Nwater)]
  
    MolDict={'Water':Water}   
    Sol_Nsol = [(MolDict[x[0]],)+ x[1:2] for x in Sol_N]
    
    Topo, NAtoms = CreateTopo(Sol_Nsol)
    
    L,N=BoxSize_NumPar(Sol_Nsol)
    Topo.setUnitCellDimensions(Vec3(L[0],L[1],L[2])*nanometers)
    
    
    """ 
    ### look at topology ###
    MyAtoms = list(Topo.atoms()) # Create a list of atom objects.
    MyBonds = list(Topo.bonds()) # Create a list of bonded atom pairs.
  
    for atom in MyAtoms: # Loop through the atoms.
        print atom.name, atom.element, atom.index, atom.residue.name # Print the name of the atom.
    for bond in MyBonds: # Loop through the bonded atom pairs.
        print bond[0].name, bond[1].name, bond[0].index, bond[1].index # Print the names of atoms in each bond.
    """
  
    #==============================================================================
    # Create System
  
    if ForceFieldChoice==None:
        forcefield = ForceField(water_forcefield+'.xml')
    else:
        forcefield = ForceField(ForceFieldChoice)
    system = forcefield.createSystem(Topo, nonbondedMethod=CutoffPeriodic, nonbondedCutoff=min(L)/2.0, constraints=HAngles)
    system.setDefaultPeriodicBoxVectors(Vec3(L[0],0,0),Vec3(0,L[1],0),Vec3(0,0,L[2]))
  
  
    #============================================================================
    # Simulation Parameters
    Tinit = 373.15
    Tsim = 273.15 + T
    NAnnealSteps=100
    MaxFileSize = 1000 # number of frames per file
    
    n_equilsteps = int(t_equilibrate/t_step)
    n_Files = int(ceil(t_simulate/t_reportinterval/float(MaxFileSize)))
    n_reportinterval = int(t_reportinterval/t_step)

    
    FileSizes= [MaxFileSize for i in range(n_Files)]
    LastFileSize = int(mod(int(t_simulate/t_reportinterval),MaxFileSize))
    if LastFileSize != 0:
        FileSizes[-1] = LastFileSize  
    
    
    if water_forcefield == 'swm4ndp':
        	integrator = DrudeLangevinIntegrator(Tinit*kelvin, 20.0/picosecond, 
                                              1.0*kelvin, 5.0/picosecond, t_step)
    else:
        	integrator = LangevinIntegrator(Tinit*kelvin, 20.0/picosecond,  t_step)
  
    #platform = Platform.getPlatformByName('OpenCL')
    
    platform = Platform.getPlatformByName(PlatformName)
  #  CUDA_device_number={0.0:"0", 0.17: "1", 0.32:"2", 0.33:"2", 0.5:"3"}
  #  platform.setPropertyDefaultValue("CudaDevice", CUDA_device_number[z])
    
    xmlsys= XmlSerializer.serialize(system)
    f=open('system_'+run_label+'.xml','w')
    f.write(xmlsys)
    f.close()

    #============================================================================
    # Setting up a working simulation (without Particle number nan exception)
    
    if CheckPointFileAvail == False:
        SetupFlag = 0
        
        
        while SetupFlag >=0 and SetupFlag<5:
            
            #try:
         
            # Generate molecular positions 
            if InitPositionPDB == None:
                AllPositions=MolPositions(Sol_Nsol,L)
                AllPositions_Atoms=[x for sublist in AllPositions for x in sublist]*nanometer
            else:
                pdb = PDBFile(InitPositionPDB)
                AllPositions_Atoms=pdb.getPositions()
            print('Completed initializing molecule positions...')
            
            # simulation setup
            
            simulation = Simulation(Topo, system, integrator, platform)  
            simulation.context.setPositions(AllPositions_Atoms)
            simulation.minimizeEnergy()
            
            # simulated annealing to equilibrium configuration
            timestart=datetime.datetime.now()
            print('Annealing the system...')
            print('Target T[K], Current Simulation T[K]')
            for i in range(NAnnealSteps+1):
                TAnnealStep=Tsim+(Tinit-Tsim)/float(NAnnealSteps)*(NAnnealSteps-i)
                integrator.setTemperature(TAnnealStep*kelvin)
                print('%11.2f,%11.2f' %(Tsim, TAnnealStep))
                simulation.step(1000)
            
            timestop=datetime.datetime.now()
            timelapsed = str(datetime.timedelta(seconds=(timestop-timestart).seconds))
            print("T = %.2f deg-C:: computing time for annealing %s" %(T,timelapsed))
            
            # modify SetupFlag
            SetupFlag = -1
                
            # except:
            #     SetupFlag += 1
                
            #     print('exception in initial config set up catched for T = %i' % T)
                
            #     if SetupFlag == 5:
            #         raise Exception('Unable to setup a working configuration for \
            #                         the molecules, possibly due to the energy.')

    else:
        
        AllPositions_Atoms=[Vec3(0.0, 0.0, 0.0) for x in range(system.getNumParticles())]
        simulation = Simulation(Topo, system, integrator, platform)
        simulation.context.setPositions(AllPositions_Atoms*nanometer)
        with open('cp_'+run_label+'.chk', 'rb') as f:
            simulation.context.loadCheckpoint(f.read())
                
    #=========================================================================
    # equilibration
    timestart=datetime.datetime.now()
    simulation.step(n_equilsteps)  
    timestop=datetime.datetime.now()
    timelapsed = str(datetime.timedelta(seconds=(timestop-timestart).seconds))
    print("T = %.2f deg-C:: computing time for equilibration %s" %(T,timelapsed))
    
    initial_state=simulation.context.getState(getPositions = True, getEnergy=True)
    initial_state_pdb=PDBReporter('inistate_'+run_label+'.pdb',1)
    initial_state_pdb.report(simulation, initial_state)
    print("Initial potential energy = %s" %initial_state.getPotentialEnergy())    
    
    #=========================================================================
    # Actual simulation  
    
    print('Simulation in progress...')
    
    for i_file in range(n_Files):
        
        timestart=datetime.datetime.now()
    
        # files to write        
        filelabel = (run_label+'_%i' %i_file)
        timenow = datetime.datetime.now().strftime('_%y%m%d%H%M')
        
        simulation.reporters.append(
        StateDataReporter('statedata_'+filelabel+'.txt',n_reportinterval, 
                      step=True, time=True,potentialEnergy=True, temperature=True)) 
                      
        simulation.reporters.append(
        DCDReporter('dcd_'+filelabel+'.dcd', n_reportinterval))      
        
        if ReportVelocity == True:
            VelocityReport = open('velocity_'+filelabel+'.txt','w')
        

        for steps in range(FileSizes[i_file]):
            
            # simulation
            simulation.step(n_reportinterval)
            
            # report velocity
            if ReportVelocity == True:
                state_now=simulation.context.getState(getPositions = True, getVelocities = True)
            
                state_time=state_now.getTime().value_in_unit(picosecond)
                state_velocity=state_now.getVelocities().value_in_unit(nanometer/picosecond)   
            
                VelocityReport.write("Velocity xyz in nm/ps at Time = %f ps \n" %state_time )
              
                for j in range(NAtoms):
                    atom = MyAtoms[j]
                    VelocityReport.write("%i,%3s,%2s," %(atom.index, atom.residue.name,atom.name))
                    VelocityReport.write("%+.8f,%+.8f,%+.8f\n"  %state_velocity[j])
            
        timestop=datetime.datetime.now()
        timelapsed = str(datetime.timedelta(seconds=(timestop-timestart).seconds))
        print("T = %i :: time for simulation of file %s \n" %(T,timelapsed))
        
        # Stop report for this step
        simulation.reporters=[]
        
        if ReportVelocity == True:
            VelocityReport.close()
        
        
        # writing checkpoint file
        with open('cp_'+run_label+'.chk', 'wb') as f:
            f.write(simulation.context.createCheckpoint())
            
            
        # running analysis script after every file generated
        #ComputeOrderParam((run_label+timenow,'.'))
        
        # removing raw data files
        # os.remove('dcd_'+run_label+timenow+'.dcd')
        # os.remove('statedata_'+run_label+timenow+'.txt')
        
    #ComputeOrderParam((run_label,'.')) # only for run06

    
    """
    state_now=simulation.context.getState(getPositions = True, getEnergy=True)
    state_now_pdb=PDBReporter('statenow_'+run_label+'.pdb',1)
    state_now_pdb.report(simulation, state_now)
    print state_now.getPotentialEnergy()
    """
    



