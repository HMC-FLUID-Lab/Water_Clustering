### Generate Initial Positions of moleculues on a lattice

from numpy import sqrt, matrix, array, dot,ceil
from math import pi,sin,cos,asin
from random import randrange
from openmm.unit.unit_math import norm
import numpy as np





def rotation_matrix(axis,theta):
    """
    3D rotation matrix
    
    example:
    v = np.array([3,5,0])
    axis = np.array([4,4,1])
    theta = 1.2 
    print(np.dot(rotation_matrix(axis,theta),v))
    """

    axis = axis/np.sqrt(np.dot(axis,axis))
    a = np.cos(theta/2)
    b,c,d = -axis*np.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])




def distance(a,b):
    dist=sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
    return dist






def check_pos(MoleculePos,AllPositions,L): 
    """
    function to check if a proposed position for water is outside the simulation box
    or too near other water molecules
    return False if the location of water molecule is not acceptable
    """    
    
    
    test=True
    A = [s for i in range(len(AllPositions)) for s in AllPositions[i] ]
    n = len(A)    
    i = 0
    
    for xyz in MoleculePos:   # check molecule inside Box
        for i in range(3):        
            test = test and xyz[i]<L[i]/2 and xyz[i]>-L[i]/2
            
    
    while test==True and i<n:       # check molecules doe not overlap
        for xyz in MoleculePos:
            test = test and distance(xyz,A[i])>0.1
                    
        i +=1
        
    return test



def PlaceMo(sol,AllPositions,lat_pts,L):
    """
    Finding acceptable position for a molecule
    """

    
    search_switch=True
    rot_axis=(np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]), \
              np.array([1,0,1]), np.array([1,1,0]), np.array([0,1,1]))
    rot_theta=(pi/4,-pi/4,pi*3/4,-pi*3/4, pi, pi/2, -pi/2, 0.0)
    #rot_theta=(0.0,)
    
    
    try_sol_positions =[list(array(sol.positions[i])) for i in range(sol.numparticles)] 
       # initial solvent positions
    
    while search_switch==True:
        
        try_lat_index = randrange(len(lat_pts))
        lat           = lat_pts[try_lat_index]
        
        num_trials=0
        
        while search_switch == True and num_trials<6 :
                
            try_rot_axis =rot_axis[randrange(len(rot_axis))]
            try_rot_theta =rot_theta[randrange(len(rot_theta))]
        
            try_sol_pos   = [list(array(lat)\
            +np.dot(rotation_matrix(try_rot_axis,try_rot_theta),array(try_sol_positions[i]))) \
            for i in range(sol.numparticles)]
                
            acceptable    = check_pos(try_sol_pos,AllPositions,L)
        
            if acceptable == True:
                search_switch=False
                lat_pts.pop(try_lat_index)
            
            #print('num_trial %3.3f rot_axis %8.3f rot_theta %8.3f\n' %(num_trials, try_rot_axis[0] , try_rot_theta))
            num_trials +=1
            
    return (try_sol_pos,lat_pts)
        

def BoxSize_NumPar(sol_Nsol,aspect_ratio=(1,1,1)):   
    N=0
    V=0    
    
    for i in range(len(sol_Nsol)):
        N += sol_Nsol[i][1]
        V += sol_Nsol[i][0].v*sol_Nsol[i][1]
    
    s = aspect_ratio[0]*aspect_ratio[1]*aspect_ratio[2]
    L = (V/float(s))**(1/3.0)
    L_tuple = (aspect_ratio[0]*L,aspect_ratio[1]*L ,aspect_ratio[2]*L)
    

    return L_tuple,N
    



def MolPositions(sol_Nsol,LBox, exclude_radius=0):
    """
    generates the positions for all molecules in the system
    sol_N=[(atom1,N1,pos1),...(sol1,N1),(sol2,N2), ... ]
    
    returns AllPositions = [list of atomic position for molecule 1, list of atomic positions for molecule 2, etc]
    """
    
    
    #print("Generating initial placement of molecules ...")
    
    
    
    L = LBox
    
    N= BoxSize_NumPar(sol_Nsol)[1]
    
    v = L[0]*L[1]*L[2]/N    # average molecular volume
    #h = v**(1/3.0)/grid_density        # lattice grid size
    h= L[0]/ceil(N**(1/3))

    n = [int(L[i]/h) for i in range(3)]
    
    lat_pts=[(-L[0]/2+h/2+i*h,-L[1]/2+h/2+j*h,-L[2]/2+h/2+k*h) 
              for i in range(n[0]) for j in range(n[1]) for k in range(n[2])]
    
    
    if exclude_radius>0:             
        for i in reversed(range(len(lat_pts))):
            if norm(lat_pts[i])<exclude_radius+0.05:
                lat_pts.pop(i)
    
    #for i in range(len(lat_pts)):
    #    print norm(lat_pts[i])
    
    AllPositions = [list([]) for _ in range(len(sol_Nsol))]
    Sol_needPos= list(range(len(sol_Nsol)))
    for i in reversed(range(len(sol_Nsol))):
        #if hasattr(sol_Nsol[i][0], 'position'): 
        #    AllPositions[i].append([x for x in sol_Nsol[i][0].position[0]])
        #    Sol_needPos.pop(i)    
        if len(sol_Nsol[i]) == 3:# check if position is set for the molecule
            AllPositions[i] = sol_Nsol[i][2]+array(sol_Nsol[i][0].positions)
            Sol_needPos.pop(i)# remove those atoms at fixed positions
    
    AllMoIndex =  [(i,j,sol_Nsol[i][0].v) for i in  Sol_needPos for j in range(sol_Nsol[i][1])]  
    AllMoIndex = reversed(sorted(AllMoIndex, key=lambda tup: tup[2])) # sort according to volume      
        

    #f=open('xyz.xyz','w')    
    #f.write(str(N*5))
    #f.write('\n\n')
 
    
    for MoIndex in AllMoIndex:
        #print(MoIndex)
        #print(len(lat_pts))
        
        sol_type = MoIndex[0]  
        sol=sol_Nsol[MoIndex[0]][0]
        
        ThisMoleculePos,lat_pts=PlaceMo(sol,AllPositions,lat_pts,L)
        AllPositions[sol_type].extend(ThisMoleculePos)
        
    
        for i in range(len(ThisMoleculePos)):
            item=ThisMoleculePos[i]
            
            #f.write('%-6s %8.3f %8.3f %8.3f\n' %(sol.atoms[i].name,item[0]*10,item[1]*10,item[2]*10))
    
    
    #print("Completed initial placement of molecules.")
    return AllPositions
    
      
def RotateMolecule(Molecular_xyz, axis_theta_list, center_xyz=None):
    """
    Molecular_xyz = xyz of all atoms in molecule
    axis_theta_list = list of (axis,theta) tuples where axis is the rotation
                      axis and theta is the angle. Each rotation would be 
                      successively performed
    center_xyz = center for the rotation
    """
    
    if center_xyz == None:
        center_xyz=Molecular_xyz[0]
    
    for s in axis_theta_list:
        axis = s[0]
        theta = s[1]
        
        M = rotation_matrix(axis,theta);
        rS=array(Molecular_xyz)-array(center_xyz)
        rot_rS = dot(M,rS.transpose()).transpose()
        new_xyz = array(center_xyz) + rot_rS   

        Molecular_xyz = [tuple(i) for i in new_xyz]
    
    return Molecular_xyz
    
    


