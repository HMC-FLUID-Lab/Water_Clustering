### Define Properties of Each Molecule ### 

from math import radians as rad
from math import pi,sin,cos,asin
from openmm.app import *
from openmm import *
from openmm.unit import *

class Molecule(object):
    """ represents a Moleule
        attributes: positions, atoms, bonds, residue"""    



#======================================================
def createWater(forcefield, pos_O=(0,0,0)):
    

                 
    
    if forcefield in ['swm4ndp']:
        """ 
        Pos_O = position of oxyten (in nanometers!)
        """        
        water=Molecule()
    
        water.v=0.0300053 # volume of water molecules (nm^3)
    
        r_OH=0.09572 #length of OH bond in nm
        r_OM=0.024034 #length of OM in nm, M is the aucillary particle    
        r_OD=-0.0005    
        d_OH=rad(104.52) # HOH angle
        pos_H1= (pos_O[0]-r_OH,pos_O[1],pos_O[2])
        pos_H2= (pos_O[0]-r_OH*cos(d_OH),pos_O[1]+r_OH*sin(d_OH),pos_O[2])
        pos_M = (pos_O[0]-r_OM*cos(d_OH/2),pos_O[1]+r_OM*sin(d_OH/2),pos_O[2])
        pos_D = (pos_O[0]-r_OD*cos(d_OH/2),pos_O[1]+r_OD*sin(d_OH/2),pos_O[2])
        
        water.positions=[pos_O, pos_H1, pos_H2, pos_M, pos_D]
        water.atoms=[Atom('O', Element.getBySymbol('O'),   0, 'HOH', 'HOH-O'),
                     Atom('H1',Element.getBySymbol('H'),   1, 'HOH', 'HOH-H1'),
                     Atom('H2',Element.getBySymbol('H'),   2, 'HOH', 'HOH-H2'),
                     Atom('M', None, 3, 'HOH','HOH-M'),
                     Atom('OD', None, 4, 'HOH','HOH-OD')]
        
        water.bonds = [(0,1),(0,2)] # list pairs of bonds 
        water.residue = 'HOH'
        water.drudepairs=[(4,0)]   # list pairs of drude particles      
        water.numparticles=5
    
    elif forcefield in ['tip4pfb']:
        """ 
        Pos_O = position of oxyten (in nanometers!)
        """        
        water=Molecule()
    
        water.v=0.0300053 # volume of water molecules (nm^3)
    
        r_OH=0.09572 #length of OH bond in nm
        r_OM=0.015   #length of OM in nm, M is the aucillary particle    
        d_OH=rad(104.52) # HOH angle (in radians)
        pos_H1= (pos_O[0]-r_OH,pos_O[1],pos_O[2])
        pos_H2= (pos_O[0]-r_OH*cos(d_OH),pos_O[1]+r_OH*sin(d_OH),pos_O[2])
        pos_M = (pos_O[0]-r_OM*cos(d_OH/2),pos_O[1]+r_OM*sin(d_OH/2),pos_O[2])
        
        water.positions=[pos_O, pos_H1, pos_H2, pos_M]
        water.atoms=[Atom('O', Element.getBySymbol('O'),   0, 'HOH', 'HOH-O'),
                     Atom('H1',Element.getBySymbol('H'),   1, 'HOH', 'HOH-H1'),
                     Atom('H2',Element.getBySymbol('H'),   2, 'HOH', 'HOH-H2'),
                     Atom('M', None, 3, 'HOH','HOH-M')]
        
        water.bonds = [(0,1),(0,2)] # list pairs of bonds 
        water.residue = 'HOH'
        water.numparticles=4

    elif forcefield in ['spce']:
        """
        SPC/E model built by Minwoo, SRP 2020
        """
        water=Molecule()

        water.v=0.0300053 # volume of water molecules (nm^3)
        r_OH=0.1 #length of OH bond in nm
        d_OH=rad(109.47) # HOH angle (in radians)
        pos_H1= (pos_O[0]-r_OH,pos_O[1],pos_O[2])
        pos_H2= (pos_O[0]-r_OH*cos(d_OH),pos_O[1]+r_OH*sin(d_OH),pos_O[2])
        water.positions=[pos_O, pos_H1, pos_H2]
        water.atoms=[Atom('O', Element.getBySymbol('O'),   0, 'HOH', 'HOH-O'),
                     Atom('H1',Element.getBySymbol('H'),   1, 'HOH', 'HOH-H1'),
                     Atom('H2',Element.getBySymbol('H'),   2, 'HOH', 'HOH-H2')]
        
        water.bonds = [(0,1),(0,2)] # list pairs of bonds 
        water.residue = 'HOH'
        water.numparticles=3
    
    elif forcefield in ['tip4p2005']:
        """
        TIP4P/2005 model built by Samen, SRP 2020
        """
        water=Molecule()
        
        water.v=0.0300053 # volume of water molecules (nm^3)
        r_OH=0.09572 #length of OH bond in nm
        r_OM=0.01546   #length of OM in nm, M is the aucillary particle    
        d_OH=rad(104.52) # HOH angle (in radians)
        pos_H1= (pos_O[0]-r_OH,pos_O[1],pos_O[2])
        pos_H2= (pos_O[0]-r_OH*cos(d_OH),pos_O[1]+r_OH*sin(d_OH),pos_O[2])
        pos_M = (pos_O[0]-r_OM*cos(d_OH/2),pos_O[1]+r_OM*sin(d_OH/2),pos_O[2])
        water.positions=[pos_O, pos_H1, pos_H2, pos_M]
        water.atoms=[Atom('O', Element.getBySymbol('O'),   0, 'HOH', 'HOH-O'),
                     Atom('H1',Element.getBySymbol('H'),   1, 'HOH', 'HOH-H1'),
                     Atom('H2',Element.getBySymbol('H'),   2, 'HOH', 'HOH-H2'),
                     Atom('M', None, 3, 'HOH','HOH-M')]
        water.bonds = [(0,1),(0,2)] # list pairs of bonds 
        water.residue = 'HOH'
        water.numparticles=4
        
    
    elif forcefield in ['tip5p']:
        """
        TIP5P model built by Dragan, SRP 2020
        """
        
        water = Molecule()

        water.v = 0.0300053  # volume of water molecules (nm^3)

        r_OH = 0.09572  # length of OH bond in nm
        r_OM = 0.07  # length of OM in nm
        d_OH = rad(104.52)  # HOH angle (in radians)
        d_OM = rad(109.47)  # MOM angle (in radians)

        pos_H1 = (pos_O[0] - r_OH, pos_O[1], pos_O[2])
        pos_H2 = (pos_O[0] - r_OH * cos(d_OH), pos_O[1] + r_OH * sin(d_OH), pos_O[2])
        pos_M1 = (pos_O[0], pos_O[1] + r_OM * cos(d_OM), pos_O[2] - r_OM * sin(d_OM))
        pos_M2 = (pos_O[0], pos_O[1], pos_O[2] - r_OM)

        water.positions = [pos_O, pos_H1, pos_H2, pos_M1, pos_M2]
        water.atoms = [Atom('O', Element.getBySymbol('O'), 0, 'HOH', 'HOH-O'),
                       Atom('H1', Element.getBySymbol('H'), 1, 'HOH', 'HOH-H1'),
                       Atom('H2', Element.getBySymbol('H'), 2, 'HOH', 'HOH-H2'),
                       Atom('M1', None, 3, 'HOH', 'HOH-M1'),
                       Atom('M2', None, 4, 'HOH', 'HOH-M2')]

        water.bonds = [(0, 1), (0, 2)]  # list pairs of bonds
        water.residue = 'HOH'
        water.numparticles = 5
    
    
    return water



def createTip4p2005Water(pos_O=(0,0,0)):
    """ 
    Pos_O = position of oxyten (in nanometers!)
    """        
    water=Molecule()

    water.v=0.0300053 # volume of water molecules (nm^3)

    r_OH=0.09572 #length of OH bond in nm
    r_OM=0.01546 #length of OM in nm, M is the aucillary particle    
    d_OH=rad(104.52) # HOH angle
    pos_H1= (pos_O[0]-r_OH,pos_O[1],pos_O[2])
    pos_H2= (pos_O[0]-r_OH*cos(d_OH),pos_O[1]+r_OH*sin(d_OH),pos_O[2])
    pos_M = (pos_O[0]-r_OM*cos(d_OH/2),pos_O[1]+r_OM*sin(d_OH/2),pos_O[2])
    
    water.positions=[pos_O, pos_H1, pos_H2, pos_M]
    water.atoms=[Atom('O', Element.getBySymbol('O'),   0, 'HOH', 'HOH-O'),
                 Atom('H1',Element.getBySymbol('H'),   1, 'HOH', 'HOH-H1'),
                 Atom('H2',Element.getBySymbol('H'),   2, 'HOH', 'HOH-H2'),
                 Atom('M', None, 3, 'HOH','HOH-M')]
    
    water.bonds = [(0,1),(0,2)] # list pairs of bonds 
    water.residue = 'HOH'
    water.numparticles=4
                 
    return water


#==========================================================================================

def Prism_pos(gamma,gamma2, b,theta):
    """    
    generate the position coordinates of a rotated prism
    b=edge length
    theta = angle between the three equal-length edges
    gamma = angle of the central perpendicular line relative to the vertical line
    """
    c=2.0*b*sin(theta/2.0)
    d=2.0/3.0*c*sin(pi/3)
    a=sqrt(b**2-d**2)
    alpha=asin(d/b)
    
    A=[-d,-a,0]       # coordinate in the upright position
    B=[d/2.0,-a, c/2.0]
    C=[d/2.0,-a, -c/2.0]
     
    # rotate codinate by angle gamma in the anticlockwise direction
    A=(cos(gamma)*A[0]-sin(gamma)*A[1],sin(gamma)*A[0]+cos(gamma)*A[1], A[2])
    B=(cos(gamma)*B[0]-sin(gamma)*B[1],sin(gamma)*B[0]+cos(gamma)*B[1], B[2])
    C=(cos(gamma)*C[0]-sin(gamma)*C[1],sin(gamma)*C[0]+cos(gamma)*C[1], C[2])    
    
    # rotate codinate by angle gamma in the anticlockwise direction
    A=(cos(gamma2)*A[0]-sin(gamma2)*A[2],A[1],sin(gamma2)*A[0]+cos(gamma2)*A[2])
    B=(cos(gamma2)*B[0]-sin(gamma2)*B[2],B[1],sin(gamma2)*B[0]+cos(gamma2)*B[2])
    C=(cos(gamma2)*C[0]-sin(gamma2)*C[2],C[1],sin(gamma2)*C[0]+cos(gamma2)*C[2])
    return (A, B, C)

