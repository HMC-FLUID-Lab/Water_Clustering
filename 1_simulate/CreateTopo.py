"""
This script contains function that would create the Topology object of the system

Created June 2014
"""

from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

#from molecules import *
from MolPositions import *


def put_moledules(molecule, Topo, ThisChain):
    
    ThisResidue=Topo.addResidue(molecule.residue, ThisChain)
             
    AtomsInThisMolecue=[]
    for i in molecule.atoms:        
        ThisAtom=Topo.addAtom(i.name, i.element, ThisResidue)
        AtomsInThisMolecue.append(ThisAtom)
    for i in molecule.bonds:
        Topo.addBond(AtomsInThisMolecue[i[0]],AtomsInThisMolecue[i[1]])


def CreateTopo(Sol_Nsol):

    Topo=Topology()    

    ThisChain=Topo.addChain()  
    Atom_count=0

    for each_solvent in Sol_Nsol:
        for j in range(each_solvent[1]):
            put_moledules(each_solvent[0], Topo, ThisChain)
            Atom_count += each_solvent[0].numparticles

    return Topo, Atom_count



"""
### look at topology ###
MyAtoms = list(Topo.atoms()) # Create a list of atom objects.
MyBonds = list(Topo.bonds()) # Create a list of bonded atom pairs.

for atom in MyAtoms: # Loop through the atoms.
    print atom.name, atom.element, atom.index, atom.residue.name # Print the name of the atom.
for bond in MyBonds: # Loop through the bonded atom pairs.
    print bond[0].name, bond[1].name, bond[0].index, bond[1].index # Print the names of atoms in each bond.
"""  
