# -*- coding: utf-8 -*-
"""
Program: Tree growth 

Objective:  - Time marching a tree growing for visualization. 
            - Trunk forms parabolically
            - Branches form as linear, parabolic, cubic to quadic(?)
            - Later stages maybe implement a neural network for functions. 
           
"""

print('\n\n\nInitialize libraries...\n')

import numpy as np
from numpy.linalg import inv 
print('Numerical module initialized.')
import scipy as sci
print('Scientific module initialized.')
import matplotlib.pylab as plt
import matplotlib.pyplot as plt2
from mpl_toolkits.mplot3d import Axes3D
print('Plotting module initialized.')

import sys as sys
import psutil as psutil                    # This is a system management thing
print('System modules initialized.')
import pandas as pand
print('Data handling module initialized.')

import copy as copy
import multiprocessing as mp
import random as rand
import time as time
print('Additional modules initialized.')

print('\nInitilization complete.')

start = time.time()


""" ------------------------- START OF CODE -----------------------------------
"""

# Initiate dummy classes for important variables such as control and discretization.
# Collect groups of variables (objects) in a group 
class CV: pass   # Control variables
class DV: pass   # Discretization variables 
class WV: pass   # Working variables
    
# Discretization properties
DV.NP    = 3         # Number of polynomials per branch and trunk.
DV.Pn    = 100       # Discretized points along polynomials
DV.Ncp   = 4         # Number of control points along branch and trunk Note this is linked to 3rd order poly's. Cant change it. 

# INITIAL Trunk dimensions
CV.Th_v = 10
CV.Tr_max = 1
CV.Tr_min = CV.Tr_max/2


""" Trunk and branch properties """
CV.Nb = 100        # Estimate number of branches/Memory increase of spatial dynamic arrays

# Polynomial
CV.Tmax_x = 0.1    # Max trunk varience in x for polynomial
CV.Tmax_y = 0.1    # Max trunk varience in y for polynomial
CV.Bmax_x = 0.2    # Max branch varience in y_bc # Note bc is branch coordinate system. 
CV.Bmax_y = 0.2    # Max branch varience in z_bc

# Aesthetic
CV.start_B  = 0.7  # Percentage trunk where branches start

# Radii of branches
CV.PBr_max = 0.4   # Primary max branch radius
CV.PBr_min = 0.2   # Primary min branch primary radius
CV.SBr_max = 0.2   # Secondary "" ""
CV.SBr_min = 0.1   # "" ""
CV.TBr_max = 0.1   # Tertiary "" ""
CV.TBr_min = 0.01  # "" ""

# Branches per segment(z) on trunk 
CV.Nb_dhmax = 2
CV.Nb_dhmin = 1 

# Branch length (based on trunk height)
CV.PBL_max = (CV.Th_v/4) 
CV.PBL_min = (CV.Th_v/8)
CV.SBL_max = CV.PBL_max/4
CV.SBL_min = CV.PBL_max/8

# Branch gradient on trunk cartesian coordinate system
CV.Spmax = 3   # Upper limit
CV.max_pspace = 10     # INDEX spacing between primary branchs in trunk cart. coord. in z axis
CV.min_pspace = 5  


# ------------------- Define functions -------------------------------------


# This function sets the polynomial on the x-y plane. (Varies y in terms of x)

from polycall import polycall_vy_x, polycall_vz_x, polycall_vy_z, polycall_vx_z, polycall_vx_y, polycall_vz_y

# Set dictionary for different polynomials creatiosn with respect to axis 
function_caller = {'poly_yx':polycall_vy_x, 
                   'poly_zx':polycall_vz_x,
                   'poly_xz':polycall_vx_z,
                   'poly_yz':polycall_vy_z,
                   'poly_xy':polycall_vx_y,
                   'poly_zy':polycall_vz_y}


# ----------------------- Pre-allocate memory ---------------------------------


# Initiate control point three dimensional array 
# Data i = x -> z [0,1,2], j = Cp's along branch (4) => [0,1,2,3], k = trunk, branch 1,2,3... 
Cp = np.zeros((3,DV.NP+1,CV.Nb),dtype = np.float64)

# Initiate "skeletal" tree matrix (coordinates)
# rows = spatial dimensions (x,y,z)
# cols is number of discretized points along a branch (+1 for end point)
# extrusions for different branches, note index 0 for trunk
Sk  = np.zeros([3,(DV.NP*DV.Pn)+1,CV.Nb]) 



# ----- Generate trunk profile (3rd order piecewise polynomial)------

k = 0

# Initiate trunk at origin and generate random variables for control points
Cp[0,:,k] = np.array([0,round(rand.uniform(-CV.Tmax_x,CV.Tmax_x),2),round(rand.uniform(-CV.Tmax_x,CV.Tmax_x),2),round(rand.uniform(-CV.Tmax_x,CV.Tmax_x),2)],dtype = np.float64)   # Initial trunk y position
Cp[1,:,k] = np.array([0,round(rand.uniform(-CV.Tmax_y,CV.Tmax_y),2),round(rand.uniform(-CV.Tmax_y,CV.Tmax_y),2),round(rand.uniform(-CV.Tmax_y,CV.Tmax_y),2)],dtype = np.float64)   # Initial trunk y position
Cp[2,:,k] = np.array([0,round(CV.Th_v*1/4,1),                             round(CV.Th_v*3/4,1),                             CV.Th_v])  # Initial trunk z position

# Call functions to generate polynomials for trunk profile 
function_caller.get('poly_xz')(Cp,k,Sk,DV)
function_caller.get('poly_yz')(Cp,k,Sk,DV)

"""
# Make trunk strait line
Cp[0,:,k] = np.array([0,0,0,0],dtype = np.float64)   # Initial trunk y position
Cp[1,:,k] = np.array([0,0,0,0],dtype = np.float64)   # Initial trunk y position
Cp[2,:,k] = np.array([0,round(CV.Th_v*1/4,1),                             round(CV.Th_v*3/4,1),                             CV.Th_v])  # Initial trunk z position
z_p0 = np.arange(Cp[2,0,k],  Cp[2,1,k]                                    ,  (1/DV.Pn)*(Cp[2,1,k]- Cp[2,0,k]))
z_p1 = np.arange(Cp[2,1,k],  Cp[2,2,k]                                    ,  (1/DV.Pn)*(Cp[2,2,k]- Cp[2,1,k]))
z_p2 = np.arange(Cp[2,2,k],  Cp[2,3,k] + (1/(2*DV.Pn))*(Cp[2,3,k]- Cp[2,2,k]),  (1/DV.Pn)*(Cp[2,3,k]- Cp[2,2,k]))
        
# Parse data from function onto tree spatial matrix
Sk[2,:,k] = np.concatenate([z_p0, z_p1, z_p2])  # z data
"""

# Establish varying radius along trunk in relation to height
Tr = np.arange(CV.Tr_max,   CV.Tr_min -((CV.Tr_max-CV.Tr_min)/(DV.NP*DV.Pn)),   
               -((CV.Tr_max-CV.Tr_min)/(DV.NP*DV.Pn)))
Pr = np.arange(CV.PBr_max,   CV.PBr_min -((CV.PBr_max-CV.PBr_min)/(DV.NP*DV.Pn)),   
               -((CV.PBr_max-CV.PBr_min)/(DV.NP*DV.Pn)))
"""# Constant radius
Tr = np.tile(1,(301))
"""
# ----- Calculate primary branch origins and estimate number of branches ------
 
# Call variable start_b and apply it to index of trunk z positions 
k        = round(CV.start_B*np.size(Sk[2,:,0]))

# Establish random index to skip portions of tree
# Do not want to have uniform distribution of branches (probably, depends on the tree)
# Initializing at 0 to start at Nk_TS
WV.Nk_skip = 0

# Establish correct index for Cp array (since starting at Nk_TS) (+1 to not overwrite trunk Cps)
# Nk_skip also needs to be accounted for
ko = 1    

# ext for incase more branches are present than Cp,Sk array size. Used to initiate
# condition to extend static matrix. (Basically making a static into dynamic array)
WV.ext = 1

# Loop along trunk. 
# Objective: Generate control points of branches and estimate how many there is. 
# Important note, Sk[[0:2],:,0] is the trunk x,y,z coordinates 
# k = branch number
# ko = for data allocation on Cp array
while k < np.size(Sk[2,:,0]):
    
    """
    Geometry and control point section of loop
    """    
    # Establish number of branches on segment 
    WV.Nb_ds = rand.randint(CV.Nb_dhmin,CV.Nb_dhmax)
    
    """ Primary branches loop for generating branches at each segment
    """
    for i in range(0,WV.Nb_ds):
        # Starts at Nk_TS!    
        # Initial Cp of branches  -  k positions such that branch(k) starts at correct point along trunk!
        # Note, march along length y on branch reference frame
        
        # Branch length function of radius of branch and current trunk height
        WV.BL = rand.uniform(CV.PBL_min*(CV.Th_v/(Sk[2,k,0])),CV.PBL_max*(CV.Th_v/(Sk[2,k,0])))

    
        """ Establish - PRIMARY - branch control points IN BRANCH COORDINATE SYSTEM!
        """        
        # Cps in BRANCH COORDINATE SYSTEM. TRANSFORM AFTER! 
        # Poly dependent (march down) is the x dimension. 
        
        Cp[0,:,ko] = np.array([0 , 
                              round(WV.BL/6,1),
                              round(WV.BL/3,1),
                              round(WV.BL,1)],                                      dtype = np.float64)        
    
        Cp[1,:,ko] = np.array([0,  
                               round(rand.uniform(-CV.PBr_max/2,CV.PBr_max/2),2),
                               round(rand.uniform(-CV.PBr_max/2,CV.PBr_max/2),2),
                               round(rand.uniform(-CV.PBr_max/2,CV.PBr_max/2),2)],     dtype = np.float64)
    

        # Note, z axis is same for all coord systems. So origin is still Sk[2,k,0]
        # z origin Sk[2,k,0]
        
        # Gradient of primary branches in relation to trunk cartesian on xz plane
        WV.Sp = rand.uniform(0,CV.Spmax)
        
        # mx + z offset to add gradient onto control points for z (makes branches go upwards)        
        Cp[2,:,ko] = np.array([0 ,  
                               WV.Sp*Cp[0,1,ko]+ round(rand.uniform(-CV.PBr_max/2,CV.PBr_max/2),2),
                               WV.Sp*Cp[0,2,ko]+ round(rand.uniform(-CV.PBr_max/2,CV.PBr_max/2),2),
                               WV.Sp*Cp[0,3,ko]+ round(rand.uniform(-CV.PBr_max/2,CV.PBr_max/2),2)],     dtype = np.float64)
    
        """ Call polynomial functions to generate polynomials for primary branches.
            This generates primary branch coordinates in primary branch coordinate system (Sk[:,:,ko]"""
        function_caller.get('poly_yx')(ko,Cp,Sk,DV)
        function_caller.get('poly_zx')(ko,Cp,Sk,DV)
        print('Primary branch')
        
        """ 
        Establish secondary branches. Each has its own coordinate system relative to primary branch coordinate system 
        - Need to march down branch
        - Establish one or two branch(es) per segment 
        - Establish length, gradient initial etc. 
        - Same process as trunk 
        """ #

        
        """    Inner index management section of loop    """
        # Establish correct index for Cp array. March index with respect to branch(k) iterations 
        ko = ko + 1
        
        # Extend arrays if number of branches exceeds pre-allocated Cp,Sk dimensions. 
        if ko == (WV.ext*CV.Nb)-1: 
            Sk = np.concatenate((Sk[:,:,:],np.zeros((3,(DV.NP*DV.Pn)+1,CV.Nb))),axis = 2) 
            Cp = np.concatenate((Cp[:,:,:],np.zeros((3,    DV.NP+1,    CV.Nb))),axis = 2)
         
            # Prep for next static array extension. 
            WV.ext = WV.ext + 1 
    
    
    """
    Index management section of loop
    """
    # Establish random index to skip portions of tree
    # Do not want to have uniform distribution of branches (probably, depends on the tree)
    # ----NOTE: Need to account for radius of branches i.e. + ceil(max(Tr)/(NP*PN))
    WV.Nk_skip = rand.randint(copy.deepcopy(CV.min_pspace),copy.deepcopy(CV.max_pspace))   


    # + Nk_skip: March k in relation to Nk_skip value (Marching down trunk at random intervals)
    k = k + WV.Nk_skip 

end  = time.time()
print(f'Time taken: {end-start:.2f} seconds')

ax  = plt.figure().add_subplot(111, projection='3d',aspect='equal');   
for i in range (0,ko):
    if i == 0 :
        ax.plot_wireframe(Sk[0,0:-1,i], Sk[1,0:-1,i], Sk[2,0:-1,i].reshape(1,-1),color='tab:brown', linewidth = 15)
    else:    
        ax.plot_wireframe(Sk[0,0:-1,i], Sk[1,0:-1,i], Sk[2,0:-1,i].reshape(1,-1),color='tab:brown', linewidth = 3)
        ax.scatter3D(Sk[0,0,i], Sk[1,0,i], Sk[2,0,i].reshape(1,-1))
    plt.draw

"""
ax  = plt.figure(1).add_subplot(111, projection='3d');   
ax.plot_wireframe(Sk[0,0:-1,ko-2], Sk[1,0:-1,ko-2], Sk[2,0:-1,ko-2].reshape(1,-1))
plt.draw
"""
