# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:24:50 2019

@author: nicho

Function ! 
"""

# Import relevant packages
import numpy as np
import random as rand
from numpy.linalg import inv


# Define library of functions

def polycall_vy_x(ko,Cp,Sk,DV):      
    k = ko 
    
    # Allocate memory for working matrices 
    Rm  = np.zeros([12, 1])
    CM  = np.zeros([12,12])
    Pcf = np.zeros([12,1])

    So = round(rand.uniform(0.1,0.2),2)             # Initial gradient 
    Se = round(rand.uniform(-0.5,0.5),2)            # End gradient

    Rm = np.array([[So],
                   [Cp[1,0,k]],
                   [Cp[1,1,k]],
                   [0],
                   [0],
                   [Cp[1,1,k]],
                   [Cp[1,2,k]],
                   [0],
                   [0],
                   [Cp[1,2,k]],
                   [Cp[1,3,k]],
                   [Se]],dtype = np.float64)
    
    # Connectivity coefficients from general equations of 3rd order poly's with
    # enforced equated gradients and curvature at connectivity points
    CM = np.array([[0, 1, 2*Cp[0,0,k], 3*Cp[0,0,k]**2, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, Cp[0,0,k], Cp[0,0,k]**2, Cp[0,0,k]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, Cp[0,1,k], Cp[0,1,k]**2, Cp[0,1,k]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 2*Cp[0,1,k], 3*Cp[0,1,k]**2, 0, -1, -2*Cp[0,1,k], -3*Cp[0,1,k]**2,0, 0, 0, 0],
                   [0, 0, 2, 6*Cp[0,1,k], 0, 0, -2, -6*Cp[0,1,k], 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, Cp[0,1,k], Cp[0,1,k]**2, Cp[0,1,k]**3, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, Cp[0,2,k], Cp[0,2,k]**2, Cp[0,2,k]**3, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 2*Cp[0,2,k], 3*Cp[0,2,k]**2, 0, -1, -2*Cp[0,2,k], -3*Cp[0,2,k]**2],
                   [0, 0, 0, 0, 0, 0, 2, 6*Cp[0,2,k], 0, 0, -2, -6*Cp[0,2,k]],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, Cp[0,2,k], Cp[0,2,k]**2, Cp[0,2,k]**3],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, Cp[0,3,k], Cp[0,3,k]**2, Cp[0,3,k]**3],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2*Cp[0,3,k], 3*Cp[0,3,k]**2]],dtype = np.float64)

    # Solving system of equations. Rm = CM*Pcf

    # Pcf[:,k] = inv(CM)@Rm.flatten() # Flatten for converting [12,1] to [12,]
    Pcf = inv(CM)@Rm
    
    # Interpolate between control points to plot poly's
    # E.g. Cp0 - Cp1, Cp1 - Cp2, Cp2 - Cp3
    if sum(Sk[0,:,k]) == 0 :
        x_p0 = np.arange(Cp[0,0,k],  Cp[0,1,k]                                       ,  (1/DV.Pn)*(Cp[0,1,k]- Cp[0,0,k]))
        x_p1 = np.arange(Cp[0,1,k],  Cp[0,2,k]                                       ,  (1/DV.Pn)*(Cp[0,2,k]- Cp[0,1,k]))
        x_p2 = np.arange(Cp[0,2,k],  Cp[0,3,k] + (1/(2*DV.Pn))*(Cp[0,3,k]- Cp[0,2,k]),  (1/DV.Pn)*(Cp[0,3,k]- Cp[0,2,k]))
        
        # Parse data from function onto tree spatial matrix
        Sk[0,:,k] = np.concatenate([x_p0, x_p1, x_p2])  # z data
    
    elif sum(Sk[0,:,k]) != 0 :
        pass

    # Calculate polynomial coordinates using general equations for 3rd order.
    #  x=                      a                                   + bx                                  +           cx^2                           +           bx^3
    y_p0 = np.tile(Pcf[0,0],(np.size(Sk[0,0:DV.Pn,k])           )) + Pcf[1,0]*Sk[0,0:DV.Pn,k]            +  Pcf[2,0] *Sk[0,0:DV.Pn,k]**2            +  Pcf[3,0]*Sk[0,0:DV.Pn,k] **3
    y_p1 = np.tile(Pcf[4,0],(np.size(Sk[0,DV.Pn:2*DV.Pn,k])     )) + Pcf[5,0]*Sk[0,DV.Pn:2*DV.Pn,k]      +  Pcf[6,0] *Sk[0,DV.Pn:2*DV.Pn,k]**2      +  Pcf[7,0]*Sk[0,DV.Pn:2*DV.Pn,k]**3
    y_p2 = np.tile(Pcf[8,0],(np.size(Sk[0,2*DV.Pn:3*DV.Pn+1,k]) )) + Pcf[9,0]*Sk[0,2*DV.Pn:3*DV.Pn+1,k]  +  Pcf[10,0]*Sk[0,2*DV.Pn:3*DV.Pn+1,k]**2  +  Pcf[11,0]*Sk[0,2*DV.Pn:3*DV.Pn+1,k]**3


    # Parse data from function onto tree spatial matrix
    # Note, flatten needed to convert to 1D to slice onto matrix
    Sk[1,:,k] = np.concatenate([y_p0, y_p1, y_p2])

""" Plotting to see if working as intended
    if k == 10 or k == 20 or k == 30 or k ==40 :
        fig = plt.figure(1)
        fig.add_subplot(111,aspect='equal')
        plt.plot(y_p0,Sk[0,0:Pn,k],'ro',y_p1,Sk[0,Pn:2*Pn,k],'go',y_p2,Sk[0,2*Pn:3*Pn+1,k],'bo')
"""


def polycall_vz_x(ko,Cp,Sk,DV):       
    
    k = ko 
    
    # Allocate memory for working matrices 
    Rm  = np.zeros([12, 1])
    CM  = np.zeros([12,12])
    Pcf = np.zeros([12,1])

    So = round(rand.uniform(0.1,0.2),2)             # Initial gradient 
    Se = round(rand.uniform(-0.5,0.5),2)            # End gradient

    Rm = np.array([[So],
                   [Cp[2,0,k]],
                   [Cp[2,1,k]],
                   [0],
                   [0],
                   [Cp[2,1,k]],
                   [Cp[2,2,k]],
                   [0],
                   [0],
                   [Cp[2,2,k]],
                   [Cp[2,3,k]],
                   [Se]],dtype = np.float64)
    
    # Connectivity coefficients from general equations of 3rd order poly's with
    # enforced equated gradients and curvature at connectivity points
    CM = np.array([[0, 1, 2*Cp[0,0,k], 3*Cp[0,0,k]**2, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, Cp[0,0,k], Cp[0,0,k]**2, Cp[0,0,k]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, Cp[0,1,k], Cp[0,1,k]**2, Cp[0,1,k]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 2*Cp[0,1,k], 3*Cp[0,1,k]**2, 0, -1, -2*Cp[0,1,k], -3*Cp[0,1,k]**2,0, 0, 0, 0],
                   [0, 0, 2, 6*Cp[0,1,k], 0, 0, -2, -6*Cp[0,1,k], 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, Cp[0,1,k], Cp[0,1,k]**2, Cp[0,1,k]**3, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, Cp[0,2,k], Cp[0,2,k]**2, Cp[0,2,k]**3, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 2*Cp[0,2,k], 3*Cp[0,2,k]**2, 0, -1, -2*Cp[0,2,k], -3*Cp[0,2,k]**2],
                   [0, 0, 0, 0, 0, 0, 2, 6*Cp[0,2,k], 0, 0, -2, -6*Cp[0,2,k]],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, Cp[0,2,k], Cp[0,2,k]**2, Cp[0,2,k]**3],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, Cp[0,3,k], Cp[0,3,k]**2, Cp[0,3,k]**3],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2*Cp[0,3,k], 3*Cp[0,3,k]**2]],dtype = np.float64)

    # Solving system of equations. Rm = CM*Pcf

    """
    Can parralelize this
    """
    # Pcf[:,k] = inv(CM)@Rm.flatten() # Flatten for converting [12,1] to [12,]
    Pcf = inv(CM)@Rm
    
    # Interpolate between control points to plot poly's
    # E.g. Cp0 - Cp1, Cp1 - Cp2, Cp2 - Cp3
    if sum(Sk[0,:,k]) == 0 :
        x_p0 = np.arange(Cp[0,0,k],  Cp[0,1,k]                                       ,  (1/DV.Pn)*(Cp[0,1,k]- Cp[0,0,k]))
        x_p1 = np.arange(Cp[0,1,k],  Cp[0,2,k]                                       ,  (1/DV.Pn)*(Cp[0,2,k]- Cp[0,1,k]))
        x_p2 = np.arange(Cp[0,2,k],  Cp[0,3,k] + (1/(2*DV.Pn))*(Cp[0,3,k]- Cp[0,2,k]),  (1/DV.Pn)*(Cp[0,3,k]- Cp[0,2,k]))
        
        # Parse data from function onto tree spatial matrix
        Sk[0,:,k] = np.concatenate([x_p0, x_p1, x_p2])  # z data
    
    elif sum(Sk[0,:,k]) != 0 :
        pass

    # Calculate polynomial coordinates using general equations for 3rd order.
    #  x=                      a                              + bx                            +           cx^2                     +           bx^3
    z_p0 = np.tile(Pcf[0,0],(np.size(Sk[0,0:DV.Pn,k])           )) + Pcf[1,0]*Sk[0,0:DV.Pn,k]            +  Pcf[2,0] *Sk[0,0:DV.Pn,k]**2            +  Pcf[3,0]*Sk[0,0:DV.Pn,k] **3
    z_p1 = np.tile(Pcf[4,0],(np.size(Sk[0,DV.Pn:2*DV.Pn,k])     )) + Pcf[5,0]*Sk[0,DV.Pn:2*DV.Pn,k]      +  Pcf[6,0] *Sk[0,DV.Pn:2*DV.Pn,k]**2      +  Pcf[7,0]*Sk[0,DV.Pn:2*DV.Pn,k]**3
    z_p2 = np.tile(Pcf[8,0],(np.size(Sk[0,2*DV.Pn:3*DV.Pn+1,k]) )) + Pcf[9,0]*Sk[0,2*DV.Pn:3*DV.Pn+1,k]  +  Pcf[10,0]*Sk[0,2*DV.Pn:3*DV.Pn+1,k]**2  +  Pcf[11,0]*Sk[0,2*DV.Pn:3*DV.Pn+1,k]**3


    # Parse data from function onto tree spatial matrix
    # Note, flatten needed to convert to 1D to slice onto matrix
    Sk[2,:,k] = np.concatenate([z_p0, z_p1, z_p2])

"""
    if k == 10 or k == 20 or k == 30 or k ==40 :
        fig = plt.figure(2)
        fig.add_subplot(111,aspect='equal')
        plt.plot(z_p0,Sk[0,0:Pn,k],'ro',z_p1,Sk[0,Pn:2*Pn,k],'go',z_p2,Sk[0,2*Pn:3*Pn+1,k],'bo')
"""

# Generate a 3rd order polynomial with varying x with respect to z
def polycall_vx_z(Cp,k,Sk,DV):      
    
    # Allocate memory for working matrices 
    Rm  = np.zeros([12, 1])
    CM  = np.zeros([12,12])
    Pcf = np.zeros([12,1])

    So = round(rand.uniform(0.1,0.2),2)             # Initial gradient 
    Se = round(rand.uniform(-0.5,0.5),2)            # End gradient
    
    Rm = np.array([[So],
                   [Cp[0,0,k]],
                   [Cp[0,1,k]],
                   [0],
                   [0],
                   [Cp[0,1,k]],
                   [Cp[0,2,k]],
                   [0],
                   [0],
                   [Cp[0,2,k]],
                   [Cp[0,3,k]],
                   [Se]],dtype = np.float64)
    
    # Connectivity coefficients from general equations of 3rd order poly's with
    # enforced equated gradients and curvature at connectivity points

    CM = np.array([[0, 1, 2*Cp[2,0,k], 3*Cp[2,0,k]**2, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, Cp[2,0,k], Cp[2,0,k]**2, Cp[2,0,k]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, Cp[2,1,k], Cp[2,1,k]**2, Cp[2,1,k]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 2*Cp[2,1,k], 3*Cp[2,1,k]**2, 0, -1, -2*Cp[2,1,k], -3*Cp[2,1,k]**2,0, 0, 0, 0],
                   [0, 0, 2, 6*Cp[2,1,k], 0, 0, -2, -6*Cp[2,1,k], 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, Cp[2,1,k], Cp[2,1,k]**2, Cp[2,1,k]**3, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, Cp[2,2,k], Cp[2,2,k]**2, Cp[2,2,k]**3, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 2*Cp[2,2,k], 3*Cp[2,2,k]**2, 0, -1, -2*Cp[2,2,k], -3*Cp[2,2,k]**2],
                   [0, 0, 0, 0, 0, 0, 2, 6*Cp[2,2,k], 0, 0, -2, -6*Cp[2,2,k]],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, Cp[2,2,k], Cp[2,2,k]**2, Cp[2,2,k]**3],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, Cp[2,3,k], Cp[2,3,k]**2, Cp[2,3,k]**3],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2*Cp[2,3,k], 3*Cp[2,3,k]**2]],dtype = np.float64)

    # Solving system of equations. Rm = CM*Pcf
    # Pcf[:,k] = inv(CM)@Rm.flatten() # Flatten for converting [12,1] to [12,]
    Pcf = inv(CM)@Rm
      
    # Interpolate between control points to plot poly's
    # E.g. Cp0 - Cp1, Cp1 - Cp2, Cp2 - Cp3
    if sum(Sk[2,:,k]) == 0 :
        z_p0 = np.arange(Cp[2,0,k],  Cp[2,1,k]                                       ,  (1/DV.Pn)*(Cp[2,1,k]- Cp[2,0,k]))
        z_p1 = np.arange(Cp[2,1,k],  Cp[2,2,k]                                       ,  (1/DV.Pn)*(Cp[2,2,k]- Cp[2,1,k]))
        z_p2 = np.arange(Cp[2,2,k],  Cp[2,3,k] + (1/(2*DV.Pn))*(Cp[2,3,k]- Cp[2,2,k]),  (1/DV.Pn)*(Cp[2,3,k]- Cp[2,2,k]))
        
        # Parse data from function onto tree spatial matrix
        Sk[2,:,k] = np.concatenate([z_p0, z_p1, z_p2])  # z data
    
    elif sum(Sk[2,:,k]) != 0 :
        pass

    # Calculate polynomial coordinates using general equations for 3rd order.
    #  x=                      a                              + bz                            +           bz^2                     +           bz^3
    x_p0 = np.tile(Pcf[0,0],(np.size(Sk[2,0:DV.Pn,k])           )) + Pcf[1,0]*Sk[2,0:DV.Pn,k]            +  Pcf[2,0]*Sk[2,0:DV.Pn,k]**2             +  Pcf[3,0]*Sk[2,0:DV.Pn,k] **3
    x_p1 = np.tile(Pcf[4,0],(np.size(Sk[2,DV.Pn:2*DV.Pn,k])     )) + Pcf[5,0]*Sk[2,DV.Pn:2*DV.Pn,k]      +  Pcf[6,0]*Sk[2,DV.Pn:2*DV.Pn,k]**2       +  Pcf[7,0]*Sk[2,DV.Pn:2*DV.Pn,k]**3
    x_p2 = np.tile(Pcf[8,0],(np.size(Sk[2,2*DV.Pn:3*DV.Pn+1,k]) )) + Pcf[9,0]*Sk[2,2*DV.Pn:3*DV.Pn+1,k]  +  Pcf[10,0]*Sk[2,2*DV.Pn:3*DV.Pn+1,k]**2  + Pcf[11,0]*Sk[2,2*DV.Pn:3*DV.Pn+1,k]**3


    # Parse data from function onto tree spatial matrix
    # Note, flatten needed to convert to 1D to slice onto matrix
    Sk[0,:,k] = np.concatenate([x_p0, x_p1, x_p2])


"""
    fig = plt.figure()
    fig.add_subplot(111,aspect='equal')
    plt.plot(z_p0,x_p0,'ro',z_p1,x_p1,'go',z_p2,x_p2,'bo')
"""


def polycall_vy_z(Cp,k,Sk,DV):   
    
    # Allocate memory for working matrices 
    Rm  = np.zeros([12, 1])
    CM  = np.zeros([12,12])
    Pcf = np.zeros([12,1])
    
    So = round(rand.uniform(0.1,0.2),2)             # Initial gradient 
    Se = round(rand.uniform(-0.5,0.5),2)            # End gradient

    # Establish results matrix for system of equations
    Rm = np.array([[So],
                   [Cp[1,0,k]],
                   [Cp[1,1,k]],
                   [0],
                   [0],
                   [Cp[1,1,k]],
                   [Cp[1,2,k]],
                   [0],
                   [0],
                   [Cp[1,2,k]],
                   [Cp[1,3,k]],
                   [Se]],dtype = np.float64)
    
    # Connectivity coefficients from general equations of 3rd order poly's with
    # enforced equated gradients and curvature at connectivity points
    CM = np.array([[0, 1, 2*Cp[2,0,k], 3*Cp[2,0,k]**2, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, Cp[2,0,k], Cp[2,0,k]**2, Cp[2,0,k]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, Cp[2,1,k], Cp[2,1,k]**2, Cp[2,1,k]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 2*Cp[2,1,k], 3*Cp[2,1,k]**2, 0, -1, -2*Cp[2,1,k], -3*Cp[2,1,k]**2,0, 0, 0, 0],
                   [0, 0, 2, 6*Cp[2,1,k], 0, 0, -2, -6*Cp[2,1,k], 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, Cp[2,1,k], Cp[2,1,k]**2, Cp[2,1,k]**3, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, Cp[2,2,k], Cp[2,2,k]**2, Cp[2,2,k]**3, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 2*Cp[2,2,k], 3*Cp[2,2,k]**2, 0, -1, -2*Cp[2,2,k], -3*Cp[2,2,k]**2],
                   [0, 0, 0, 0, 0, 0, 2, 6*Cp[2,2,k], 0, 0, -2, -6*Cp[2,2,k]],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, Cp[2,2,k], Cp[2,2,k]**2, Cp[2,2,k]**3],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, Cp[2,3,k], Cp[2,3,k]**2, Cp[2,3,k]**3],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2*Cp[2,3,k], 3*Cp[2,3,k]**2]],dtype = np.float64)

    # Solving system of equations. Rm = CM*Pcf

    # Pcf[:,k] = inv(CM)@Rm.flatten() # Flatten for converting [12,1] to [12,]
    Pcf = inv(CM)@Rm
      
    # Interpolate between control points to plot poly's
    # E.g. Cp0 - Cp1, Cp1 - Cp2, Cp2 - Cp3
    # Condition to stop calculating twice (Need to calculate y and x variations with z)
    if sum(Sk[2,:,k]) == 0 :
        z_p0 = np.arange(Cp[2,0,k],  Cp[2,1,k]                                       ,  (1/DV.Pn)*(Cp[2,1,k]- Cp[2,0,k]))
        z_p1 = np.arange(Cp[2,1,k],  Cp[2,2,k]                                       ,  (1/DV.Pn)*(Cp[2,2,k]- Cp[2,1,k]))
        z_p2 = np.arange(Cp[2,2,k],  Cp[2,3,k] + (1/(2*DV.Pn))*(Cp[2,3,k]- Cp[2,2,k]),  (1/DV.Pn)*(Cp[2,3,k]- Cp[2,2,k]))
        
        # Parse data from function onto tree spatial matrix
        Sk[2,:,k] = np.concatenate([z_p0, z_p1, z_p2])  # z data
    
    elif sum(Sk[2,:,k]) != 0 :
        pass



    # Calculate polynomial coordinates using general equations for 3rd order.
    #  y=                      a                              + bz                            +           bz^2                     +           bz^3
    y_p0 = np.tile(Pcf[0,0],(np.size(Sk[2,0:DV.Pn,k])           )) + Pcf[1,0]*Sk[2,0:DV.Pn,k]            +  Pcf[2,0]*Sk[2,0:DV.Pn,k]**2             +  Pcf[3,0]*Sk[2,0:DV.Pn,k] **3
    y_p1 = np.tile(Pcf[4,0],(np.size(Sk[2,DV.Pn:2*DV.Pn,k])     )) + Pcf[5,0]*Sk[2,DV.Pn:2*DV.Pn,k]      +  Pcf[6,0]*Sk[2,DV.Pn:2*DV.Pn,k]**2       +  Pcf[7,0]*Sk[2,DV.Pn:2*DV.Pn,k]**3
    y_p2 = np.tile(Pcf[8,0],(np.size(Sk[2,2*DV.Pn:3*DV.Pn+1,k]) )) + Pcf[9,0]*Sk[2,2*DV.Pn:3*DV.Pn+1,k]  +  Pcf[10,0]*Sk[2,2*DV.Pn:3*DV.Pn+1,k]**2  +  Pcf[11,0]*Sk[2,2*DV.Pn:3*DV.Pn+1,k]**3


    Sk[1,:,k] = np.concatenate([y_p0, y_p1, y_p2])
    
    polycall_vy_z.test = 1


"""
    # Plot branches or trunk independently
    fig = plt.figure()
    fig.add_subplot(111,aspect='equal')
    plt.plot(z_p0,x_p0,'ro',z_p1,x_p1,'go',z_p2,x_p2,'bo')
"""


def polycall_vx_y(ko,Cp,Sk,DV):      
    
    # I really hope this doesnt save! Doesnt! VICTORY 
    # Will probably need an expression to change k = for different branches? 
    # Or just use them all as ko? 
    k = ko 
    
    # Allocate memory for working matrices 
    Rm  = np.zeros([12, 1])
    CM  = np.zeros([12,12])
    Pcf = np.zeros([12,1])

    So = round(rand.uniform(0.1,0.2),2)             # Initial gradient 
    Se = round(rand.uniform(-0.5,0.5),2)            # End gradient

    Rm = np.array([[So],
                   [Cp[0,0,k]],
                   [Cp[0,1,k]],
                   [0],
                   [0],
                   [Cp[0,1,k]],
                   [Cp[0,2,k]],
                   [0],
                   [0],
                   [Cp[0,2,k]],
                   [Cp[0,3,k]],
                   [Se]],dtype = np.float64)
    
    # Connectivity coefficients from general equations of 3rd order poly's with
    # enforced equated gradients and curvature at connectivity points
    CM = np.array([[0, 1, 2*Cp[1,0,k], 3*Cp[1,0,k]**2, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, Cp[1,0,k], Cp[1,0,k]**2, Cp[1,0,k]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, Cp[1,1,k], Cp[1,1,k]**2, Cp[1,1,k]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 2*Cp[1,1,k], 3*Cp[1,1,k]**2, 0, -1, -2*Cp[1,1,k], -3*Cp[1,1,k]**2,0, 0, 0, 0],
                   [0, 0, 2, 6*Cp[1,1,k], 0, 0, -2, -6*Cp[1,1,k], 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, Cp[1,1,k], Cp[1,1,k]**2, Cp[1,1,k]**3, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, Cp[1,2,k], Cp[1,2,k]**2, Cp[1,2,k]**3, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 2*Cp[1,2,k], 3*Cp[1,2,k]**2, 0, -1, -2*Cp[1,2,k], -3*Cp[1,2,k]**2],
                   [0, 0, 0, 0, 0, 0, 2, 6*Cp[1,2,k], 0, 0, -2, -6*Cp[1,2,k]],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, Cp[1,2,k], Cp[1,2,k]**2, Cp[1,2,k]**3],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, Cp[1,3,k], Cp[1,3,k]**2, Cp[1,3,k]**3],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2*Cp[1,3,k], 3*Cp[1,3,k]**2]],dtype = np.float64)

    # Solving system of equations. Rm = CM*Pcf
    # Pcf[:,k] = inv(CM)@Rm.flatten() # Flatten for converting [12,1] to [12,]
    Pcf = inv(CM)@Rm
    
    # Interpolate between control points to plot poly's
    # E.g. Cp0 - Cp1, Cp1 - Cp2, Cp2 - Cp3
    if sum(Sk[1,:,k]) == 0 :
        y_p0 = np.arange(Cp[1,0,k],  Cp[1,1,k]                                       ,  (1/DV.Pn)*(Cp[1,1,k]- Cp[1,0,k]))
        y_p1 = np.arange(Cp[1,1,k],  Cp[1,2,k]                                       ,  (1/DV.Pn)*(Cp[1,2,k]- Cp[1,1,k]))
        y_p2 = np.arange(Cp[1,2,k],  Cp[1,3,k] + (1/(DV.Pn))*(Cp[1,3,k]- Cp[1,2,k])  ,  (1/DV.Pn)*(Cp[1,3,k]- Cp[1,2,k]))
        # Due to smaller size of secondary branch this is behaving oddly, this condition filters out the additional pooint at the end... 
        if np.size(y_p2) > np.size(y_p1)+1:
            y_p2 = y_p2[0:-1]
        
        polycall_vx_y.y_p0 = y_p0
        polycall_vx_y.y_p1 = y_p1
        polycall_vx_y.y_p2 = y_p2

        # Parse data from function onto tree spatial matrix
        Sk[1,:,k] = np.concatenate([y_p0, y_p1, y_p2])  # z data
    
    elif sum(Sk[1,:,k]) != 0 :
        pass

    # Calculate polynomial coordinates using general equations for 3rd order.
    #  x=                      a                                   + bx                                  +           cx^2                           +           bx^3
    x_p0 = np.tile(Pcf[0,0],(np.size(Sk[1,0:DV.Pn,k])           )) + Pcf[1,0]*Sk[1,0:DV.Pn,k]            +  Pcf[2,0] *Sk[1,0:DV.Pn,k]**2            +  Pcf[3,0] *Sk[1,0:DV.Pn,k] **3
    x_p1 = np.tile(Pcf[4,0],(np.size(Sk[1,DV.Pn:2*DV.Pn,k])     )) + Pcf[5,0]*Sk[1,DV.Pn:2*DV.Pn,k]      +  Pcf[6,0] *Sk[1,DV.Pn:2*DV.Pn,k]**2      +  Pcf[7,0] *Sk[1,DV.Pn:2*DV.Pn,k]**3
    x_p2 = np.tile(Pcf[8,0],(np.size(Sk[1,2*DV.Pn:3*DV.Pn+1,k]) )) + Pcf[9,0]*Sk[1,2*DV.Pn:3*DV.Pn+1,k]  +  Pcf[10,0]*Sk[1,2*DV.Pn:3*DV.Pn+1,k]**2  +  Pcf[11,0]*Sk[1,2*DV.Pn:3*DV.Pn+1,k]**3


    # Parse data from function onto tree spatial matrix
    # Note, flatten needed to convert to 1D to slice onto matrix
    Sk[0,:,k] = np.concatenate([x_p0, x_p1, x_p2])

""" Plotting to see if working as intended
    if k == 10 or k == 20 or k == 30 or k ==40 :
        fig = plt.figure(1)
        fig.add_subplot(111,aspect='equal')
        plt.plot(y_p0,Sk[0,0:Pn,k],'ro',y_p1,Sk[0,Pn:2*Pn,k],'go',y_p2,Sk[0,2*Pn:3*Pn+1,k],'bo')
"""

def polycall_vz_y(ko,Cp,Sk,DV):      
    
    # I really hope this doesnt save! Doesnt! VICTORY 
    # Will probably need an expression to change k = for different branches? 
    # Or just use them all as ko? 
    k = ko 
    
    # Allocate memory for working matrices 
    Rm  = np.zeros([12, 1])
    CM  = np.zeros([12,12])
    Pcf = np.zeros([12,1])

    So = round(rand.uniform(0.1,0.2),2)             # Initial gradient 
    Se = round(rand.uniform(-0.5,0.5),2)            # End gradient

    Rm = np.array([[So],
                   [Cp[2,0,k]],
                   [Cp[2,1,k]],
                   [0],
                   [0],
                   [Cp[2,1,k]],
                   [Cp[2,2,k]],
                   [0],
                   [0],
                   [Cp[2,2,k]],
                   [Cp[2,3,k]],
                   [Se]],dtype = np.float64)
    
    # Connectivity coefficients from general equations of 3rd order poly's with
    # enforced equated gradients and curvature at connectivity points
    CM = np.array([[0, 1, 2*Cp[1,0,k], 3*Cp[1,0,k]**2, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, Cp[1,0,k], Cp[1,0,k]**2, Cp[1,0,k]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, Cp[1,1,k], Cp[1,1,k]**2, Cp[1,1,k]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 2*Cp[1,1,k], 3*Cp[1,1,k]**2, 0, -1, -2*Cp[1,1,k], -3*Cp[1,1,k]**2,0, 0, 0, 0],
                   [0, 0, 2, 6*Cp[1,1,k], 0, 0, -2, -6*Cp[1,1,k], 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, Cp[1,1,k], Cp[1,1,k]**2, Cp[1,1,k]**3, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, Cp[1,2,k], Cp[1,2,k]**2, Cp[1,2,k]**3, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 2*Cp[1,2,k], 3*Cp[1,2,k]**2, 0, -1, -2*Cp[1,2,k], -3*Cp[1,2,k]**2],
                   [0, 0, 0, 0, 0, 0, 2, 6*Cp[1,2,k], 0, 0, -2, -6*Cp[1,2,k]],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, Cp[1,2,k], Cp[1,2,k]**2, Cp[1,2,k]**3],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, Cp[1,3,k], Cp[1,3,k]**2, Cp[1,3,k]**3],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2*Cp[1,3,k], 3*Cp[1,3,k]**2]],dtype = np.float64)

    # Solving system of equations. Rm = CM*Pcf
    # Pcf[:,k] = inv(CM)@Rm.flatten() # Flatten for converting [12,1] to [12,]
    Pcf = inv(CM)@Rm
    
    # Interpolate between control points to plot poly's
    # E.g. Cp0 - Cp1, Cp1 - Cp2, Cp2 - Cp3
    if sum(Sk[1,:,k]) == 0 :
        y_p0 = np.arange(Cp[1,0,k],  Cp[1,1,k]                                       ,  (1/DV.Pn)*(Cp[1,1,k]- Cp[1,0,k]))
        y_p1 = np.arange(Cp[1,1,k],  Cp[1,2,k]                                       ,  (1/DV.Pn)*(Cp[1,2,k]- Cp[1,1,k]))
        y_p2 = np.arange(Cp[1,2,k],  Cp[1,3,k] + (1/(2*DV.Pn))*(Cp[1,3,k]- Cp[1,2,k]),  (1/DV.Pn)*(Cp[1,3,k]- Cp[1,2,k]))
        
        # Parse data from function onto tree spatial matrix
        Sk[1,:,k] = np.concatenate([y_p0, y_p1, y_p2])  # z data
    
    elif sum(Sk[1,:,k]) != 0 :
        pass


    # Calculate polynomial coordinates using general equations for 3rd order.
    #  x=                      a                                   + bx                                  +           cx^2                           +           bx^3
    z_p0 = np.tile(Pcf[0,0],(np.size(Sk[1,0:DV.Pn,k])           )) + Pcf[1,0]*Sk[1,0:DV.Pn,k]            +  Pcf[2,0] *Sk[1,0:DV.Pn,k]**2            +  Pcf[3,0]* Sk[1,0:DV.Pn,k] **3
    z_p1 = np.tile(Pcf[4,0],(np.size(Sk[1,DV.Pn:2*DV.Pn,k])     )) + Pcf[5,0]*Sk[1,DV.Pn:2*DV.Pn,k]      +  Pcf[6,0] *Sk[1,DV.Pn:2*DV.Pn,k]**2      +  Pcf[7,0]* Sk[1,DV.Pn:2*DV.Pn,k]**3
    z_p2 = np.tile(Pcf[8,0],(np.size(Sk[1,2*DV.Pn:3*DV.Pn+1,k]) )) + Pcf[9,0]*Sk[1,2*DV.Pn:3*DV.Pn+1,k]  +  Pcf[10,0]*Sk[1,2*DV.Pn:3*DV.Pn+1,k]**2  +  Pcf[11,0]*Sk[1,2*DV.Pn:3*DV.Pn+1,k]**3


    # Parse data from function onto tree spatial matrix
    # Note, flatten needed to convert to 1D to slice onto matrix
    Sk[2,:,k] = np.concatenate([z_p0, z_p1, z_p2])

""" Plotting to see if working as intended
    if k == 10 or k == 20 or k == 30 or k ==40 :
        fig = plt.figure(1)
        fig.add_subplot(111,aspect='equal')
        plt.plot(y_p0,Sk[0,0:Pn,k],'ro',y_p1,Sk[0,Pn:2*Pn,k],'go',y_p2,Sk[0,2*Pn:3*Pn+1,k],'bo')
"""
