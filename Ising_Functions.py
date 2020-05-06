#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

# Values of quantities used throughout script
L = 32
T=1.0
J=1.0
h=0.0

#Functions

def initial(n):
    '''
    Function that initializes the lattice to be used throughout script
         -INPUT: (1) n: number of sites along one side of the square lattice --> n**2 is total number of sites
         -OUTPUT: returns an (N,N) array array with values of either [-1,+1] randomly assigned to each site
    '''
    return np.random.choice([-1,1], size=(n,n))

def plot(lattice, title):
    '''
    Function that plots the lattice, just used for visualisation purposes
         -INPUT:  (1) lattice: an (N,N) array array with values of [-1,+1] at each site 
                  (2) title: A string to be used as a title of the outputted plot
         -OUTPUT: (1) returns a colourmap plot of lattice site values 
    '''
    fig1 = plt.figure(figsize=(5,5))
    ax1 = fig1.add_subplot(111)
    X,Y = np.meshgrid(range(len(lattice)+1), range(len(lattice)+1))
    plt.title(str(title))
    ax1.pcolormesh(X, Y, lattice, cmap=plt.cm.Dark2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.show()

def sweep(lattice, T, J, h):
    '''
    Function that loops over each site in inputted lattice and performs a Metropolis algorithm flip
         -INPUT:  (1) lattice: an (N,N) array with values of [-1,+1] at each site
                  (2) T: the temperature at which the probablity of a flip accuring is calculated (default of T=1.0 (low))
                  (3) J: coupling constant (default of J=1.0)
                  (4) h: external magnetic field strength (set to default of 0.0)
         -OUTPUT: returns lattice same as inputted lattice but site has been flipped with a probability 
    '''
    n = lattice.shape[0]
    for i in range(n):
        for j in range(n):            
            p=i
            q=j
            s =  lattice[p, q]
            neigh = lattice[(p+1)%n,q] + lattice[(p-1),q] + lattice[p,(q+1)%n] + lattice[p,(q-1)]
            deltaE = 2*J*s*neigh + 2*h*s
            if deltaE <= 0.:
                s *= -1
            elif np.random.rand() < np.exp(-deltaE*(1/(T))):
                s *= -1
            lattice[p, q] = s
    return lattice

def sweeper(lattice, n_iter):
    '''
    Function that implements the function [sweep] a specified number of times recursively on the same lattice
         -INPUT:  (1) lattice: an (N,N) array with values of [-1,+1] at each site
                  (2) n_iter: number of sweeps to perform on the lattice 
         -OUTPUT: returns lattice same as inputted lattice but with n_iter number of sweeps performed
    '''       
    for i in range(n_iter):
        sweep(lattice)  
    return lattice

def Erg(lattice):
    '''
    Function that calculates the energy per site (intensive) of given lattice
         -INPUT:  (1) lattice: an (N,N) array with values of [-1,+1] at each site
         -OUTPUT: returns scalar value for average energy per site
    '''       
    x = lattice.shape[0]
    E = 0
    for p in range(x):
        for q in range(x):            
            S = lattice[p,q]
            neigh = lattice[(p+1)%x, q] + lattice[(p-1), q] + lattice[p,(q+1)%x] + lattice[p,(q-1)]
            E += -neigh*S
    return E/float(lattice.shape[0]**2)

def Mag(lattice):
    '''
    Function that calculates the magnetization per site (intensive) of given lattice
         -INPUT:  (1) lattice: an (N,N) array with values of [-1,+1] at each site
         -OUTPUT: returns scalar value for average magnetization per site
    '''
    lattice = lattice.reshape(L,L)
    magnetization = np.sum(lattice)
    return magnetization/float(L**2)
