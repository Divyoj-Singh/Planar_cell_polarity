# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 14:23:45 2021

@author: csb
"""


## importing libraries:
import numpy as np
import matplotlib.pyplot as plt
import os
from numba import njit
# import plotting
from scipy.integrate import solve_ivp
import matplotlib.lines as mlines
from collections import OrderedDict
import seaborn as sns
import time as process_time
from num2tex import num2tex as n2t
import matplotlib.cm as cm
import matplotlib as mpl

from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
font = {'family' : 'normal','weight' : 'bold'} 
rc('text', usetex=True)

figx = 6.0
figy = 5.5
lw = 2.0
legendFontSize=24
tickFontSize=20
labelFontSize=16
panelLabelFontSize=24
alpha0 = 0.05
start=0;
stop=0;
step=1
@njit
def do_timestep(aL,bL,aR,bR,aT,bT,alpha, beta):
    '''function to give dxdt at a time step '''

    # total membrane protein concentration:
    a0 = aL + aR
    b0 = bL + bR
    # intitating dxdt to arrays of zeros:
    daL=np.zeros((ny,nx));daR=np.zeros((ny,nx));dbL=np.zeros((ny,nx));dbR=np.zeros((ny,nx));
    
    ## Equations for al,aR,bl and bR:
    daL=alpha*(aT-a0)- beta*aL/(1+np.roll(bR,+1)**2);
    # aR
    daR=alpha*(aT-a0)- beta*aR/(1+np.roll(bL,-1)**2) ;
    # bL
    dbL=alpha*(bT-b0)-beta*bL/(1+np.roll(aR,+1)**2) ;
    # bR
    dbR=alpha*(bT-b0)-beta*bR/(1+np.roll(aL,-1)**2) ;
            
            
    return  (daL,dbL,daR,dbR)

#@njit
def fixed_step_solver(t_span,ic_aL,ic_bL,ic_aR,ic_bR,aT,bT,noise,alpha, beta):

    time=np.arange(dt,T_max+dt,dt)
    aL,bL,aR,bR = 1*ic_aL,1*ic_bL,1*ic_aR,1*ic_bR
    
    
    aL_T = np.zeros((len(t_span)+1,ny,nx)); aR_T = np.zeros((len(t_span)+1,ny,nx));
    bL_T = np.zeros((len(t_span)+1,ny,nx)); bR_T = np.zeros((len(t_span)+1,ny,nx));
    index=0;
    aL_T[index]=aL;bL_T[index]=bL; aR_T[index]=aR;bR_T[index]=bR;
    
    
#    noise_array=noise*np.random.normal(loc=0.0, scale=1,size=(len(time),4,ny,nx))

    for ti, t in enumerate(time):
        
        daL,dbL,daR,dbR = do_timestep(aL,bL,aR,bR,aT,bT,alpha, beta)
        aL+= daL*dt;  bL+= dbL*dt;
        aR+= daR*dt;  bR+= dbR*dt;
        
        if (ti%resolution)==1:
            index+=1
            aL_T[index]=aL;bL_T[index]=bL;
            aR_T[index]=aR;bR_T[index]=bR;
            
            
    return(aL_T,bL_T,aR_T,bR_T)

def simulate(t_span,rho,noise,alpha, beta):
    ''' function to iterate over time and return arrays with the result '''
    
    print('solving odes ..')
    
    # total proteins in the cells
    noise_array=noise*np.random.normal(loc=0.0, scale=1,size=(2,ny,nx))
    # total proteins in the cells
    aT = rho*(1+noise_array[0,:,:])
    bT = rho*(1+noise_array[1,:,:])
  
    ## initializing aL,bR,bL,aR
    ic_aL = np.zeros((ny,nx)) + 0.10*rho;  ic_bL = np.zeros((ny,nx)) + 0.10001*rho;
    ic_aR = np.zeros((ny,nx)) + 0.10001*rho;  ic_bR = np.zeros((ny,nx)) + 0.10*rho;
       
    aL_t,bL_t,aR_t,bR_t = fixed_step_solver(t_span,ic_aL,ic_bL,ic_aR,ic_bR,aT,bT,noise,alpha, beta)
      
    return ( aL_t,bL_t,aR_t,bR_t)

if __name__ == "__main__":
    t_initial= process_time.time()
    # Lattice:
    w,h = 500,1;
    dx,dy=1,1;
    nx=int(w/dx)
    ny=1#1;

    # time:
    T_max=500;    
    dt=0.01
    resolution=100;
    
    t_span=np.arange(dt*resolution,T_max+dt*resolution,dt*resolution)
    # parameters:
    alpha=10;
    beta=1;
    noise=0.0
    epsilon=0;
    no_of_instances=50 
        
    #%% Dynamics:
    main_folder="./"
    
    for rho in [2.8]:
        folder= main_folder+ "rho=" +str(rho)+"/"
        if not os.path.exists(folder):
            os.makedirs(folder)
    
        noise_array = np.array([1e-5,1e-4,1e-3,1e-2]);
        n_array=["$10^{-5}$","$10^{-4}$","$10^{-3}$","$10^{-2}$"]
        rho0_array=noise_array*0;p_array=noise_array*0;delta_p_array=noise_array*0
        subplots=["B","C","D","E"]

    
        for noisei, noise in enumerate(noise_array):
            print (noisei, noise)
            aL_t, aR_t, bL_t, bR_t = simulate(t_span,rho,noise,alpha, beta)
            
            aL=aL_t[:,:,start:nx-stop+step:step];aR=aR_t[:,:,start:nx-stop+step:step]
            bL=bL_t[:,:,start:nx-stop+step:step];bR=bR_t[:,:,start:nx-stop+step:step]
            ## Calculating observables:
            pa = aR - aL; pb = bR - bL
            a0 = aR + aL; b0 = bR + bL
            average_p=pa+pb;delta_p=pa-pb;
            #%% Plotting:
            f=plt.figure(figsize=(5,5))
            ## delta p
            plt.plot(delta_p[-1,0,:]);
            plt.title("Noise Amplitude "+r"$(S) = $" + n_array[noisei],fontsize=labelFontSize)
            plt.ylabel('Polarity of each cell '+r"$(p(i))$",fontsize=labelFontSize)
            plt.xlabel("Cell index (i)",fontsize=labelFontSize)
            plt.ylim([-4 ,4])
            plt.text(-100,3.42,subplots[noisei],weight="bold",fontsize=panelLabelFontSize+4)
            plt.tight_layout()
    
            plt.savefig(folder+"rho="+str(rho)+"_noise="+str(noise)+".png")    
            plt.close()
        
            