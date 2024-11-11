# -*- coding: utf-8 -*-
"""
Created on 15 October 2023

@author: divyoj
"""
## importing libraries:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import os
# # note that this must be executed before 'import numba'
# os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'
from numba import njit
import time as process_time
import plotting
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mp
import warnings

sns.set_theme(context='notebook', style='ticks',font='arial',font_scale=1.2)
cm = 1/2.54  # centimeters in inches
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
def fixed_step_solver(t_span,ic_aL,ic_bL,ic_aR,ic_bR,aT,bT,noise,alpha, beta ):

    time=np.arange(dt,T_max+dt,dt)
    aL,bL,aR,bR = 1*ic_aL,1*ic_bL,1*ic_aR,1*ic_bR
    
    
    aL_T = np.zeros((len(t_span)+1,ny,nx)); aR_T = np.zeros((len(t_span)+1,ny,nx));
    bL_T = np.zeros((len(t_span)+1,ny,nx)); bR_T = np.zeros((len(t_span)+1,ny,nx));
    index=0;
    aL_T[index]=aL;bL_T[index]=bL; aR_T[index]=aR;bR_T[index]=bR;
    
    
    noise_array=noise*np.random.normal(loc=0.0, scale=1,size=(len(time),4,ny,nx))

    for ti, t in enumerate(time):
        
        daL,dbL,daR,dbR = do_timestep(aL,bL,aR,bR,aT,bT,alpha, beta, )
        aL+= daL*dt+ dt*noise_array[ti,0,:,:];  bL+= dbL*dt+ dt*noise_array[ti,1,:,:];
        aR+= daR*dt+ dt*noise_array[ti,2,:,:];  bR+= dbR*dt+ dt*noise_array[ti,3,:,:];
        
        if (ti%resolution)==1:
            index+=1
            aL_T[index]=aL;bL_T[index]=bL;
            aR_T[index]=aR;bR_T[index]=bR;
            
            
    return(aL_T,bL_T,aR_T,bR_T)

def simulate(t_span,rho,noise,alpha, beta ):
    ''' function to iterate over time and return arrays with the result '''
    
    #print('solving odes ..')
    
    # total proteins in the cells
    aT = rho + np.zeros((ny,nx))
    bT = rho + np.zeros((ny,nx))
  
    ## initializing aL,bR,bL,aR
    ic_aL = np.zeros((ny,nx)) + 0.10*aT;  ic_bL = np.zeros((ny,nx)) + 0.10001*bT;
    ic_aR = np.zeros((ny,nx)) + 0.10001*aT;  ic_bR = np.zeros((ny,nx)) + 0.10*bT;
       
    aL_t,bL_t,aR_t,bR_t = fixed_step_solver(t_span,ic_aL,ic_bL,ic_aR,ic_bR,aT,bT,noise,alpha, beta )
      
    return ( aL_t,bL_t,aR_t,bR_t)

if __name__ == "__main__":
    t_initial= process_time.time()
    # Lattice:
    w,h = 500,40;
    dx,dy=1,1;
    nx=int(w/dx)
    ny=1#1;

    # time:
    T_max=500;    
    dt=0.01
    resolution=100;
    
    t_span=np.arange(dt*resolution,T_max+dt*resolution,dt*resolution)
    # parameters:
    alpha=1;
    beta=1;
    noise=0.0
    epsilon=0;
    no_of_instances=50
        
    #%% delta p vs noise for different values of rho above the critical point:
    
    main_folder="./"
    
    f, axs = plt.subplots(figsize=(12*cm,8*cm))  
    for rho in [2.6,2.8]:
    
        noise_amp_array = np.unique(np.sort(np.around(np.concatenate((np.logspace(-7,-1,num=10),np.logspace(-5.5,-3,num=10))),25)));
        
        delta_p_array=np.empty((len(noise_amp_array),no_of_instances))

        for ri, noise in tqdm(enumerate(noise_amp_array)):
            
            for instance in range(0,no_of_instances,1):
                aL_t, aR_t, bL_t, bR_t =  simulate(t_span,rho,noise,alpha, beta )
                delta_p_array[ri,instance]=abs(plotting.calculate_delta_p(aL_t,bL_t,aR_t,bR_t))
                
        data_file = main_folder + "epsilon=" + str(epsilon) + "_alpha=" + str(alpha) + "_rho=" + str(rho)
        np.savez(data_file, noise_amp_array=noise_amp_array, delta_p_array=delta_p_array)

        axs.plot(noise_amp_array,np.mean(delta_p_array,axis=1),'.-',label=str(rho));
        axs.fill_between(noise_amp_array,np.mean(delta_p_array,axis=1)-np.std(delta_p_array,axis=1),np.mean(delta_p_array,axis=1)+np.std(delta_p_array,axis=1),alpha=0.2)
        axs.set_ylabel("Polarization"+r'$(p)$')
        axs.set_xlabel("Noise amplitude" +r"$(\eta)$")
        axs.set_xscale('log')
        
        axs.legend(title=r'$\rho$')
        
    #f.suptitle(r"zeta="+str(zeta))
    f.subplots_adjust(top=0.95, bottom=0.20, left=0.20, right=0.95, hspace=0.75,wspace=0.75)

    f.savefig(main_folder+"Noise_in_protein_kinetics"+".png",dpi=800)    
    plt.close()


