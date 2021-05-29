# -*- coding: utf-8 -*-
"""
Created on Sun May 16 00:21:14 2021

@author: divyoj
"""


## importing libraries:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import os
from numba import njit
import time as process_time
import plotting
from scipy.integrate import solve_ivp

## functions:
@njit
def do_timestep(t,z,aT,bT,alpha, beta, gamma, zeta):
    ''' function to give dxdt at a time step '''
    
    aL = z[0*(nx*ny):1*(nx*ny)].reshape((ny,nx))
    bL = z[1*(nx*ny):2*(nx*ny)].reshape((ny,nx))
    aR = z[2*(nx*ny):3*(nx*ny)].reshape((ny,nx))
    bR = z[3*(nx*ny):4*(nx*ny)].reshape((ny,nx))
    
    # total membrane protein concentration:
    a0 = aL + aR
    b0 = bL + bR
    # intitating dxdt to arrays of zeros:
    daL=np.zeros((ny,nx));daR=np.zeros((ny,nx));dbL=np.zeros((ny,nx));dbR=np.zeros((ny,nx));
    ## Equations for al,aR,bl and bR:
    # aL 
    daL[0,1:nx-1]=alpha*(aT[0,1:nx-1]-a0[0,1:nx-1])-beta*aL[0,1:nx-1]+beta*gamma*( aL[0,1:nx-1]*bR[0,1-1:nx-1-1] ) -zeta*(aL[0,1:nx-1]-aR[0,1:nx-1])**3;
    # aR
    daR[0,1:nx-1]=alpha*(aT[0,1:nx-1]-a0[0,1:nx-1])-beta*aR[0,1:nx-1]+beta*gamma*( aR[0,1:nx-1]*bL[0,1+1:nx-1+1] ) -zeta*(aR[0,1:nx-1]-aL[0,1:nx-1])**3;
    # bL
    dbL[0,1:nx-1]=alpha*(bT[0,1:nx-1]-b0[0,1:nx-1])-beta*bL[0,1:nx-1]+beta*gamma*( bL[0,1:nx-1]*aR[0,1-1:nx-1-1] ) -zeta*(bL[0,1:nx-1]-bR[0,1:nx-1])**3;
    # bR
    dbR[0,1:nx-1]=alpha*(bT[0,1:nx-1]-b0[0,1:nx-1])-beta*bR[0,1:nx-1]+beta*gamma*( bR[0,1:nx-1]*aL[0,1+1:nx-1+1] ) -zeta*(bR[0,1:nx-1]-bL[0,1:nx-1])**3;

    # Boundary Conditions:
    # aL 
    daL[0,1:nx-1]=alpha*(aT[0,1:nx-1]-a0[0,1:nx-1])-beta*aL[0,1:nx-1]+beta*gamma*( aL[0,1:nx-1]*bR[0,1-1:nx-1-1] ) -zeta*(aL[0,1:nx-1]-aR[0,1:nx-1])**3;
    # aR
    daR[0,1:nx-1]=alpha*(aT[0,1:nx-1]-a0[0,1:nx-1])-beta*aR[0,1:nx-1]+beta*gamma*( aR[0,1:nx-1]*bL[0,1+1:nx-1+1] ) -zeta*(aR[0,1:nx-1]-aL[0,1:nx-1])**3;
    # bL
    dbL[0,1:nx-1]=alpha*(bT[0,1:nx-1]-b0[0,1:nx-1])-beta*bL[0,1:nx-1]+beta*gamma*( bL[0,1:nx-1]*aR[0,1-1:nx-1-1] ) -zeta*(bL[0,1:nx-1]-bR[0,1:nx-1])**3;
    # bR
    dbR[0,1:nx-1]=alpha*(bT[0,1:nx-1]-b0[0,1:nx-1])-beta*bR[0,1:nx-1]+beta*gamma*( bR[0,1:nx-1]*aL[0,1+1:nx-1+1] ) -zeta*(bR[0,1:nx-1]-bL[0,1:nx-1])**3;

    # Boundary Conditions:
    #aL
    daL[0,0]=daL[0,1];
    daL[0,nx-1]=alpha*(aT[0,nx-1]-a0[0,nx-1])-beta*aL[0,nx-1]+beta*gamma*(aL[0,nx-1]*bR[0,nx-1-1])-zeta*(aL[0,nx-1]-aR[0,nx-1])**3;
    
    #aR
    daR[0,0]=alpha*(aT[0,0]-a0[0,0])-beta*aR[0,0]+beta*gamma*( aR[0,0]*bL[0,1] ) -zeta*(aR[0,0]-aL[0,0])**3;
    daR[0,nx-1]=daR[0,nx-2];
    
    #bL
    dbL[0,0]=dbL[0,1];
    dbL[0,nx-1]=alpha*(bT[0,nx-1]-b0[0,nx-1])-beta*bL[0,nx-1]+beta*gamma*(bL[0,nx-1]*aR[0,nx-1-1])-zeta*(bL[0,nx-1]-bR[0,nx-1])**3;
    
    #bR
    dbR[0,0]=alpha*(bT[0,0]-b0[0,0])-beta*bR[0,0]+beta*gamma*( bR[0,0]*aL[0,1] ) -zeta*(bR[0,0]-bL[0,0])**3;
    dbR[0,nx-1]=dbR[0,nx-2];
    
    
    daL=daL*((aT>=a0) | (daL<0));  daR=daR*((aT>=a0) | (daR<0))
    dbL=dbL*((bT>=b0) | (dbL<0));  dbR=dbR*((bT>=b0) | (dbR<0))
    
    daL=daL*((aL>=0) | (daL>0));       daR=daR*((aR>=0) | (daR>0))
    dbL=dbL*((bL>=0) | (dbL>0));       dbR=dbR*((bR>=0) | (dbR>0))

    return np.concatenate((daL.flatten(),dbL.flatten(),daR.flatten(),dbR.flatten()))

def solver(ic,aT,bT,alpha, beta, gamma, zeta):
    
    ## initilizing the arrays to store the values over time:
    aL_t = np.zeros((T_max+1,ny,nx)); aR_t = np.zeros((T_max+1,ny,nx));
    bL_t = np.zeros((T_max+1,ny,nx)); bR_t = np.zeros((T_max+1,ny,nx));
    
    ## Solving the initial value problem:
    sol = solve_ivp(lambda t,y: do_timestep(t,y,aT,bT,alpha, beta, gamma, zeta),t_span=[0,T_max],y0=ic,t_eval=list(np.linspace(0,T_max,T_max+1)))

    t = sol.t
    aball = sol.y
    for t_index, ts in enumerate(t):
        aball_at_ts = aball[:,t_index]
    
        aL_t[t_index]= aball_at_ts[0*(nx*ny):1*(nx*ny)].reshape((ny,nx));
        bL_t[t_index]= aball_at_ts[1*(nx*ny):2*(nx*ny)].reshape((ny,nx));
        aR_t[t_index]= aball_at_ts[2*(nx*ny):3*(nx*ny)].reshape((ny,nx));
        bR_t[t_index]= aball_at_ts[3*(nx*ny):4*(nx*ny)].reshape((ny,nx));
    
    return (aL_t,bL_t,aR_t,bR_t,t)


#@njit
def simulate(rho,epsilon_array,alpha, beta, gamma, zeta):
    ''' function to iterate over time and return arrays with the result '''
    ## initilizing the arrays to store the values over time:
    bT = rho + np.zeros((ny,nx))
    rho0_array=1*epsilon_array;pa=1*epsilon_array;pb=1*epsilon_array
    
    for index, epsilon in enumerate(epsilon_array):
        
        ## for the first instance:
        if index==0:
            print('starting with negative',index,epsilon)
            # total proteins in the cells
            aT = rho + np.zeros((ny,nx))+rho*epsilon*np.linspace(-0.5,0.5,nx)*np.ones((ny,nx))
            bT = rho + np.zeros((ny,nx))
            ## initializing aL,bR,bL,aR
            
            aL = np.zeros((ny,nx)) + 0.1*rho; aR = np.zeros((ny,nx)) + 0.100001*rho
            bL = np.zeros((ny,nx)) + 0.100001*rho; bR = np.zeros((ny,nx)) + 0.1*rho
            
            
            ## Collecting the initial conditions into a single array:
            ic = np.array(aL.flatten().tolist()+bL.flatten().tolist()+aR.flatten().tolist()+bR.flatten().tolist())
            # Solving the odes for the first time:
            aL_t,bL_t,aR_t,bR_t,t = solver(ic,aT,bT,alpha, beta, gamma, zeta)
            
            # Plotting at each rho
            rho0_array[index],pa[index],pb[index]=plotting.plots_at_epsilon(aL_t,bL_t,aR_t,bR_t,epsilon,folder)
            
            ## Adding the data:
            aL_T=1*aL_t;aR_T=1*aR_t;
            bL_T=1*bL_t;bR_T=1*bR_t;
            T=t*1;
        
        elif index==no_of_epsilon:
            print('now returning back',index,epsilon)
            # total proteins in the cells
            aT = rho + np.zeros((ny,nx))+rho*epsilon*np.linspace(-0.5,0.5,nx)*np.ones((ny,nx))
            bT = rho + np.zeros((ny,nx))
            ## initializing aL,bR,bL,aR
            
            aL = np.zeros((ny,nx)) + 0.100001*rho; aR = np.zeros((ny,nx)) + 0.1*rho
            bL = np.zeros((ny,nx)) + 0.1*rho; bR = np.zeros((ny,nx)) + 0.100001*rho
            
            ## Collecting the initial conditions into a single array:
            ic = np.array(aL.flatten().tolist()+bL.flatten().tolist()+aR.flatten().tolist()+bR.flatten().tolist())
            # Solving the odes for the first time:
            aL_t,bL_t,aR_t,bR_t,t = solver(ic,aT,bT,alpha, beta, gamma, zeta)
            t=t+T_max*index
            # Plotting at each rho
            rho0_array[index],pa[index],pb[index]=plotting.plots_at_epsilon(aL_t,bL_t,aR_t,bR_t,epsilon,folder)
            
            ## Adding the data:
            aL_T=np.concatenate((aL_T,aL_t),axis=0); aR_T=np.concatenate((aR_T,aR_t),axis=0);
            bL_T=np.concatenate((bL_T,bL_t),axis=0); bR_T=np.concatenate((bR_T,bR_t),axis=0);
            T=np.concatenate((T,t),axis=0)
        
            
        else:
            print('other cases',index,epsilon)
            # total proteins of a in the cell changes 
            aT = rho + np.zeros((ny,nx))+rho*epsilon*np.linspace(-0.5,0.5,nx)*np.ones((ny,nx))
            
            ## Initial values of a and b are set as steady state value of previous epsilon
            aL=aL_t[-1,:,:]*1;aR=aR_t[-1,:,:]*1;
            bL=bL_t[-1,:,:]*1;bR=bR_t[-1,:,:]*1;
            
            aL_t,bL_t,aR_t,bR_t,t = solver(ic,aT,bT,alpha, beta, gamma, zeta)
            t=t+T_max*index
            # Plotting at each rho
            rho0_array[index],pa[index],pb[index]=plotting.plots_at_epsilon(aL_t,bL_t,aR_t,bR_t,epsilon,folder)
            
            aL_T=np.concatenate((aL_T,aL_t),axis=0); aR_T=np.concatenate((aR_T,aR_t),axis=0);
            bL_T=np.concatenate((bL_T,bL_t),axis=0); bR_T=np.concatenate((bR_T,bR_t),axis=0);
            T=np.concatenate((T,t),axis=0)
        
        print(pa[index],pb[index])    
    
    ## Plotting:
    f, axs = plt.subplots(3,1,figsize=(12,9))     
    ## rho0 vs rho
    axs[0].plot(epsilon_array[:no_of_epsilon],rho0_array[:no_of_epsilon],'b.-',label=r'increasing $\epsilon $');
    axs[0].plot(epsilon_array[no_of_epsilon:],rho0_array[no_of_epsilon:],'r.-',label=r'decreasing $\epsilon $');
    axs[0].set_title(r"$\rho_{0} \ v/s \  \epsilon$");
    axs[0].set_ylabel(r"$\rho_{0}$")
    axs[0].set_xlabel(r"$\epsilon$")
    axs[0].legend()        
    ## rho0 vs rho
    axs[1].plot(epsilon_array[:no_of_epsilon],pa[:no_of_epsilon],'b.-',label=r'increasing $\epsilon $');
    axs[1].plot(epsilon_array[no_of_epsilon:],pa[no_of_epsilon:],'r.-',label=r'decreasing $\epsilon $');
    axs[1].set_title(r'$p_{a}\ v/s \ \epsilon$');
    axs[1].set_ylabel(r"$p_{a} $")
    axs[1].set_xlabel(r"$\epsilon$")
    axs[1].legend()
    ## rho0 vs rho
    axs[2].plot(epsilon_array[:no_of_epsilon],pb[:no_of_epsilon],'b.-',label=r'increasing $\epsilon $');
    axs[2].plot(epsilon_array[no_of_epsilon:],pb[no_of_epsilon:],'r.-',label=r'decreasing $\epsilon $');
    axs[2].set_title(r'$p_{b} \ v/s \ \epsilon $');
    axs[2].set_ylabel(r"$p_{b}  $")
    axs[2].set_xlabel(r"$\epsilon$")
    axs[2].legend()
    
    f.suptitle(r"$\rho=$"+str(rho))
    f.subplots_adjust(top=0.85, bottom=0.20, left=0.20, right=0.95, hspace=0.50,wspace=0.50)
    f.savefig(folder+"observables_wrt_epsilon"+".png",dpi=500)    
    plt.close()

    ## Plotting the differences:
    f, axs = plt.subplots(3,1,figsize=(12,9))     
    ## rho0 vs rho
    axs[0].plot(epsilon_array[:no_of_epsilon],rho0_array[:no_of_epsilon]-np.flip(rho0_array[no_of_epsilon:]),'b.-');
    axs[0].set_title(r"$\Delta \rho_{0} \ v/s \  \epsilon$");
    axs[0].set_ylabel(r"$\Delta \rho_{0}$")
    axs[0].set_xlabel(r"$\epsilon$")
         
    ## rho0 vs rho
    axs[1].plot(epsilon_array[:no_of_epsilon],pa[:no_of_epsilon]-np.flip(pa[no_of_epsilon:]),'b.-');
    axs[1].set_title(r'$\Delta p_{a}\ v/s \ \epsilon$');
    axs[1].set_ylabel(r"$\Delta p_{a} $")
    axs[1].set_xlabel(r"$\epsilon$")
    
    ## rho0 vs rho
    axs[2].plot(epsilon_array[:no_of_epsilon],pb[:no_of_epsilon]-np.flip(pb[no_of_epsilon:]),'b.-');
    axs[2].set_title(r'$\Delta p_{b} \ v/s \ \epsilon $');
    axs[2].set_ylabel(r"$\Delta p_{b}  $")
    axs[2].set_xlabel(r"$\epsilon$")
    
    f.suptitle(r"$\rho=$"+str(rho))
    f.subplots_adjust(top=0.85, bottom=0.20, left=0.20, right=0.95, hspace=0.50,wspace=0.50)
    f.savefig(folder+"difference in observables"+".png",dpi=500)    
    plt.close()    
    
    return (aL_T,bL_T,aR_T,bR_T,T)

if __name__ == "__main__":
    # Lattice:
    w,h = 50,2;
    dx,dy=0.1,1;
    nx=int(w/dx)
    ny=1;#int(h/dx);
    # time:
    T_max=500;
    
    # parameters:
    alpha=10; 
    gamma=1
    beta=1;
    zeta=0.01;
    epsilon=0.01;
    no_of_epsilon=51
    
    #%% Characterisation over epsilon for multiple large values of rho::

    rho_array=[0.9,1.1,1.2,1.5]#0.9,0.8,1.0,
    f, axs = plt.subplots(3,1,figsize=(4,9))        
        
    for rhoi, rho in enumerate(rho_array):
        print("rho=",rho)
        
        #folder for storing the data:
        folder="./new/"+"zeta="+str(zeta)+"_alpha="+str(alpha)+"_rho="+str(rho)+"/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        epsilons = np.around(np.concatenate((np.linspace(-1,1,no_of_epsilon),np.linspace(1,-1,no_of_epsilon))),3)
        
        aL_T,bL_T,aR_T,bR_T,T = simulate(rho,epsilons,alpha, beta, gamma, zeta)
        
        
        plotting.plots_over_time(aL_T,bL_T,aR_T,bR_T,T,rho,folder)