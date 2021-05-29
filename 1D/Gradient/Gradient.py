# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:41:38 2021

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
import plotting_gradient
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
            
    #return np.array(daL.flatten().tolist()+dbL.flatten().tolist()+daR.flatten().tolist()+dbR.flatten().tolist())
    return np.concatenate((daL.flatten(),dbL.flatten(),daR.flatten(),dbR.flatten()))

#@njit
def simulate(rho,epsilon,alpha, beta, gamma, zeta):
    ''' function to iterate over time and return arrays with the result '''
    ## initilizing the arrays to store the values over time:
    aL_t = np.zeros((T_max+1,ny,nx)); aR_t = np.zeros((T_max+1,ny,nx));
    bL_t = np.zeros((T_max+1,ny,nx)); bR_t = np.zeros((T_max+1,ny,nx));
    
    # total proteins in the cells
    aT = rho + np.zeros((ny,nx))+epsilon*rho*np.linspace(-0.5,0.5,nx)*np.ones((ny,nx))
    bT = rho + np.zeros((ny,nx))
    
    ## initializing aL,bR,bL,aR
    
    aL = np.zeros((ny,nx)) + 0.1*rho; aR = np.zeros((ny,nx)) + 0.100001*rho
    bL = np.zeros((ny,nx)) + 0.100001*rho; bR = np.zeros((ny,nx)) + 0.1*rho
    
    
    ## Collecting the initial conditions into a single array:
    ic = np.array(aL.flatten().tolist()+bL.flatten().tolist()+aR.flatten().tolist()+bR.flatten().tolist())
    
    ## Solving the initial value problem:
    sol = solve_ivp(lambda t,y: do_timestep(t,y,aT,bT,alpha, beta, gamma, zeta),t_span=[0,T_max],y0=ic,t_eval=list(np.linspace(0,T_max,T_max+1)))

    t=sol.t
    aball=sol.y
    for t_index, ts in enumerate(t):
      
        aball_at_ts = aball[:,t_index]
    
        aL_t[t_index]= aball_at_ts[0*(nx*ny):1*(nx*ny)].reshape((ny,nx));
        bL_t[t_index]= aball_at_ts[1*(nx*ny):2*(nx*ny)].reshape((ny,nx));
        aR_t[t_index]= aball_at_ts[2*(nx*ny):3*(nx*ny)].reshape((ny,nx));
        bR_t[t_index]= aball_at_ts[3*(nx*ny):4*(nx*ny)].reshape((ny,nx));

    
    #return (aL_t[:,:,10:nx-10],aR_t[:,:,10:nx-10],bL_t[:,:,10:nx-10],bR_t[:,:,10:nx-10])
    return (aL_t,bL_t,aR_t,bR_t)



if __name__ == "__main__":
    # Lattice:
    w,h = 10,2;
    dx,dy=0.01,1;
    nx=int(w/dx)
    ny=1;#int(h/dx);
    # time:
    T_max=500;
    
    # parameters:
    alpha=10; 
    gamma=1    ;beta=1;
    zeta=0.0;
    #epsilon=0.1;
    main_folder="./aR greater than aL/"
    # #%% Characterisation over epsilon for multiple small values of rho:

    rho_array=[0.2,0.1]
    f, axs = plt.subplots(3,1,figsize=(4,9))        
        
    for rhoi, rho in enumerate(rho_array):
        print("rho=",rho)
        #folder for storing the data:
        folder=main_folder+"zeta="+str(zeta)+"_alpha="+str(alpha)+"_rho="+str(rho)+"/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        epsilons = np.around(np.linspace(-1,1,21),5);rho0_array=epsilons.copy();
        pa=epsilons.copy();pb=epsilons.copy()

        for ri, epsilon in enumerate(epsilons):
            print (ri, epsilon)
 
            aL_t, bL_t, aR_t, bR_t = simulate(rho,epsilon,alpha, beta, gamma, zeta)
        
        
            # Plotting at each rho
            rho0_array[ri],pa[ri],pb[ri]=plotting_gradient.plots_at_rho(aL_t,bL_t,aR_t,bR_t,epsilon,folder)
        
        ## rho0 vs rho
        axs[0].plot(epsilons,rho0_array,'.-',label=str(rho));
        axs[0].set_title(r"$\rho_{0} \ v/s \  \epsilon$");
        axs[0].set_ylabel(r"$\rho_{0}$")
        axs[0].set_xlabel(r"$\epsilon$")
        axs[0].legend(ncol=2)        
        ## rho0 vs rho
        axs[1].plot(epsilons,pa,'.-',label=str(rho));
        axs[1].set_title(r'$p_{a}\ v/s \ \epsilon$');
        axs[1].set_ylabel(r"$p_{a}$")
        axs[1].set_xlabel(r"$\epsilon$")
        axs[1].legend(ncol=2)
        ## rho0 vs rho
        axs[2].plot(epsilons,pb,'.-',label=str(rho));
        axs[2].set_title(r'$p_{b} \ v/s \ \epsilon $');
        axs[2].set_ylabel(r"$p_{b} $")
        axs[2].set_xlabel(r"$\epsilon$")
        axs[2].legend(ncol=2)
        
    f.suptitle(r"zeta="+str(zeta))
    f.subplots_adjust(top=0.85, bottom=0.20, left=0.20, right=0.95, hspace=0.50,wspace=0.50)
    
    f.savefig(main_folder+"Gradient_over_epsilon_low_rho_zeta="+str(zeta)+".png",dpi=500)    
    plt.close()

    #%% Characterisation over epsilon for multiple large values of rho::
    
    rho_array=[0.9,1.0,1.1,1.2]
    f, axs = plt.subplots(3,1,figsize=(4,9))        
        
    for rhoi, rho in enumerate(rho_array):
        print("rho=",rho)
        #folder for storing the data:
        folder=main_folder+"zeta="+str(zeta)+"_alpha="+str(alpha)+"_rho="+str(rho)+"/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        epsilons = np.sort(np.around(np.concatenate((np.linspace(-1,1,51),np.linspace(-0.1,0.1,21))),5));
        rho0_array=epsilons.copy();
        pa=epsilons.copy();pb=epsilons.copy()

        for ri, epsilon in enumerate(epsilons):
            print (ri, epsilon)
 
            aL_t, bL_t, aR_t, bR_t = simulate(rho,epsilon,alpha, beta, gamma, zeta)
        
        
            # Plotting at each rho
            rho0_array[ri],pa[ri],pb[ri]=plotting_gradient.plots_at_rho(aL_t,bL_t,aR_t,bR_t,epsilon,folder)
        
        ## rho0 vs rho
        axs[0].plot(epsilons,rho0_array,'.-',label=str(rho));
        axs[0].set_title(r"$\rho_{0} \ v/s \  \epsilon$");
        axs[0].set_ylabel(r"$\rho_{0}$")
        axs[0].set_xlabel(r"$\epsilon$")
        axs[0].legend(ncol=2)        
        ## rho0 vs rho
        axs[1].plot(epsilons,pa,'.-',label=str(rho));
        axs[1].set_title(r'$p_{a}\ v/s \ \epsilon$');
        axs[1].set_ylabel(r"$p_{a}$")
        axs[1].set_xlabel(r"$\epsilon$")
        axs[1].legend(ncol=2)
        ## rho0 vs rho
        axs[2].plot(epsilons,pb,'.-',label=str(rho));
        axs[2].set_title(r'$p_{b} \ v/s \ \epsilon $');
        axs[2].set_ylabel(r"$p_{b} $")
        axs[2].set_xlabel(r"$\epsilon$")
        axs[2].legend(ncol=2)
        
    f.suptitle(r"zeta="+str(zeta))
    f.subplots_adjust(top=0.85, bottom=0.20, left=0.20, right=0.95, hspace=0.50,wspace=0.50)
    
    f.savefig(main_folder+"Gradient_over_epsilon_fine_high_rho_zeta="+str(zeta)+".png",dpi=500)    
    plt.close()
    
    # #%% Characterisation over rho:
    epsilon_array=[0.5,0.1,0.01,0]
    f, axs = plt.subplots(3,1,figsize=(4,9))        
        
    for epsi, epsilon in enumerate(epsilon_array):
        print("epsilon=",epsilon)
        #folder for storing the data:
        folder=main_folder+"zeta="+str(zeta)+"_alpha="+str(alpha)+"_epsilon="+str(epsilon)+"/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        rhos = np.sort(np.around(np.concatenate((np.linspace(0.8,1.2,21),np.linspace(0.95,1.05,26))),5));rho0_array=rhos.copy();
        pa=rhos.copy();pb=rhos.copy()

        for ri, rho in enumerate(rhos):
            print (ri, rho)
 
            aL_t, bL_t, aR_t, bR_t = simulate(rho,epsilon,alpha, beta, gamma, zeta)
        
            #% Plotting at each rho:
            rho0_array[ri],pa[ri],pb[ri]=plotting_gradient.plots_at_rho(aL_t,bL_t,aR_t,bR_t,rho,folder)
            
            
        ## rho0 vs rho
        axs[0].plot(rhos,rho0_array,'.-',label=str(epsilon));
        axs[0].set_title(r"$\rho_{0} \ v/s \  \rho$");
        axs[0].set_ylabel(r"$\rho_{0}$")
        axs[0].set_xlabel(r"$\rho$")
        axs[0].legend(ncol=2)        
        ## rho0 vs rho
        axs[1].plot(rhos,pa,'.-',label=str(epsilon));
        axs[1].set_title(r'$p_{a} \ v/s \ \rho$');
        axs[1].set_ylabel(r"$p_{a}$")
        axs[1].set_xlabel(r"$\rho$")
        axs[1].legend(ncol=2)
        ## rho0 vs rho
        axs[2].plot(rhos,pb,'.-',label=str(epsilon));
        axs[2].set_title(r'$p_{b} \ v/s \ \rho $');
        axs[2].set_ylabel(r"$p_{b} $")
        axs[2].set_xlabel(r"$\rho$")
        axs[2].legend(ncol=2)
        
    f.suptitle(r"zeta="+str(zeta))
    f.subplots_adjust(top=0.85, bottom=0.20, left=0.20, right=0.95, hspace=0.50,wspace=0.50)
    
    f.savefig(main_folder+"Gradient_over_rho_zeta="+str(zeta)+".png",dpi=500)    
    plt.close()
    
