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
import plotting
from scipy.integrate import solve_ivp
@njit(parallel=True)
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
    daR[0,nx-1]=daR[0,nx-1-1];
    
    #bL
    dbL[0,0]=dbL[0,1];
    dbL[0,nx-1]=alpha*(bT[0,nx-1]-b0[0,nx-1])-beta*bL[0,nx-1]+beta*gamma*(bL[0,nx-1]*aR[0,nx-1-1])-zeta*(bL[0,nx-1]-bR[0,nx-1])**3;
    
    #bR
    dbR[0,0]=alpha*(bT[0,0]-b0[0,0])-beta*bR[0,0]+beta*gamma*( bR[0,0]*aL[0,1] ) -zeta*(aR[0,0]-aL[0,0])**3;
    dbR[0,nx-1]=dbR[0,nx-2];
    
    # checking if membrane concentration went above total concetration:
    # checking if membrane concentration went above total concetration:
    for i in range(ny):
        for j in range(nx):
            if a0[i,j]>aT[i,j]:
                if daL[i,j]>0:
                    daL[i,j]=0
                if daL[i,j]>0:
                    daR[i,j]=0
                
            if b0[i,j]>bT[i,j]:
                if dbL[i,j]>0:
                    dbL[i,j]=0
                if dbL[i,j]>0:
                    dbR[i,j]=0
            
    #return np.array(daL.flatten().tolist()+dbL.flatten().tolist()+daR.flatten().tolist()+dbR.flatten().tolist())
    return np.concatenate((daL.flatten(),dbL.flatten(),daR.flatten(),dbR.flatten()))

#@njit
def simulate(rho,alpha, beta, gamma, zeta):
    ''' function to iterate over time and return arrays with the result '''
    ## initilizing the arrays to store the values over time:
    aL_t = np.zeros((T_max+1,ny,nx)); aR_t = np.zeros((T_max+1,ny,nx));
    bL_t = np.zeros((T_max+1,ny,nx)); bR_t = np.zeros((T_max+1,ny,nx));

    # total proteins in the cells
    aT = rho + np.zeros((ny,nx))
    bT = rho + np.zeros((ny,nx))
  
    ## initializing aL,bR,bL,aR
    aL = np.zeros((ny,nx)) + 0.01*rho; aR = np.zeros((ny,nx)) + 0.1*rho
    bL = np.zeros((ny,nx)) + 0.1*rho; bR = np.zeros((ny,nx)) + 0.01*rho
    
    ## Collecting the initial conditions into a single array:
    ic = np.array(aL.flatten().tolist()+bL.flatten().tolist()+aR.flatten().tolist()+bR.flatten().tolist())
    
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

    return (aL_t,bL_t,aR_t,bR_t)



if __name__ == "__main__":
    # Lattice:
    w,h = 10,2;
    dx,dy=0.01,0.1;
    nx=int(w/dx)
    ny=1;#int(h/dx);
    
    # time:
    T_max=50;
    
    # parameters:
    alpha=10;
    gamma=1;beta=1;
    zeta=0.0;
    
    #%% Characterisation over rho:
    alpha_array=[5,10,20]# np.round(np.logspace(-1,2,5),2)
    f, axs = plt.subplots(3,1,figsize=(4,9))        
        
    for alphai, alpha in enumerate(alpha_array):
        print("alpha=",alpha)
        # folder for storing the data:
        folder="./Uniform_rho/"+"zeta="+str(zeta)+"_alpha="+str(alpha)+"/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        rhos = np.around(np.linspace(0.8,1.5,50),3);rho0_array=rhos*1;
        p_array=rhos*1;delta_p_array=rhos*1
    
        for ri, rho in enumerate(rhos):
            print (ri, rho)
     
            aL_t, bL_t, aR_t, bR_t = simulate(rho,alpha, beta, gamma, zeta)
            
            
             # Plotting at each rho
            #%% Plotting at each rho:
            rho0_array[ri],p_array[ri],delta_p_array[ri]=plotting.plots_at_rho(aL_t,bL_t,aR_t,bR_t,rho,folder)
            
            
        ## rho0 vs rho
        axs[0].plot(rhos,rho0_array,label=str(alpha));
        axs[0].set_title(r"$\rho_{0} \ v/s \  \rho$");
        axs[0].set_ylabel(r"$\rho_{0}$")
        axs[0].set_xlabel(r"$\rho$")
        axs[0].legend(ncol=2)        
        ## rho0 vs rho
        axs[1].plot(rhos,p_array,label=str(alpha));
        axs[1].set_title(r'$<p> \ v/s \ \rho$');
        axs[1].set_ylabel(r"$<p>$")
        axs[1].set_xlabel(r"$\rho$")
        axs[1].legend(ncol=2)
        ## rho0 vs rho
        axs[2].plot(rhos,delta_p_array,label=str(alpha));
        axs[2].set_title(r'$\Delta p \ v/s \ \rho $');
        axs[2].set_ylabel(r"$\Delta p $")
        axs[2].set_xlabel(r"$\rho$")
        axs[2].legend(ncol=2)
        
    f.suptitle(r"zeta="+str(zeta))
    f.subplots_adjust(top=0.85, bottom=0.20, left=0.20, right=0.95, hspace=0.50,wspace=0.50)
    
    f.savefig("./Uniform_rho/"+"Uniform_over_rho_zeta="+str(zeta)+".png",dpi=500)    
    plt.close()
    
    
    #%% Characterisation over rho:
    rhos_array=[0.8,1,1.5]# np.round(np.logspace(-1,2,5),2)
    f, axs = plt.subplots(3,1,figsize=(4,9))        
        
    for rhoi, rho in enumerate(rhos_array):
        print("rho=",rho)
        #folder for storing the data:
        folder="./Uniform_alpha/"+"zeta="+str(zeta)+"_rho="+str(rho)+"/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        alphas = np.round(np.logspace(-1, 1.5, num=20),3);rho0_array=alphas.copy();
        p_array=alphas.copy();delta_p_array=alphas.copy()
    
        for alphai, alpha in enumerate(alphas):
            print (alphai, alpha)
     
            aL_t, aR_t, bL_t, bR_t = simulate(rho,alpha, beta, gamma, zeta)
            
            
            # Plotting at each rho
            rho0_array[alphai],p_array[alphai],delta_p_array[alphai]=plotting.plots_at_rho(aL_t,bL_t,aR_t,bR_t,alpha,folder)
           
            
            ## rho0 vs alpha
        axs[0].plot(alphas,rho0_array,label=str(rho));
        axs[0].set_title(r"$\rho_{0} \ vs\  \alpha$");
        axs[0].set_ylabel(r"$\rho_{0}$")
        axs[0].set_xscale('log')
        axs[0].legend(ncol=2)        
            
        axs[1].plot(alphas,p_array,label=str(rho));
        axs[1].set_title(r'$<p> \ vs \  \alpha$');
        axs[1].set_ylabel(r"$<p>$")
        axs[1].set_xscale('log')
        axs[1].legend(ncol=2)        
            
        axs[2].plot(alphas,delta_p_array,label=str(rho));
        axs[2].set_title(r'$\Delta p \ vs\  \alpha$');
        axs[2].set_ylabel(r"$\Delta p$")
        axs[2].set_xlabel(r"$\alpha$")
        axs[2].set_xscale('log')
        axs[2].legend(ncol=2)        
            
    f.suptitle(r"zeta="+str(zeta))
    f.subplots_adjust(top=0.85, bottom=0.20, left=0.20, right=0.95, hspace=0.50,wspace=0.50)
    
    f.savefig("./Uniform_alpha/"+"Uniform_over_alpha_zeta="+str(zeta)+".png",dpi=500)    
    plt.close()
    