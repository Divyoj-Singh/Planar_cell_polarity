# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:41:38 2021

@author: divyoj
"""# -*- coding: utf-8 -*-
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
    
    # checking if membrane concentration went above total concetration:
    for i in range(ny):
        for j in range(nx):
            if a0[i,j]>aT[i,j]:
                
                if daL[i,j]>0:
                    daL[i,j]=0
                if daR[i,j]>0:
                    daR[i,j]=0
                
            if b0[i,j]>bT[i,j]:
                if dbL[i,j]>0:
                    dbL[i,j]=0
                if dbR[i,j]>0:
                    dbR[i,j]=0

            if aL[i,j]<=0:
                daL[i,j]=0
            if bL[i,j]<=0:
                dbL[i,j]=0

            if aR[i,j]<=0:
                daR[i,j]=0
            if bR[i,j]<=0:
                dbR[i,j]=0
            
    
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
    aL = np.zeros((ny,nx)) + 0.6*rho; aR = np.zeros((ny,nx)) + 0.1*rho
    bL = np.zeros((ny,nx)) + 0.1*rho; bR = np.zeros((ny,nx)) + 0.6*rho
    
    ## Deletion in one range:
    aL[0,:int(nx/2)]=aR[0,:int(nx/2)]=aT[0,:int(nx/2)]=0
    
    
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

    #return (aL_t[:,:,10:nx-10],aR_t[:,:,10:nx-10],bL_t[:,:,10:nx-10],bR_t[:,:,10:nx-10])
    return (aL_t,bL_t,aR_t,bR_t)



if __name__ == "__main__":
    # Lattice:
    w,h = 10,2;
    dx,dy=0.01,0.1;
    nx=int(w/dx)
    ny=1;#int(h/dx);
    # time:
    T_max=100;
    
    # parameters:
    alpha=10; 
    gamma=1; beta=1;
    zeta=0.0;
    
#%% Characterisation over rho:    
    # folder for storing the data:
    folder="./"+"zeta="+str(zeta)+"_alpha="+str(alpha)+"/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    rhos = np.around(np.linspace(0.8,1.2,21),2);rho0_array=rhos.copy();
    p_array=rhos.copy();delta_p_array=rhos.copy()

    for ri, rho in enumerate(rhos):
        print (ri, rho)
 
        aL_t, bL_t, aR_t, bR_t = simulate(rho,alpha, beta, gamma, zeta)
        
        
         # Plotting at each rho
        #%% Plotting at each rho:
        rho0_array[ri],p_array[ri],delta_p_array[ri]=plotting.plots_at_rho(aL_t,bL_t,aR_t,bR_t,rho,folder)
        
        
    
    # Plotting final graphs after iterating over rhos
    plotting.plots_over_rho(rhos,rho0_array,p_array,delta_p_array,folder)        
  