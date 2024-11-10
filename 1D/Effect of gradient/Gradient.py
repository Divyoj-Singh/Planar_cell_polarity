# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:41:38 2021

@author: divyoj
"""

## importing libraries:
import numpy as np
import matplotlib.pyplot as plt
import os
# # note that this must be executed before 'import numba'
# os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'
from numba import njit
import time as process_time
import plotting_gradient
from scipy.integrate import solve_ivp
import seaborn as sns
sns.set_theme(context='notebook', style='ticks',font='arial',font_scale=1.2)
cm = 1/2.54  # centimeters in inches
## functions:
@njit
def do_timestep(t,z,aT,bT,alpha, beta):
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
    daL[0,1:nx-1]=alpha*(aT[0,1:nx-1]-a0[0,1:nx-1])-beta*aL[0,1:nx-1]/(1+bR[0,1-1:nx-1-1]**2);
    # aR
    daR[0,1:nx-1]=alpha*(aT[0,1:nx-1]-a0[0,1:nx-1])-beta*aR[0,1:nx-1]/(1+bL[0,1+1:nx-1+1]**2 )
    # bL
    dbL[0,1:nx-1]=alpha*(bT[0,1:nx-1]-b0[0,1:nx-1])-beta*bL[0,1:nx-1]/(1+aR[0,1-1:nx-1-1]**2) 
    # bR
    dbR[0,1:nx-1]=alpha*(bT[0,1:nx-1]-b0[0,1:nx-1])-beta*bR[0,1:nx-1]/(1+aL[0,1+1:nx-1+1]**2 )

    # Boundary Conditions:
    #aL
    daL[0,0]=daL[0,1];
    daL[0,nx-1]=alpha*(aT[0,nx-1]-a0[0,nx-1])-beta*aL[0,nx-1]/(1+bR[0,nx-1-1]**2);
    
    #aR
    daR[0,0]=alpha*(aT[0,0]-a0[0,0])-beta*aR[0,0]/(1+bL[0,1]**2) ;
    daR[0,nx-1]=daR[0,nx-2];
    
    #bL
    dbL[0,0]=dbL[0,1];
    dbL[0,nx-1]=alpha*(bT[0,nx-1]-b0[0,nx-1])-beta*bL[0,nx-1]/(1+aR[0,nx-1-1]**2);
    
    #bR
    dbR[0,0]=alpha*(bT[0,0]-b0[0,0])-beta*bR[0,0]/(1+aL[0,1]**2);
    dbR[0,nx-1]=dbR[0,nx-2];
                
    return np.concatenate((daL.flatten(),dbL.flatten(),daR.flatten(),dbR.flatten()))

#@njit
def simulate(rho,epsilon,alpha, beta):
    ''' function to iterate over time and return arrays with the result '''
    ## initilizing the arrays to store the values over time:
    aL_t = np.zeros((T_max+1,ny,nx)); aR_t = np.zeros((T_max+1,ny,nx));
    bL_t = np.zeros((T_max+1,ny,nx)); bR_t = np.zeros((T_max+1,ny,nx));
    
    # total proteins in the cells
    aT = rho + np.zeros((ny,nx))+epsilon*np.linspace(-0.5,0.5,nx)*np.ones((ny,nx))
    bT = rho + np.zeros((ny,nx))
    
    ## initializing aL,bR,bL,aR
    
    aL = np.zeros((ny,nx)) + 0.100001*rho; aR = np.zeros((ny,nx)) + 0.10*rho
    bL = np.zeros((ny,nx)) + 0.10*rho; bR = np.zeros((ny,nx)) + 0.100001*rho
    
    
    ## Collecting the initial conditions into a single array:
    ic = np.array(aL.flatten().tolist()+bL.flatten().tolist()+aR.flatten().tolist()+bR.flatten().tolist())
    
    ## Solving the initial value problem:
    sol = solve_ivp(lambda t,y: do_timestep(t,y,aT,bT,alpha, beta),t_span=[0,T_max],y0=ic,t_eval=list(np.linspace(0,T_max,T_max+1)))

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
    t_initial= process_time.time()
    # Lattice:
    w,h = 50,2;
    dx,dy=0.1,1;
    nx=int(w/dx)
    ny=1;#int(h/dx);
    # time:
    T_max=1000;
    
    # parameters:
    alpha=1; 
    beta=1;

    main_folder="./"
    
    # #%% Characterisation over epsilon for multiple small values of rho:
    rho_array=[2.2,2.4]

    #folder for storing the data:
    folder=main_folder+"rho_low/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    f, axs = plt.subplots(1,2,figsize=(20*cm,8*cm))        
        
    for rhoi, rho in enumerate(rho_array):
        print("rho=",rho)
        t_initial_for_rho=process_time.time()
                
        epsilons = np.around(np.linspace(-0.1,0.1,11),5);rho0_array=epsilons*1;
        pa=epsilons*1;pb=epsilons*1

        for ri, epsilon in enumerate(epsilons):
            print (ri, epsilon)
    
            aL_t, bL_t, aR_t, bR_t = simulate(rho,epsilon,alpha, beta)
        
            # Plotting at each rho
            rho0_array[ri],pa[ri],pb[ri]=plotting_gradient.plots_at_rho(aL_t,bL_t,aR_t,bR_t,epsilon,folder,plots=False)
    
        ## saving data in npz file:
        np.savez(folder+"data_rho="+str(rho)+".npz",epsilons=epsilons,pa=pa,pb=pb)
        ## Pa vs rho
        axs[0].plot(epsilons,pa,'.-',label=str(rho));
        ##axs[1].set_title("Polarity of Ft v/s Gradient");
        axs[0].set_ylabel("Polarity of Ft "+r"$(p_{f})$")
        axs[0].set_xlabel("Gradient of Ft "+r"$(\epsilon)$")
        axs[0].legend(title=r"$\rho$")
        #axs[1].set_aspect("equal")
    
        ## Pb vs rho
        axs[1].plot(epsilons,pb,'.-',label=str(rho));
        ##axs[2].set_title("Polarity of Ds v/s Gradient");
        axs[1].set_ylabel("Polarity of Ds "+r"$(p_{d})$")
        axs[1].set_xlabel("Gradient of Ft "+r"$(\epsilon)$")
        axs[1].legend(title=r"$\rho$")
        #axs[2].set_aspect("equal")
    
    f.subplots_adjust(top=0.90, bottom=0.20, left=0.15, right=0.98, hspace=0.15,wspace=0.45)
    
    f.savefig(folder+"p_vs_epsilon_low_rho"+".png",dpi=500)    
    plt.close()

    #%% Characterisation over epsilon for multiple large values of rho::
    
    rho_array=[2.6,2.8]
    #folder for storing the data:
    folder=main_folder+"high_rho/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    f, axs = plt.subplots(1,2,figsize=(20*cm,8*cm))        
        
    for rhoi, rho in enumerate(rho_array):
        print("rho=",rho)
        t_initial_for_rho=process_time.time()
        
        
        epsilons = np.sort(np.around(np.concatenate((np.linspace(-0.1,0.1,10),np.linspace(-0.01,0.01,10))),5));
        rho0_array=epsilons*1; pa=epsilons*1;pb=epsilons*1

        for ri, epsilon in enumerate(epsilons):
            print (ri, epsilon)
    
            aL_t, bL_t, aR_t, bR_t = simulate(rho,epsilon,alpha, beta)
        
            # Plotting at each rho
            rho0_array[ri],pa[ri],pb[ri]=plotting_gradient.plots_at_rho(aL_t,bL_t,aR_t,bR_t,epsilon,folder,plots=False)
        
        ## saving data in npz file:
        np.savez(folder+"data_rho="+str(rho)+".npz",epsilons=epsilons,pa=pa,pb=pb)
        
        ## Pa vs rho
        axs[0].plot(epsilons,pa,'.-',label=str(rho));
        #axs[1].set_title("Polarity of Ft v/s Gradient");
        axs[0].set_ylabel("Polarity of Ft "+r"$(p_{f})$")
        axs[0].set_xlabel("Gradient of Ft "+r"$(\epsilon)$")
        axs[0].legend(title=r"$\rho$")
        #axs[1].set_aspect("equal")
    
        ## Pb vs rho
        axs[1].plot(epsilons,pb,'.-',label=str(rho));
        #axs[2].set_title("Polarity of Ds v/s Gradient");
        axs[1].set_ylabel("Polarity of Ds "+r"$(p_{d})$")
        axs[1].set_xlabel("Gradient of Ft "+r"$(\epsilon)$")
        axs[1].legend(title=r"$\rho$")
        #axs[2].set_aspect("equal")
    
        
    f.subplots_adjust(top=0.90, bottom=0.20, left=0.15, right=0.98, hspace=0.15,wspace=0.45)
    
    f.savefig(folder+"p_vs_epsilon_high_rho"+".png",dpi=500)    
    plt.close()
    
    # #%% Characterisation over rho:
    epsilon_array=[0,0.1,0.5]
    
    #folder for storing the data:
    folder=main_folder+"epsilon/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    f, axs = plt.subplots(1,2,figsize=(20*cm,8*cm))        
    
    for epsi, epsilon in enumerate(epsilon_array):
        print("epsilon=",epsilon)
        t_initial_for_rho=process_time.time()
        
        rhos = np.sort(np.around(np.concatenate((np.linspace(2,3,10),np.linspace(2.5,2.6,10))),5));
        rho0_array=rhos*1;pa=rhos*1;pb=rhos*1

        for ri, rho in enumerate(rhos):
            print (ri, rho)
    
            aL_t, bL_t, aR_t, bR_t = simulate(rho,epsilon,alpha, beta)
        
            #% Plotting at each rho:
            rho0_array[ri],pa[ri],pb[ri]=plotting_gradient.plots_at_rho(aL_t,bL_t,aR_t,bR_t,rho,folder,plots=False)
        
        ## saving data in npz file:
        np.savez(folder+"data_epsilon="+str(epsilon)+".npz",rhos=rhos,pa=pa,pb=pb) 
    
        ## rho0 vs rho
        axs[0].plot(rhos,pa,'.-',label=str(epsilon));
        #axs[1].set_title("Polarity of Ft v/s Total protein conc.");
        axs[0].set_ylabel("Polarity of Ft "+r"$(p_{f})$")
        axs[0].set_xlabel("total protein conc. "+r"$(\rho)$")
        axs[0].legend(title=r"$\epsilon$")
        #axs[1].set_aspect("equal")
    
        ## rho0 vs rho
        axs[1].plot(rhos,pb,'.-',label=str(epsilon));
        #axs[2].set_title("Polarity of Ds v/s Total protein conc.");
        axs[1].set_ylabel("Polarity of Ds "+r"$(p_{d})$")
        axs[1].set_xlabel("total protein conc. "+r"$(\rho)$")
        axs[1].legend(title=r"$\epsilon$")
        #axs[2].set_aspect("equal")
    
    f.subplots_adjust(top=0.90, bottom=0.20, left=0.15, right=0.98, hspace=0.15,wspace=0.45)

    f.savefig(folder+"Gradient_over_rho"+".png",dpi=500)    
    plt.close()
        
    print("total time (in min) ",( process_time.time()-t_initial)/60,"\n")
    
