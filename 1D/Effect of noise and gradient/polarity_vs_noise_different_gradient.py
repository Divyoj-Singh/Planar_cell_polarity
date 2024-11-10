# -*- coding: utf-8 -*-
"""
Created on 15 october 2023

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
import plotting_gradient as plotting_gradient
import seaborn as sns
from tqdm import tqdm
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
    return  (daL,dbL,daR,dbR)

#@njit
def fixed_step_solver(t_span,ic_aL,ic_bL,ic_aR,ic_bR,aT,bT,noise,alpha, beta):

    time=np.arange(dt,T_max+dt,dt)
    aL,bL,aR,bR = 1*ic_aL,1*ic_bL,1*ic_aR,1*ic_bR
    
    
    aL_T = np.zeros((len(t_span)+1,ny,nx)); aR_T = np.zeros((len(t_span)+1,ny,nx));
    bL_T = np.zeros((len(t_span)+1,ny,nx)); bR_T = np.zeros((len(t_span)+1,ny,nx));
    index=0;
    aL_T[index]=aL;bL_T[index]=bL; aR_T[index]=aR;bR_T[index]=bR;
    noise_array=noise*np.random.normal(loc=0.0, scale=1,size=(len(time),4,ny,nx) )

    for ti, t in enumerate(time):
        # calculating dxdt at a time step:
        
        daL,dbL,daR,dbR = do_timestep(aL,bL,aR,bR,aT,bT,alpha, beta)
        aL+= daL*dt+ dt*noise_array[ti,0,:,:];  bL+= dbL*dt+ dt*noise_array[ti,1,:,:];
        aR+= daR*dt+ dt*noise_array[ti,2,:,:];  bR+= dbR*dt+ dt*noise_array[ti,3,:,:];
        
        if (ti%resolution)==1:
            index+=1
            aL_T[index]=aL;bL_T[index]=bL;
            aR_T[index]=aR;bR_T[index]=bR;
            
            
    return(aL_T,bL_T,aR_T,bR_T)
 
def simulate(t_span,rho,epsilon,noise,alpha, beta):
    ''' function to iterate over time and return arrays with the result '''
    
    # total proteins in the cells
    aT = rho + np.zeros((ny,nx))+epsilon*np.linspace(-0.5,0.5,nx)*np.ones((ny,nx))
    bT = rho + np.zeros((ny,nx))
    
     ## initializing aL,bR,bL,aR

    ic_aL = np.zeros((ny,nx)) +  0.100001*rho;  ic_bL = np.zeros((ny,nx)) + 0.10*rho;
    ic_aR = np.zeros((ny,nx)) + 0.10*rho;  ic_bR = np.zeros((ny,nx)) + 0.100001*rho;
       
    aL_t,bL_t,aR_t,bR_t = fixed_step_solver(t_span,ic_aL,ic_bL,ic_aR,ic_bR,aT,bT,noise,alpha, beta)
      
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
    rho=2.6
    no_of_instances=50
    
    
    #%% delta p vs noise for different values of epsilon above the critical point:

    main_folder="./"
    folder= main_folder +  "polarity_vs_noise_for_different_gradient/"
            
    if not os.path.exists(folder):
        os.makedirs(folder)
        
        
    f, axs = plt.subplots(1,2,figsize=(20*cm,8*cm)) 
    for epsilon in [0,0.01,0.1]:
        
        #noise_amp_array = np.around((np.logspace(-7,-2,num=50)),10);
        noise_amp_array = np.around((np.logspace(-7,0,num=50)),10);
        
        pa_array=np.empty((len(noise_amp_array),no_of_instances));
        pb_array=np.empty((len(noise_amp_array),no_of_instances));
    
        
        for ri, noise in tqdm(enumerate(noise_amp_array)):
                           
            for instance in range(0,no_of_instances,1):
                 
                
                aL_t, aR_t, bL_t, bR_t = simulate(t_span,rho,epsilon,noise,alpha, beta)
                pa,pb=plotting_gradient.calculate_pa_pb(aL_t,bL_t,aR_t,bR_t)
                #rho0,p,delta_p=plotting_average.plots_at_rho(aL_t,bL_t,aR_t,bR_t,noise,folder,draw_plots=False)
                pa_array[ri,instance]=pa
                pb_array[ri,instance]= pb
        
            
        data_file=folder+"epsilon="+str(epsilon)+"_rho="+str(rho)
        np.savez(data_file,noise_amp_array=noise_amp_array,pa_array=pa_array,pb_array=pb_array)
    
        ## Pa vs rho
        axs[0].plot(noise_amp_array,np.mean(pa_array,axis=1),'.-',label=str(epsilon));
        # fill between the standard deviation:
        axs[0].fill_between(noise_amp_array,np.mean(pa_array,axis=1)-np.std(pa_array,axis=1),np.mean(pa_array,axis=1)+np.std(pa_array,axis=1),alpha=0.2)

        axs[0].set_ylabel("Polarity of Ft "+r"$(p_{f})$")
        axs[0].set_xlabel("Noise amplitude" +r"$(\eta)$")
        axs[0].legend(title=r"$\epsilon$")
        axs[0].set_xscale('log')
        #axs[1].set_aspect("equal")
    
        ## Pb vs rho
        axs[1].plot(noise_amp_array,np.mean(pb_array,axis=1),'.-',label=str(epsilon));
        # fill between the standard deviation:
        axs[1].fill_between(noise_amp_array,np.mean(pb_array,axis=1)-np.std(pb_array,axis=1),np.mean(pb_array,axis=1)+np.std(pb_array,axis=1),alpha=0.2)
        axs[1].set_ylabel("Polarity of Ds "+r"$(p_{d})$")
        axs[1].set_xlabel("Noise amplitude" +r"$(\eta)$")
        axs[1].legend(title=r"$\epsilon$")
        axs[1].set_xscale('log')
        #axs[2].set_aspect("equal")
            
    f.subplots_adjust(top=0.90, bottom=0.20, left=0.15, right=0.98, hspace=0.15,wspace=0.45)
    
    f.savefig(main_folder+"polarity_vs_noise_for_different_gradient"+".png",dpi=1000)    
    plt.close()
    print("total time (in min) ",( process_time.time()-t_initial)/60,"\n")