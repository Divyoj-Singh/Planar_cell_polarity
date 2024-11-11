# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:51:40 2021

@author: divyoj
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
start=10;
stop=10;
step=1

def calculate_delta_p(aL,bL,aR,bR):
    nx=np.size(aL,2);
    ## Slicing the array:
    
    aL=aL[-1,:,start:nx-stop+step:step];aR=aR[-1,:,start:nx-stop+step:step]
    bL=bL[-1,:,start:nx-stop+step:step];bR=bR[-1,:,start:nx-stop+step:step]
    
    ## Calculating delta P:
    pa = aR - aL; pb = bR - bL
    delta_p=pa-pb;

    return (np.mean(delta_p))

def plots_at_rho(aL,bL,aR,bR,rho,folder,draw_plots=True):
    nx=np.size(aL,2);
    ## Slicing the array:
    aL=aL[:,:,start:nx-stop+step:step];aR=aR[:,:,start:nx-stop+step:step]
    bL=bL[:,:,start:nx-stop+step:step];bR=bR[:,:,start:nx-stop+step:step]
    
    ## Calculating observables:
    pa = aR - aL; pb = bR - bL
    a0 = aR + aL; b0 = bR + bL
    average_p=pa+pb;delta_p=pa-pb;
    nx=np.size(aL,2);
    
    if draw_plots==True:
        
        ## Plotting:
        fig, axs = plt.subplots(3,2,figsize=(9,8))        
        ## <p>
        axs[0,0].plot(average_p[-1,0,:]);
        axs[0,0].set_title(r'$<p>$');
        axs[0,0].set_ylabel(r"$<p>$")
        axs[0,0].set_xlabel("x")
    
        ## delta p
        axs[0,1].plot(delta_p[-1,0,:]);
        axs[0,1].set_title(r'$\Delta p$');
        axs[0,1].set_ylabel(r"$\Delta p$")
        #axs[0,1].set_yscale('log')
        axs[0,1].set_xlabel("x")
    
        ## aR and aL
        axs[1,0].plot(aL[-1,0,:],'b',label ='aL');
        axs[1,0].plot(aR[-1,0,:],'r',label='aR');
        axs[1,0].legend()
        axs[1,0].set_title('aR and aL');
        axs[1,0].set_ylabel("aR and aL")
        axs[1,0].set_xlabel("x")
        
        ## bR and bL
        axs[1,1].plot(bL[-1,0,:],'-b',label ='bL');
        axs[1,1].plot(bR[-1,0,:],'-r',label='bR');
        axs[1,1].legend()
        axs[1,1].set_title('bR and bL');
        axs[1,1].set_ylabel("bR and bL")
        axs[1,1].set_xlabel("x")
        
        ## aR and aL over time
        axs[2,0].plot(aL[:,0,:],'-b',label ='aL');
        axs[2,0].plot(aR[:,0,:],'-r',label='aR');
        axs[2,0].set_title('aR and aL over time');
        axs[2,0].set_ylabel("aR and aL")
        axs[2,0].set_xlabel("time")
        
        ## 'bR and bL over time'
        axs[2,1].plot(bL[:,0,:],'-b',label ='bL');
        axs[2,1].plot(bR[:,0,:],'-r',label='bR');
        axs[2,1].set_title('bR and bL over time');
        axs[2,1].set_ylabel("bR and bL")
        axs[2,1].set_xlabel("time")
        
        fig.suptitle(str(rho))
        fig.subplots_adjust(hspace=0.5,wspace=0.5)#top=0.90, bottom=0.10, left=0.15, right=0.95, 
        
        fig.savefig(folder+str(rho)+".png")    
        plt.close()
    
    return (np.mean(a0[-1,0,:]),np.mean(average_p[-1,0,:]),np.mean(delta_p[-1,0,:]))
    
