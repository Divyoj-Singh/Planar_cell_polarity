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
start=50;
stop=50;
step=10


def plots_at_rho(aL,bL,aR,bR,rho,folder):
    nx=np.size(aL,2);
    ## Slicing the array:
    aL=aL[:,:,start:nx-stop+step:step];aR=aR[:,:,start:nx-stop+step:step]
    bL=bL[:,:,start:nx-stop+step:step];bR=bR[:,:,start:nx-stop+step:step]
    
    ## Calculating observables:
    pa = aR - aL; pb = bR - bL
    a0 = aR + aL; b0 = bR + bL
    average_p=pa+pb;delta_p=pa-pb;
    
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
    
def plots_over_rho(rhos,rho0_array,p_array,delta_p_array,folder):
    
    f, axs = plt.subplots(3,1,figsize=(4,9))        
    ## rho0 vs rho
    axs[0].plot(rhos,rho0_array);
    axs[0].set_title(r"$\rho_{0} \ v/s \  \rho$");
    axs[0].set_ylabel(r"$\rho_{0}$")
    axs[0].set_xlabel(r"$\rho$")
    
    ## rho0 vs rho
    axs[1].plot(rhos,p_array);
    axs[1].set_title(r'$<p> \ v/s \ \rho$');
    axs[1].set_ylabel(r"$<p>$")
    axs[1].set_xlabel(r"$\rho$")
    
    ## rho0 vs rho
    axs[2].plot(rhos,delta_p_array);
    axs[2].set_title(r'$\Delta p \ v/s \ \rho $');
    axs[2].set_ylabel(r"$\Delta p $")
    axs[2].set_xlabel(r"$\rho$")
    
    
    f.suptitle('transition')
    f.subplots_adjust(top=0.85, bottom=0.20, left=0.20, right=0.95, hspace=0.50,wspace=0.50)
    
    f.savefig(folder+'final'+".png")    
    plt.close()
    
    
def plots_over_alpha(alphas,rho0_array,p_array,delta_p_array,folder):
    
    f, axs = plt.subplots(3,1,figsize=(4,9))        

    ## rho0 vs alpha
    axs[0].plot(alphas,rho0_array);
    axs[0].set_title('rho0 vs alpha');
    axs[0].set_ylabel("rho0")
    axs[0].set_xscale('log')
    
    axs[1].plot(alphas,p_array);
    axs[1].set_title('<p> vs alpha');
    axs[1].set_ylabel("<p>")
    axs[1].set_xscale('log')
    
    axs[2].plot(alphas,delta_p_array);
    axs[2].set_title('delta p vs alpha');
    axs[2].set_ylabel("delta p")
    axs[2].set_xlabel("alpha")
    axs[2].set_xscale('log')
    
    f.suptitle('transition')
    f.subplots_adjust(top=0.85, bottom=0.20, left=0.20, right=0.95, hspace=0.50,wspace=0.50)
    
    f.savefig(folder+'final'+".png")    
    plt.close()
    
    
def plots_over_epsilon(epsilons,rho0_array,p_array,delta_p_array,folder):
    
    f, axs = plt.subplots(3,1,figsize=(4,9))        
    ## rho0 vs epsilon
    axs[0].plot(epsilons,rho0_array);
    axs[0].set_title(r"$\rho_{0} \ v/s \  \epsilon$");
    axs[0].set_ylabel(r"$\rho_{0}$")
    axs[0].set_xlabel(r"$\epsilon$")
    
    ## rho0 vs epsilon
    axs[1].plot(epsilons,p_array);
    axs[1].set_title(r'$<p> \ v/s \ \epsilon$');
    axs[1].set_ylabel(r"$<p>$")
    axs[1].set_xlabel(r"$\epsilon$")
    
    ## rho0 vs epsilon
    axs[2].plot(epsilons,delta_p_array);
    axs[2].set_title(r'$\Delta p \ v/s \ \epsilon $');
    axs[2].set_ylabel(r"$\Delta p $")
    axs[2].set_xlabel(r"$\epsilon$")
    
    
    f.suptitle('transition')
    f.subplots_adjust(top=0.85, bottom=0.20, left=0.20, right=0.95, hspace=0.50,wspace=0.50)
    
    f.savefig(folder+'final'+".png")    
    plt.close()
    
    
    
def plots_over_time(aL,bL,aR,bR,rho,folder):
    nx=np.size(aL,2);
    
    ## Slicing the array:
    aL=aL[:,:,start:nx-stop+step:step];aR=aR[:,:,start:nx-stop+step:step]
    bL=bL[:,:,start:nx-stop+step:step];bR=bR[:,:,start:nx-stop+step:step]
    
    ## Calculating observables:
    pa = aR - aL; pb = bR - bL
    a0 = aR + aL; b0 = bR + bL
    average_p=pa+pb;delta_p=pa-pb;
    
    ## Plotting:
    fig, axs = plt.subplots(4,1,figsize=(40,20))        
    
    ## pa and pb over time
    axs[0].plot(pa[:,0,:],'-b',label ='pa');
    axs[0].plot(pb[:,0,:],'-r',label='pb');
    axs[0].set_title(r'p_{a} and p_{b} over time');
    axs[0].set_ylabel(r"p_{a} and p_{b}")
    axs[0].set_xlabel("time")
    
    ## average p  and delta p over time
    axs[1].plot(average_p[:,0,:],'-b',label ='average_p');
    axs[1].plot(delta_p[:,0,:],'-r',label='delta_p');
    axs[1].set_title(r'$<p> and \ \delta_p \ over\  time $');
    axs[1].set_ylabel(r"\average_p and \delta_p")
    axs[1].set_xlabel("time")
    
    ## aR and aL over time
    axs[2].plot(aL[:,0,:],'-b',label ='aL');
    axs[2].plot(aR[:,0,:],'-r',label='aR');
    axs[2].set_title('aR and aL over time');
    axs[2].set_ylabel("aR and aL")
    axs[2].set_xlabel("time")
    
    ## 'bR and bL over time'
    axs[3].plot(bL[:,0,:],'-b',label ='bL');
    axs[3].plot(bR[:,0,:],'-r',label='bR');
    axs[3].set_title('bR and bL over time');
    axs[3].set_ylabel("bR and bL")
    axs[3].set_xlabel("time")
    
    fig.suptitle("rho="+str(rho))
    fig.subplots_adjust(hspace=0.5,wspace=0.5)#top=0.90, bottom=0.10, left=0.15, right=0.95, 
    
    fig.savefig(folder+"rho="+str(rho)+".png")    
    plt.close()