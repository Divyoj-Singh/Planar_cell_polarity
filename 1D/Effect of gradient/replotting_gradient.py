# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:41:38 2021

@author: divyoj
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

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
rc('xtick', labelsize=tickFontSize-2) 
rc('ytick', labelsize=tickFontSize-2) 
rc('axes', labelsize=labelFontSize)
rc('axes', titlesize=labelFontSize)
rc('legend', fontsize=legendFontSize-4)
rc('lines', linewidth=lw)
rc('legend', frameon=False)
rc('legend',title_fontsize=legendFontSize-4)
# rc('legend',bbox_to_anchor=(0.45, 0.95), loc='best')


if __name__ == "__main__":
 
    main_folder="./"
    
    # #%% Characterisation over epsilon for multiple small values of rho:
    rho_array=[2.2,2.4]

    #folder for storing the data:
    folder=main_folder+"rho_low/"
    f, axs = plt.subplots(1,2,figsize=(10,5))        
        
    for rhoi, rho in enumerate(rho_array):
        ## reading data from npz file:
        data_file=folder+"data_rho="+str(rho)+".npz"
        with np.load(data_file) as data:
            epsilons=data['epsilons']
            pa=data['pa']
            pb=data['pb']
        ## Pa vs rho
        axs[0].plot(epsilons,pa*1000,label=str(rho));
        ##axs[1].set_title("Polarity of Ft v/s Gradient");
        axs[0].set_ylabel("Polarity of Ft ($10^{-3}$) "+r"$(p_{f})$",fontsize=labelFontSize)
        axs[0].set_xlabel("Gradient of Ft "+r"$(\epsilon)$",fontsize=labelFontSize)
        #axs[0].legend(title=r"$\rho$")
        #axs[1].set_aspect("equal")
    
        ## Pb vs rho
        axs[1].plot(epsilons,pb*1000,label=str(rho));
        ##axs[2].set_title("Polarity of Ds v/s Gradient");
        axs[1].set_ylabel("Polarity of Ds ($10^{-3}$) "+r"$(p_{d})$",fontsize=labelFontSize)
        axs[1].set_xlabel("Gradient of Ft "+r"$(\epsilon)$",fontsize=labelFontSize)
        #axs[2].set_aspect("equal")
    axs[0].legend(bbox_to_anchor=(0.45, 0.95),title=r"$\rho$")
    axs[1].legend(bbox_to_anchor=(0.45, 0.95),title=r"$\rho$")
        
    f.subplots_adjust(top=0.90, bottom=0.20, left=0.15, right=0.98, hspace=0.15,wspace=0.30)
    
    f.savefig(folder+"p_vs_epsilon_low_rho"+".png",dpi=500)    
    plt.close()

    #%% Characterisation over epsilon for multiple large values of rho::
    
    rho_array=[2.6,2.8]
    #folder for storing the data:
    folder=main_folder+"high_rho/"
    f, axs = plt.subplots(1,2,figsize=(10,5))        
        
    for rhoi, rho in enumerate(rho_array):
        # reading data from npz file:
        data_file=folder+"data_rho="+str(rho)+".npz"
        with np.load(data_file) as data:
            epsilons=data['epsilons']
            pa=data['pa']
            pb=data['pb']
        
        ## Pa vs rho
        axs[0].plot(epsilons,pa,label=str(rho));
        #axs[1].set_title("Polarity of Ft v/s Gradient");
        axs[0].set_ylabel("Polarity of Ft "+r"$(p_{f})$",fontsize=labelFontSize)
        axs[0].set_xlabel("Gradient of Ft "+r"$(\epsilon)$",fontsize=labelFontSize)
        # axs[0].legend(title=r"$\rho$")
        # #axs[1].set_aspect("equal")
    
        ## Pb vs rho
        axs[1].plot(epsilons,pb,label=str(rho));
        #axs[2].set_title("Polarity of Ds v/s Gradient");
        axs[1].set_ylabel("Polarity of Ds "+r"$(p_{d})$",fontsize=labelFontSize)
        axs[1].set_xlabel("Gradient of Ft "+r"$(\epsilon)$",fontsize=labelFontSize)
        # axs[1].legend(title=r"$\rho$")
        # #axs[2].set_aspect("equal")
    
    axs[0].legend(loc='upper left',bbox_to_anchor=(0.45, 0.95),title=r"$\rho$")
    axs[1].legend(bbox_to_anchor=(0.45, 0.95),title=r"$\rho$")    
    f.subplots_adjust(top=0.90, bottom=0.20, left=0.15, right=0.98, hspace=0.15,wspace=0.30)
    
    f.savefig(folder+"p_vs_epsilon_high_rho"+".png",dpi=500)    
    plt.close()
    
    # #%% Characterisation over rho:
    epsilon_array=[0,0.1,0.5]
    #folder for storing the data:
    folder=main_folder+"epsilon/"
    f, axs = plt.subplots(1,2,figsize=(10,5))        
    
    for epsi, epsilon in enumerate(epsilon_array):
        data_epsilon=folder+"data_epsilon="+str(epsilon)+".npz"
        with np.load(data_epsilon) as data:
            rhos=data['rhos']
            pa=data['pa']
            pb=data['pb']
    
        ## rho0 vs rho
        axs[0].plot(rhos,pa,label=str(epsilon));
        #axs[1].set_title("Polarity of Ft v/s Total protein conc.");
        axs[0].set_ylabel("Polarity of Ft "+r"$(p_{f})$",fontsize=labelFontSize)
        axs[0].set_xlabel("Total Protein Conc. "+r"$(\rho)$",fontsize=labelFontSize)
        # axs[0].legend(title=r"$\epsilon$")
        #axs[1].set_aspect("equal")
    
        ## rho0 vs rho
        axs[1].plot(rhos,pb,label=str(epsilon));
        #axs[2].set_title("Polarity of Ds v/s Total protein conc.");
        axs[1].set_ylabel("Polarity of Ds "+r"$(p_{d})$",fontsize=labelFontSize)
        axs[1].set_xlabel("total protein conc. "+r"$(\rho)$",fontsize=labelFontSize)
        # axs[1].legend(title=r"$\epsilon$")
        #axs[2].set_aspect("equal")

    axs[0].legend(bbox_to_anchor=(0.45, 0.95),title=r"$\epsilon$")
    axs[1].legend(bbox_to_anchor=(0.45, 0.95),title=r"$\epsilon$")   
    
    f.subplots_adjust(top=0.90, bottom=0.20, left=0.15, right=0.98, hspace=0.15,wspace=0.30)

    f.savefig(folder+"Gradient_over_rho"+".png",dpi=500)    
    plt.close()