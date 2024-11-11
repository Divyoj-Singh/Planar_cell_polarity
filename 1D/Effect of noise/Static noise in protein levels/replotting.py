import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib as mpl
import os
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

if __name__ == "__main__":
    

    main_folder="./"
    ouput_folder="./output/"
    if not os.path.exists(ouput_folder):
        os.makedirs(ouput_folder)
    
    f = plt.figure(figsize=(5*4/3,5))
    for rho in [2.6,2.8]:
        epsilon=0
        alpha=1
                
        data_file = main_folder + "epsilon=" + str(epsilon) + "_alpha=" + str(alpha) + "_rho=" + str(rho)
        data=np.load(data_file+ ".npz")
        noise_amp_array= data["noise_amp_array"]
        delta_p_array= data["delta_p_array"]

        plt.plot(noise_amp_array,np.mean(delta_p_array,axis=1),label=str(rho));
        plt.fill_between(noise_amp_array,np.mean(delta_p_array,axis=1)-np.std(delta_p_array,axis=1)
                         ,np.mean(delta_p_array,axis=1)+np.std(delta_p_array,axis=1),alpha=0.2)
        plt.ylabel("Polarization"+r'$(p)$',fontsize=labelFontSize)
        plt.xlabel("Noise amplitude" +r"$(S)$",fontsize=labelFontSize)
        plt.xticks(fontsize=tickFontSize-2)
        plt.yticks(fontsize=tickFontSize-2)

        plt.xscale('log')
        
    plt.legend(bbox_to_anchor=(0.45, 0.95),frameon=False,fontsize=tickFontSize-4
               ,title=r'$\rho$',title_fontsize=tickFontSize-4)
            
    plt.text(1e-8,2.7,'A',weight="bold",fontsize=panelLabelFontSize+4)

    plt.tight_layout()

    plt.savefig(ouput_folder+"Noise_in_protein_kinetics"+".png",dpi=800)    
    plt.close()


