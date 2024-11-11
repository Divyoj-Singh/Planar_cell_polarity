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

if __name__ == "__main__":
    rho=2.6   

    main_folder="./"
    folder= main_folder+ "polarity_vs_gradient_for_different_noise/"
    cols = ['r','g','b']
    
    #%% delta p vs gradient for different values of noise:
    f, axs = plt.subplots(1,2,figsize=(10,5)) 
    for noisei, noise in enumerate([0,1e-3,5e-3]):
        data_file=folder+"noise="+str(noise)+"_rho="+str(rho)
        with np.load(data_file+".npz") as data:
            epsilon_array=data['noise_amp_array']
            pa_array=data['pa_array']
            pb_array=data['pb_array']

        ## Pa vs rho
        axs[0].plot(epsilon_array,np.mean(pa_array,axis=1),cols[noisei],label=str(noise));
        ## fill the standard deviation:
        axs[0].fill_between(epsilon_array,np.mean(pa_array,1)-np.std(pa_array,1)
                            ,np.mean(pa_array,1)+np.std(pa_array,1),color=cols[noisei],alpha=0.1)
        axs[0].set_ylabel("Polarity of Ft "+r"$(p_{f})$",fontsize=labelFontSize)
        axs[0].set_xlabel("Gradient of Ft "+r"$(\epsilon)$",fontsize=labelFontSize)

        # plt.xticks( fontsize=tickFontSize-2)
        # plt.yticks(fontsize=tickFontSize-2)
    
        ## Pb vs rho
        axs[1].plot(epsilon_array,np.mean(pb_array,axis=1),cols[noisei],label=str(noise));
        ## fill the standard deviation:
        axs[1].fill_between(epsilon_array,np.mean(pb_array,1)-np.std(pb_array,1)
                            ,np.mean(pb_array,1)+np.std(pb_array,1),color=cols[noisei],alpha=0.1)
        axs[1].set_ylabel("Polarity of Ds "+r"$(p_{d})$",fontsize=labelFontSize)
        axs[1].set_xlabel("Gradient of Ft "+r"$(\epsilon)$",fontsize=labelFontSize)
        # plt.xticks( fontsize=tickFontSize-2)
        # plt.yticks(fontsize=tickFontSize-2)

    axs[0].legend(loc='upper left',bbox_to_anchor=(0.45, 0.95),frameon=False,fontsize=tickFontSize-4
               ,title=r'$S$',title_fontsize=tickFontSize-4)
    axs[1].legend(loc='upper right',bbox_to_anchor=(0.45, 0.95),frameon=False,fontsize=tickFontSize-4
               ,title=r'$S$',title_fontsize=tickFontSize-4)
    
    
    plt.text(1, 1, "A", fontsize=50, transform=plt.gcf().transFigure)    
    
    f.suptitle("Polarity as function of gradient for different noise amplitude",fontsize=labelFontSize)
    f.subplots_adjust(top=0.90, bottom=0.20, left=0.15, right=0.98, hspace=0.15,wspace=0.30)
    f.savefig(folder+"polarity_vs_gradient_for_different_noise"+".png",dpi=800)    
    plt.close()