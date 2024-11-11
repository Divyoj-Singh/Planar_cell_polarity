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


#noise_type = 'StaticNoise' 
noise_type = 'DynamicNoise'
input_folder = "./data_" + noise_type + "/" 
output_folder = "./results_" + noise_type + "/"

filename= "phase_diagram_" + noise_type
data_file=output_folder+ "data_for_" +filename
file=np.load(data_file+".npz")
noiss=file['noise_amp_array']
rhos=file['rho_array']
corr_len_array=file['corr_len_array']


#%% Normalise the correlation length with respect to aligned state:
print("Max coorelation length is: ",np.max(corr_len_array))
corr_len_array=corr_len_array/np.max(corr_len_array)

#%% plot the heat map of corr_len_array:

f ,axs= plt.subplots(figsize=(5,5))
c=axs.contourf(noiss,rhos,corr_len_array,100, vmin=0, vmax=1.0)
axs.set_xscale("log")
axs.tick_params(axis='y')
axs.tick_params(axis='x',rotation=0)
clb=f.colorbar(c,ax=axs,shrink=0.8)
clb.ax.set_title(r"$\zeta(\bf{p})/L$",size=labelFontSize-5)
clb.set_label(label='Normalised Correlation length '+ r"$(\zeta(\bf{p})/L)$",size=labelFontSize-4)

axs.tick_params(axis='y',which='major',labelsize=tickFontSize-2)
axs.tick_params(axis='x',rotation=0,which='major',labelsize=tickFontSize-2)
# increase the font size labels and ticks
#plt.tick_params(axis='both', which='major', labelsize=tickFontSize)

if noise_type=='StaticNoise':
    axs.set_xlabel("Noise amplitude " +r"$(S)$",fontsize=labelFontSize)
    #plt.text(np.power(-2,10),2.1,'A',weight="bold",fontsize=panelLabelFontSize+4)
    
if noise_type=='DynamicNoise':
    axs.set_xlabel("Noise amplitude " +r"$(\eta)$",fontsize=labelFontSize)
    #plt.text(0.52,1.02,'B',weight="bold",fontsize=panelLabelFontSize+4)
    
axs.set_ylabel("Total protein conc. "+r"$(\rho$)",fontsize=labelFontSize)
    
plt.tight_layout()
plt.savefig(output_folder+filename+".jpeg",dpi=800)