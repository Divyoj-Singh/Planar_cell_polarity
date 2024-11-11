import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

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

def calcPolarity(ux,uy,pa,pb,dl):
    uxi = ux[0,:]
    uyi = uy[0,:]

    ths = np.arctan2(uyi,uxi); 
    ths[ths<0] += 2*np.pi;
    ths = np.append(ths,ths[0])

    (m,n) = np.shape(pa)
    ths2 = np.tile(ths,(m,1))

    pa0 = np.reshape(pa[:,0],(NX*NY,1))
    pac = np.hstack((pa,pa0))

    pb0 = np.reshape(pb[:,0],(NX*NY,1))
    pbc = np.hstack((pb,pb0))

    vxa = np.trapz(pac*np.cos(ths2),dx=dl,axis=1)/(6*dl)
    vya = np.trapz(pac*np.sin(ths2),dx=dl,axis=1)/(6*dl)
#    va = np.sqrt(vxa*vxa + vya*vya)

    vxb = np.trapz(pbc*np.cos(ths2),dx=dl,axis=1)/(6*dl)
    vyb = np.trapz(pbc*np.sin(ths2),dx=dl,axis=1)/(6*dl) 
#    vb = np.sqrt(vxb*vxb + vyb*vyb)

    vx = vxa - vxb
    vy = vya - vyb

    ang = np.arctan2(vy,vx)
    ang[ang<0] += 2*np.pi;

    cx = np.mean(ux,axis=1)
    cy = np.mean(uy,axis=1)      

    return (cx,cy,vx,vy,ang) 

def time_averaged_spatial_correlation(centerx,centery,px,py,Lx,Ly,filename,folder,plots=False):
    #% Plotting spatial correlation (delta_p)
    import Vector_calculus as vc   
    tmax=np.size(px,0)
    t_range=np.arange(int(0.8*tmax),tmax,int(0.01*tmax))
    lattice_constant=np.sqrt((centery[1]-centery[0])**2+(centerx[1]-centerx[0])**2)
    box_size = np.array([Lx,Ly])
    dr = np.round(lattice_constant,2)
    positions = np.array([centerx,centery]).T
    n_bins = int(np.sqrt(Lx**2+Ly**2) / dr/2) + 1
    g_t_array=np.empty((len(t_range),n_bins))

    for ti,t in enumerate( t_range):
        vectors = np.array([px[t],py[t]]).T
        # Calculate the radial distribution function
        r, g = vc.radial_distribution_function_new(positions, vectors, box_size, dr)
        g_t_array[ti]=g
    
    g_array=np.mean(g_t_array,axis=0);
    index_of_edge = np.where(r >= min(box_size)/2)[0][0]
    correlation_length=np.trapz(g_array[:index_of_edge],x=r[:index_of_edge]);
    
    if plots==True:
        # Plot spatial correlation
        fig, ax = plt.subplots(figsize=(4,3))

        ax.plot(r,g_array,'.-')
        ax.set_xlabel('Distance (r)')
        ax.set_ylabel('Correlation of '+ r"$\bf{p}$")

        fig.subplots_adjust(top=0.90, bottom=0.15, left=0.15, right=0.95, hspace=0.5,wspace=0.5)
        fig.savefig(folder+filename+"_correlation"+".png",dpi=500)    
        plt.close()


    return (correlation_length)

def spatial_correlation(centerx,centery,px,py,Lx,Ly,filename,folder,plots=False):
    #% Plotting spatial correlation (delta_p)
    import Vector_calculus as vc   
    lattice_constant=np.sqrt((centery[1]-centery[0])**2+(centerx[1]-centerx[0])**2)
    box_size = np.array([Lx,Ly])
    dr = np.round(lattice_constant,2)
    positions = np.array([centerx,centery]).T
    n_bins = int(np.sqrt(Lx**2+Ly**2) / dr/2) + 1
    vectors = np.array([px,py]).T
    # Calculate the radial distribution function
    r, g = vc.radial_distribution_function_vectorised(positions, vectors, box_size, dr)
    
    g_array=g
    index_of_edge = np.where(r >= min(box_size)/2)[0][0]
    correlation_length=np.trapz(g_array[:index_of_edge],x=r[:index_of_edge]);
    norm_corr_len=correlation_length
    
    if plots==True:
        # Plot spatial correlation
        fig, ax = plt.subplots(figsize=(4,3))

        ax.plot(r,g_array,'.-')
        ax.set_xlabel('Distance (r)')
        ax.set_ylabel('Correlation of '+ r"$\bf{p}$")

        fig.subplots_adjust(top=0.90, bottom=0.15, left=0.15, right=0.95, hspace=0.5,wspace=0.5)
        fig.savefig(folder+filename+"_correlation"+".png",dpi=500)    
        plt.close()


    return (norm_corr_len)

if __name__ == "__main__":
    
    alp = 1.0
    NX = 20; NY = 20
    import matplotlib.cm as cm
    import matplotlib as mpl
    noise_type = 'StaticNoise' 
    #noise_type = 'DynamicNoise'
    input_folder = "./data" + noise_type + "/" 
    output_folder = "./results_" + noise_type + "/"
    # delete the files inside the folder:
    import os, shutil

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ni =1 
    rhos = np.linspace(1.2,2.2,20)
    if noise_type == 'StaticNoise':
        noiss = np.logspace(-3,-0.5,20)
    if noise_type == 'DynamicNoise':
        noiss = np.logspace(-3,0,20)
    corr_len_array=np.zeros((len(rhos),len(noiss)))
    for rho_i,rho in enumerate(rhos):
        for noise_i,stnois in enumerate(noiss):
            filename='fig5CData'+str(rho)+'_'+str(stnois)+'_'+str(ni)
            with open(input_folder+filename+'.pkl', 'rb') as input_file:
                [ux,uy,pa,pb,dl,ux2,uy2,nbs2,dl2] = pkl.load(input_file) 

            (cx,cy,vx,vy,ang) = calcPolarity(ux,uy,pa,pb,dl)
            
            # cx, cy contain the cell centers
            # vx, vy are the compoennts of the polarity vector 
            Lx = max(cx)-min(cx); Ly = max(cy)-min(cy)
            norm_corr_len = spatial_correlation(cx,cy,vx,vy,Lx,Ly,filename,output_folder,plots=False)
            
            corr_len_array[rho_i,noise_i]=norm_corr_len

    # fig, ax = plt.subplots(figsize=(4,3))
    # ax.plot(noiss,corr_len_array,'.-')
    # ax.set_xlabel('Noise amplitude '+r"$\eta$")
    # ax.set_ylabel('Normalised Correlation length '+ r"$\eta(\bf{p})/L$")
    # ax.set_xscale('log')

    # fig.subplots_adjust(top=0.90, bottom=0.15, left=0.15, right=0.95, hspace=0.5,wspace=0.5)
    # fig.savefig(output_folder+"fig5C_correlation"+".png",dpi=500)
    # plt.close()
    filename= "phase_diagram_" + noise_type    
    data_file=output_folder+ "data_for_" +filename
    np.savez(data_file,corr_len_array=corr_len_array,noise_amp_array=noiss,rho_array=rhos)  
    

    #%% Normalise the correlation length with respect to aligned state:
    print("Max coorelation length is: ",np.max(corr_len_array))
    corr_len_array=corr_len_array/np.max(corr_len_array)
    
    #%% plot the heat map of corr_len_array:
    
    f, ax = plt.subplots(figsize=(8,6) )
    c=ax.contourf(noiss,rhos,corr_len_array,100, vmin=0, vmax=1.0)
    ax.set_xscale("log")
    ax.tick_params(axis='y')
    ax.tick_params(axis='x',rotation=0)
    clb=f.colorbar(c,ax=ax,shrink=0.8,label='Normalised Correlation length '+ r"$(\zeta(\bf{p})/L)$")
    clb.ax.set_title(r"$\zeta(\bf{p})/L$")
    
    if noise_type=='StaticNoise':
        ax.set_xlabel("Noise amplitude " +r"$(S)$")
    if noise_type=='DynamicNoise':
        ax.set_xlabel("Noise amplitude " +r"$(\eta)$")

    ax.set_ylabel("Total protein conc. "+r"$(\rho$)")
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        
    plt.subplots_adjust(top=0.98, bottom=0.15, left=0.15, right=0.95, hspace=0.25,wspace=0.5)

    plt.savefig(output_folder+filename+"2.jpeg",dpi=500)
