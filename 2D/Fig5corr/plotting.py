# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:51:40 2021

@author: divyoj
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
import geomProperties as geomProperties
import math
import Vector_calculus as vc

sns.set_theme(context='notebook', style='ticks',font='arial')
cm = 1/2.54  # centimeters in inches
frac=1

def movie_for_delta_p(t_span,centerx,centery,v,cells,Lx,Ly,ux_over_time,uy_over_time,boundary_cells_remove,u_name="u",filename="",folder="./",normalisation=True,grid=True):
    grid=True
    u_over_time=np.hypot(ux_over_time, uy_over_time)
    
    if normalisation==True:    
        ux_over_time=ux_over_time/u_over_time;
        uy_over_time=uy_over_time/u_over_time;
    
        
    def plot_quiver(centerx,centery,ux_at_t,uy_at_t,u_at_t, t):
        # Clear the current plot figure
        plt.clf()
    
        plt.title(f"{u_name} at t = {t:.3f} unit")
        plt.xlabel("x")
        plt.ylabel("y")
        
        if grid==True:
            for i in range(len(cells)):
                if i in boundary_cells_remove:
                    continue
                xv = v[cells[i],0];xv = np.append(xv,xv[0]) # to close the polygon
                yv = v[cells[i],1];yv = np.append(yv,yv[0]) # to close the polygon
                plt.plot(xv,yv,'k',linewidth=0.2);
        # This is to plot u_at_u_at_t (u at time-step u_at_t)
        
        plt.quiver(centerx,centery,ux_at_t,uy_at_t,scale=20,headwidth=5,pivot="mid")
        sc=plt.scatter(centerx,centery,c=u_at_t, cmap='coolwarm',alpha=0.2,marker="h", s=200,vmin=0, vmax=6)
        clb=plt.colorbar(sc,shrink=0.8)
        clb.ax.set_title(r"$|\bf{p}|$ ")
        plt.xlim(-(Lx/2)*frac, (Lx/2)*frac)
        plt.ylim(-(Ly/2)*frac, (Ly/2)*frac)
        plt.gca().set_aspect('equal')
        
        return plt
    
    def animate(t):
        
        plot_quiver(centerx,centery,ux_over_time[t],uy_over_time[t],u_over_time[t], t)
    
    anim = animation.FuncAnimation(plt.figure(),animate,interval=200,frames=np.size(ux_over_time,0))
    anim.save(folder+filename+".gif",dpi=200)
    plt.close()

def movie_for_angle(t_span,centerx,centery,v,cells,Lx,Ly,ux_over_time,uy_over_time,boundary_cells_remove,u_name="u",filename="",folder="./",normalisation=True,grid=True):
    grid=True
    u_over_time=np.hypot(ux_over_time, uy_over_time)
    
    if normalisation==True:    
        ux_over_time=ux_over_time/u_over_time;
        uy_over_time=uy_over_time/u_over_time;
    
        
    def plot_quiver(centerx,centery,ux_at_t,uy_at_t,u_at_t, t):
        # Clear the current plot figure
        plt.clf()
    
        plt.title(f"{u_name} at t = {t:.3f} min")
        plt.xlabel("x")
        plt.ylabel("y")
        
        angle=np.arctan2(uy_at_t, ux_at_t)       
        
        if grid==True:
            for i in range(len(cells)):
                if i in boundary_cells_remove:
                    continue
                xv = v[cells[i],0];xv = np.append(xv,xv[0]) # to close the polygon
                yv = v[cells[i],1];yv = np.append(yv,yv[0]) # to close the polygon
                plt.plot(xv,yv,'k',linewidth=0.2);
        # This is to plot u_at_u_at_t (u at time-step u_at_t)
        
        plt.quiver(centerx,centery,ux_at_t,uy_at_t,scale=20,headwidth=5,pivot="mid")
        sc=plt.scatter(centerx,centery,c=angle, cmap='hsv',alpha=0.2,marker="h", s=800)
        clb=plt.colorbar(sc,shrink=0.8)
        clb.ax.set_title(r"$ \arctan(\bf{p})$ ")
        plt.xlim(-(Lx/2)*frac, (Lx/2)*frac)
        plt.ylim(-(Ly/2)*frac, (Ly/2)*frac)
        plt.gca().set_aspect('equal')
        
        return plt
    
    def animate(t):
        
        plot_quiver(centerx,centery,ux_over_time[t],uy_over_time[t],u_over_time[t], t)
    
    anim = animation.FuncAnimation(plt.figure(),animate,interval=200,frames=np.size(ux_over_time,0))
    anim.save(folder+filename+".gif",dpi=200)
    plt.close()

def movie_with_fixed_arrows(t_span,centerx,centery,v,cells,Lx,Ly,ux_over_time,uy_over_time,boundary_cells_remove,u_name="u",filename="",folder="./",normalisation=True,grid=True):
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    u_over_time=np.hypot(ux_over_time, uy_over_time)
    
    if normalisation==True:    
        ux_over_time=ux_over_time/u_over_time;
        uy_over_time=uy_over_time/u_over_time;
    
    ## Creating figure and setting the limits and axis  titles  
    fig, ax = plt.subplots(1,1)
    ax.set(xlabel ='X-Axis', ylabel ='Y-Axis', xlim =(-(Lx/2)*frac, (Lx/2)*frac), ylim =(-(Ly/2)*frac, (Ly/2)*frac))
    ## initialising the quiver plot:
    Q = ax.quiver(centerx, centery, ux_over_time[0], uy_over_time[0],u_over_time[0],cmap='plasma',scale=30)
    ## divding axes for the color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ## setting colorbar
    cb = fig.colorbar(Q, cax=cax)
    
    ## condition to be set to 1 if you want grid
    if grid==True:
       for i in range(len(cells)):
        
        if i in boundary_cells_remove:
            continue
        
        xv = v[cells[i],0];xv = np.append(xv,xv[0]) # to close the polygon
        yv = v[cells[i],1];yv = np.append(yv,yv[0]) # to close the polygon
        ax.plot(xv,yv,'k',linewidth=0.2,zorder=0)

    def update_quiver(num, Q, ux_over_time, uy_over_time):
        """updates the horizontal and vertical vector components according to ux,uy and u at the time point """
    
        U = ux_over_time[num];  V = uy_over_time[num]
        C = u_over_time[num]
        
        ## limites for colorbar
        vmin=np.max(C); vmax=np.max(C)
        ## update the values in quiver plot
        Q.set_UVC(U,V,C)
        ## colorbar limits
        Q.set_clim(vmin, vmax)
        ## time point
        t=t_span[num-1]
        ## title displays the time and quantity being plotted
        ax.set_title(f"{u_name} at t = {t:.3f} min")
        return Q,

    # you need to set blit=False, or the first set of arrows never gets cleared on subsequent frames
    
    anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, ux_over_time, ux_over_time),
                                   interval=200,frames=np.size(ux_over_time,0), blit=False )
   
    anim.save(folder+filename+"_"+".gif",dpi=200)
    plt.close()

def movies_at_rho(t_span,a_t,b_t,v,cells,nbs,Lx,Ly,filename,folder,make_movies=False,plots=False):
    import numpy.ma as ma
    
    centerx,centery = geomProperties.cellCenters(v,cells)
    
    # # #%% Boundary cells and deletion cellls: 
    boundary_cells_remove=[];
    # for i in range(np.size(a_t,2)):
    #     if ( ( abs(centerx[i]) > (Lx/2)*frac) or ( abs(centery[i]) > (Ly/2)*frac) ) :
    #          boundary_cells_remove.append(i)  
    
    # zero_one_array=np.zeros(np.shape(a_t));
    # zero_one_array[:,:,boundary_cells_remove] =1
    # a_t = ma.masked_array(a_t, mask=zero_one_array)
    # b_t = ma.masked_array(b_t, mask=zero_one_array)
    
    
    # t_range=np.size(a_t,0);no_of_components=np.size(a_t,1);no_of_cells=np.size(a_t,2)
    # #%% Calculating polaritity components:
    # pax=np.zeros((t_range,no_of_cells));pbx=pax*1;pay=pax*1;pby=pax*1;
    # cos_array=np.cos(np.linspace(0, 2*np.pi,no_of_components,endpoint=False))
    # sin_array=np.sin(np.linspace(0, 2*np.pi,no_of_components,endpoint=False))

    # cos_array=(1/2*np.pi)*np.tile(cos_array,(no_of_cells,1)).transpose()
    # sin_array=(1/2*np.pi)*np.tile(sin_array,(no_of_cells,1)).transpose()
    # for t in range(t_range):
    #     pax[t] = sum(a_t[t,:,:]*cos_array)
    #     pay[t] = sum(a_t[t,:,:]*sin_array)
    #     pbx[t] = sum(b_t[t,:,:]*cos_array)
    #     pby[t] = sum(b_t[t,:,:]*sin_array)

    t_range = a_t.shape[0]
    no_of_cells = a_t.shape[2]

    # Create the cos_array and sin_array with shape (no_of_components, no_of_cells)
    cos_array = (1 / (2 * np.pi)) * np.cos(np.linspace(0, 2 * np.pi, a_t.shape[1], endpoint=False))
    sin_array = (1 / (2 * np.pi)) * np.sin(np.linspace(0, 2 * np.pi, a_t.shape[1], endpoint=False))
    cos_array = np.tile(cos_array, (no_of_cells, 1)).T
    sin_array = np.tile(sin_array, (no_of_cells, 1)).T

    # Calculate pax, pay, pbx, and pby in a vectorized manner
    pax = np.sum(a_t * cos_array, axis=(1, 2))
    pay = np.sum(a_t * sin_array, axis=(1, 2))
    pbx = np.sum(b_t * cos_array, axis=(1, 2))
    pby = np.sum(b_t * sin_array, axis=(1, 2))
    
    
    # #%% Calculations of polarities
    #Pa = np.hypot(pax, pay); Pb = np.hypot(pbx, pby)
    average_px= pax+pbx;average_py= pay+pby; #average_p=np.hypot(average_px, average_py)
    delta_px= pax-pbx;delta_py= pay-pby;#delta_p=np.hypot(delta_px, delta_py)
    if make_movies==True:
        #%% Making movies:
        movie_for_delta_p(t_span,centerx,centery,v,cells,Lx,Ly,delta_px,delta_py,boundary_cells_remove,
                           r"$\bf{p}$",filename=filename+"_delta_p",folder=folder)
    
        #movie_for_angle(t_span,centerx,centery,v,cells,Lx,Ly,delta_px,delta_py,boundary_cells_remove,
        #                   r"$angle(\Delta P)$",filename=str(filename+"_angle_delta_p"),folder=folder)
    time_series=False
    if time_series==True:
        # Plotting
        f, axs = plt.subplots(2,2)        
        
        # ## Pa
        axs[0,0].set_title("Pax over time")
        axs[0,0].plot(pax);
        
        # ## Pb
        axs[0,1].set_title("Pbx over time")
        axs[0,1].plot(pbx);
        
        # ## Pa+Pb
        axs[1,0].set_title("Pay over time")
        axs[1,0].plot(pay);
        
        # ## Pa-Pb
        axs[1,1].set_title("Pby over time")
        axs[1,1].plot(pby);
                    
        
        f.subplots_adjust(top=0.90, bottom=0.10, left=0.15, right=0.95, hspace=0.5,wspace=0.5)
        f.suptitle(filename)
        f.savefig(folder+filename+"_plots_over_time"+".png",dpi=500)    
        plt.close()
    
    correlation_length,correlation_length_fluc=new_plot_spatial_correlation(centerx,centery,delta_px,delta_py,Lx,Ly,filename,folder,plots=False)
   
    return (correlation_length,correlation_length_fluc)
    
    
        
def plots_at_rho(a_t,b_t,v,cells,nbs,Lx,Ly,filename,folder,rho,plots=False):
    import numpy.ma as ma
    sns.set_style("white")
    
    centerx,centery = geomProperties.cellCenters(v,cells)
        
    t_max=np.size(a_t,0);no_of_components=np.size(a_t,1);no_of_cells=np.size(a_t,2)
    #% Calculation of pax,pay,pbx,pby
    pax=np.zeros((no_of_cells));pbx=np.copy(pax);pay=np.copy(pax);pby=np.copy(pax);
    for j in range(no_of_components):
        pax = pax + (1/2*np.pi)*math.cos((j*2*np.pi)/no_of_components)*a_t[t_max-1][j]; 
        pay = pay + (1/2*np.pi)*math.sin((j*2*np.pi)/no_of_components)*a_t[t_max-1][j]; 
        pbx = pbx + (1/2*np.pi)*math.cos((j*2*np.pi)/no_of_components)*b_t[t_max-1][j]; 
        pby = pby + (1/2*np.pi)*math.sin((j*2*np.pi)/no_of_components)*b_t[t_max-1][j];
    a0 = np.sum(a_t[t_max-1], axis=0);
    #% Boundary cells and deletion cellls: 
    boundary_cells_remove=[];
    for i in range(np.size(a_t,2)):
        if ( ( abs(centerx[i]) > (Lx/2)*frac) or ( abs(centery[i]) > (Ly/2)*frac) ) :
            
             boundary_cells_remove.append(i) 
            
    zero_one_array=np.zeros(np.shape(pax));zero_one_array[boundary_cells_remove] =1;
    pax = ma.masked_array(pax, mask=zero_one_array);pbx = ma.masked_array(pbx, mask=zero_one_array)
    pay = ma.masked_array(pay, mask=zero_one_array);pby = ma.masked_array(pby, mask=zero_one_array)
    a0 = ma.masked_array(a0, mask=zero_one_array);
    
    
    #%% Calculations of polarities
    Pa = np.hypot(pax, pay); Pb = np.hypot(pbx, pby)
    average_px= pax+pbx;average_py= pay+pby; average_p=np.hypot(average_px, average_py)
    delta_px= pax-pbx;delta_py= pay-pby;delta_p=np.hypot(delta_px, delta_py)
    
    
    # Helmholtz Decomposition
    div_count,curl_count=plot_helmholtz_decomposition(centerx,centery,delta_px,delta_py,delta_p,Lx,Ly,filename,folder,plots=False)
    
    if plots==True: 
        #%% Steady_state_plot_for_ delta p
        f, ax = plt.subplots(figsize=(10,8)) 
        ## Grid
        for i in range(len(cells)):
            if i in boundary_cells_remove:
                continue
            
            else:
                
                xv = v[cells[i],0];xv = np.append(xv,xv[0]) # to close the polygon
                yv = v[cells[i],1];yv = np.append(yv,yv[0]) # to close the polygon
                ax.plot(xv,yv,'k',linewidth=0.5);
        
        #Delta_p
        #q=ax.quiver(centerx,centery,delta_px/delta_p,delta_py/delta_p,delta_p,cmap='coolwarm',scale=25,headwidth=5)
        ax.quiver(centerx,centery,delta_px/delta_p,delta_py/delta_p,scale=20,headwidth=5,pivot="mid")
        sc=ax.scatter(centerx,centery,c=delta_p, cmap='coolwarm',alpha=0.4,marker="h", s=800)#,vmin=0, vmax=6
        ax.set(xlim =(-(Lx/2)*frac, (Lx/2)*frac), ylim =(-(Ly/2)*frac, (Ly/2)*frac))
        clb=f.colorbar(sc,ax=ax,shrink=0.8)
        clb.ax.set_title(r"$|\bf{p}|$")
        ax.set_aspect("equal")
        f.savefig(folder+filename+"_p"+".png",dpi=800)    
        plt.close()
        
    #  #%% Steady_state_plot_for Angle for delta_p
    # f, ax = plt.subplots(figsize=(10,8)) 
    # ## Grid
    # for i in range(len(cells)):
    #     if i in boundary_cells_remove:
    #         continue
        
    #     else:
            
    #         xv = v[cells[i],0];xv = np.append(xv,xv[0]) # to close the polygon
    #         yv = v[cells[i],1];yv = np.append(yv,yv[0]) # to close the polygon
    #         ax.plot(xv,yv,'k',linewidth=0.5);
    
    # #%% Angle for delta_p
    # angle=np.arctan2(delta_py, delta_px)  
    # ax.quiver(centerx,centery,delta_px/delta_p,delta_py/delta_p,scale=20,headwidth=5,pivot="mid")
    # sc=ax.scatter(centerx,centery,c=angle, cmap='hsv',alpha=0.4,marker="h", s=800)
    # ax.set(xlim =(-(Lx/2)*frac, (Lx/2)*frac), ylim =(-(Ly/2)*frac, (Ly/2)*frac))
    # clb=f.colorbar(sc,ax=ax,shrink=0.8)
    # clb.ax.set_title(r"$\arctan(\Delta P)$ ")
    # ax.set_aspect("equal")
    # f.savefig(folder+filename+"_angle_p"+".png",dpi=800)    
    # plt.close()
    
    #% Returning the mean values:
    return (div_count,curl_count)


def new_plot_spatial_correlation(centerx,centery,delta_px,delta_py,Lx,Ly,filename,folder,plots=False):
    #% Plotting spatial correlation (delta_p)
    import Vector_calculus as vc   
    tmax=np.size(delta_px,0)
    t_range=np.arange(int(0.8*tmax),tmax,int(0.01*tmax))
    lattice_constant=np.sqrt((centery[1]-centery[0])**2+(centerx[1]-centerx[0])**2)
    box_size = np.array([Lx,Ly])
    dr = np.round(lattice_constant,2)
    positions = np.array([centerx,centery]).T
    n_bins = int(np.sqrt(Lx**2+Ly**2) / dr/2) + 1
    g_t_array=np.empty((len(t_range),n_bins))
    g_fluc_t_array=np.empty((len(t_range),n_bins))

    for ti,t in enumerate( t_range):
      
        vectors = np.array([delta_px[t],delta_py[t]]).T
        fluctuations=vectors-np.mean(vectors,axis=0)
            
        # Calculate the radial distribution function
        r, g = vc.radial_distribution_function_new(positions, vectors, box_size, dr)
        # r, g_fluc = vc.radial_distribution_function_new(positions, fluctuations, box_size, dr)
        g_t_array[ti]=g
        # g_fluc_t_array[ti]=g_fluc

    
    g_array=np.mean(g_t_array,axis=0);
    g_fluc_array=np.mean(g_fluc_t_array,axis=0)
    index_of_edge = np.where(r >= min(box_size)/2)[0][0]
    correlation_length=np.trapz(g_array[:index_of_edge],x=r[:index_of_edge]);
    correlation_length_fluc=np.trapz(g_fluc_array[:index_of_edge],x=r[:index_of_edge]);

    if plots==True:
        # Plot spatial correlation
        fig, ax = plt.subplots(2,1,figsize=(4,6))

        ax[0].plot(r,g_array,'.-')
        ax[0].set_xlabel('Distance (r)')
        ax[0].set_ylabel('Correlation of '+ r"$\bf{p}$")

        ax[1].plot(r,g_fluc_array,'.-')
        ax[1].set_xlabel('Distance (r)')
        ax[1].set_ylabel('Correlation of '+ r"$\delta\bf{p}$")

        fig.subplots_adjust(top=0.90, bottom=0.10, left=0.15, right=0.95, hspace=0.5,wspace=0.5)
        fig.savefig(folder+filename+"_correlation"+".png",dpi=500)    
        plt.close()


    return (correlation_length,correlation_length_fluc)

def plot_helmholtz_decomposition(x,y,px,py,delta_p,Lx,Ly,filename,folder,plots=False):
    from scipy.interpolate import griddata
    from scipy.interpolate import RegularGridInterpolator

    ngridx=ngridy=1*len(x) 
    points = np.vstack((x,y)).T   
    # Create grid values first.
    xi=np.linspace(-(Lx/2)*frac, (Lx/2)*frac, ngridx)
    yi=np.linspace(-(Ly/2)*frac, (Ly/2)*frac, ngridy)
    Xi, Yi = np.meshgrid(xi,yi)
    
    pxs = griddata(points, px, (Xi,Yi), method='linear') 
    pys = griddata(points, py, (Xi,Yi), method='linear')

    divp, curlp = vc.div_curl_2D(xi,yi,pxs,pys)
    divpi=RegularGridInterpolator((xi,yi),divp)

    div_count=np.mean(divp**2>1.5)
    curl_count=np.mean(curlp**2>1.5)     
    
    if plots==True:
        f, ax = plt.subplots(figsize=(10,8))
        ax0 = ax.contourf(divp**2,100,cmap='coolwarm'); #Xi,Yi,
        #ax.quiver(x,y,px,py); #ax[0].quiver(Xi,Yi,pxs,pys,color='red'); 
        ax.set_title(r"$(\nabla\cdot P)^2$" )
        #f.colorbar(ax0,ax=ax,shrink=0.8, label=r"$(\nabla\cdot P)^2$ ")
        ax.set(xlim =(-(Lx/2)*frac, (Lx/2)*frac), ylim =(-(Ly/2)*frac, (Ly/2)*frac))
        ax.set_aspect("equal")
        f.subplots_adjust(top=0.90, bottom=0.10, left=0.15, right=0.95, hspace=0.5,wspace=0.5)
        f.savefig(folder+filename+"_div_contour"+".png",dpi=500)    
        plt.close()
        
        f, ax = plt.subplots(figsize=(10,8))
        ax1 = ax.contourf(Xi,Yi,curlp**2,100,cmap='coolwarm');
        #ax.quiver(x,y,px,py); #ax[1].quiver(Xi,Yi,pxs,pys,color='red'); 
        ax.set_title(r"$(\nabla\times P)^2$")
        
        f.colorbar(ax1,ax=ax,shrink=0.8, label=r"$(\nabla\times P)^2$ ")
        ax.set(xlim =(-(Lx/2)*frac, (Lx/2)*frac), ylim =(-(Ly/2)*frac, (Ly/2)*frac))
        ax.set_aspect("equal")
        f.subplots_adjust(top=0.90, bottom=0.10, left=0.15, right=0.95, hspace=0.5,wspace=0.5)
        f.savefig(folder+filename+"_curl_contour"+".png",dpi=500)    
        plt.close()

    return (div_count,curl_count)


def plot_phase_diagram(data2Darray,index_array,column_array,filename,symbol,folder):
    f, ax = plt.subplots(figsize=(10*cm,7*cm)) 
    c=ax.pcolormesh(column_array,index_array,data2Darray,cmap='coolwarm',shading='auto')
    ax.set_xscale("log")
    ax.tick_params(axis='y')
    ax.tick_params(axis='x',rotation=0)
    clb=f.colorbar(c,ax=ax,shrink=0.8,label=filename+symbol)

    ax.set_xlabel("Noise amplitude " +r"$(S)$")
    ax.set_ylabel("Total protein conc. "+r"$(\rho$)")
    ax.set_xlim(min(column_array),max(column_array))
        
    plt.subplots_adjust(top=0.98, bottom=0.20, left=0.20, right=0.90, hspace=0.25,wspace=0.5)
    
    plt.savefig(folder+"phase_diagram"+filename+".jpeg",dpi=500)   
