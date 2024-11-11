# -*- coding: utf-8 -*-
"""
Created on March 30th 2023

@author: divyoj
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def divergence(f):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
    :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
    """
    # num_dims = len(f)
    # return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])


def div_curl_2D(x,y,u,v):
    import numpy as np
    dudx,dudy=np.gradient(u,x,y)[1],np.gradient(u,x,y)[0]
    dvdx,dvdy=np.gradient(v,x,y)[1],np.gradient(v,x,y)[0]

    curl = dvdx - dudy; div = dudx + dvdy
    return (div, curl)



def radial_distribution_function(positions, vectors, box_size, dr):
    import numpy as np
    import numpy.ma as ma
    """
    Calculate the radial distribution function.

    Parameters
    ----------
    positions : array_like, shape (n_particles, n_dimensions)
        The positions of the particles.
    vectors : array_like, shape (n_particles, n_dimensions)
        The vectors at each particle position.
    box_size : array_like, shape (n_dimensions,)
        The size of the box in each dimension.
    dr : float
        The size of the radial bins.

    Returns
    -------
    r : ndarray, shape (n_bins,)
        The centers of the radial bins.
    g : ndarray, shape (n_bins,)
        The value of the radial distribution function at each radial bin.
    """
    n_particles, n_dimensions = positions.shape
   

    # Initialize the radial distribution function
    n_bins = int(np.sqrt(box_size[0]**2+box_size[1]**2) / dr) + 1
    r = np.arange(n_bins) * dr + dr / 2.0
    g = np.zeros(n_bins)
    r_density = np.zeros(n_bins)

    # Calculate all pairwise distances with minimum image convention
    for i in range(n_particles):
        for j in range(i, n_particles):
            rij = positions[i,:] - positions[j,:]
            rij -= np.round(rij / box_size) * box_size
            r_ij = np.sqrt(np.sum(rij**2))


            # Calculate the dot product between the vectors
            dot_product = np.dot(vectors[i,:], vectors[j,:])
            # Add to the histogram
            bin_index = int(np.floor(r_ij / dr))
            g[bin_index] += dot_product
            r_density[bin_index]+=1


    # Normalize the histogram
    g=g/r_density
    g=g/g[0]
    g=ma.fix_invalid(g,fill_value=0)

    return r,g


def radial_distribution_function_vectorised(positions, vectors, box_size, dr):
    import numpy as np
    import numpy.ma as ma
    n_particles, n_dimensions = positions.shape
    n_bins = int(np.sqrt(box_size[0]**2+box_size[1]**2) / dr/2) + 1
    r = np.arange(n_bins) * dr + dr / 2.0
    g = np.zeros(n_bins)
    r_density = np.zeros(n_bins)

    # Calculate all pairwise distances with minimum image convention
    rij = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    rij -= np.round(rij / box_size) * box_size
    r_ij = np.sqrt(np.sum(rij**2, axis=2))

    # Calculate the dot product between the vectors
    dot_product = np.dot(vectors, vectors.T)

    # Calculate the bin indices for each pairwise distance
    bin_indices = (r_ij / dr).astype(int)

    # Add to the histogram
    np.add.at(g, bin_indices.ravel(), dot_product.ravel())
    np.add.at(r_density, bin_indices.ravel(), 1)

    # Normalize the histogram
    g = g / r_density
    g = g / g[0]
    g = ma.fix_invalid(g, fill_value=0)

    return r, g
