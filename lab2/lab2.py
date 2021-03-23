from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.io import savemat
import os
import sys
import seaborn as sn
from skimage import data, util
from skimage.transform import radon, rescale, iradon

"""A medical imaging script for LAB1's exercises

Authors:
    - Afonso Ferreira
    - Ana Lopes
    - Fábio Almeida
    - Madalena Cerqueira
"""


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** math.ceil(math.log2(x))


# Ex.1 Generate the modified Shepp-Logan phantom using the function
# shepp_logan_phantom, and then using the function rescale to get a 256x256
# dimension.

# generating the Shepp-logan phantom square image with Height x Width = n x n
I = data.shepp_logan_phantom() # ndarray of the phantom
scaling_factor = 256 / I.shape[0]
I = rescale(I, scaling_factor)  # 256 / 400 = 0.64

# Displaying the scaled phantom img
plt.figure()
plt.title("Modified Shepp-Logan phantom")
plt.imshow(I, cmap='gray')
plt.show()

# Creating and showing the histogram of the scaled phantom img
hist_phantom = sn.distplot(I, bins=64)
histogram_occurrences, bin_edges = np.histogram(I, bins=64)



x_ticks = [np.round(bin_edges[i], 4) for i in range(len(bin_edges) - 1) if histogram_occurrences[i] > 1000]

hist_phantom.set_xticklabels(x_ticks, fontdict={'fontsize': 8, 'fontweight': 'bold'}, rotation=35, fontsize=10,
                       linespacing=1.5)
text_str = "Bin size = " + str(np.round(1 / 64, 4))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
hist_phantom.text(0.05, 0.95, text_str, transform=hist_phantom.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.show()

# Ex.2 Simulate the sinogram obtained by collecting projections covering [0;180]° in steps of
# 1° (using the function radon)

theta = np.linspace(0., 180., 181, endpoint=True)
S = radon(I, theta)
plt.figure()
plt.title("Sinogram, [0;180]º, steps of 1º")
plt.imshow(S, cmap=plt.cm.gray)
plt.show()

# Ex.3 Simulate the associated reconstructed image using the inverse Radon transform (using
# the function iradon)

R = iradon(S, theta)
plt.figure()
plt.title("Recontructed image, [0;180]º, steps of 1º")
plt.imshow(R, cmap=plt.cm.gray)
plt.show()

# Ex.4 Repeat the simulations in 2. and 3. by covering: [0;45]°, [0;90]°, [0;180]° and [0;360]°,
# in steps of 1°
show_subplots = True
Sinogram_tensor = []
Reconstruction_tensor = []

for i in range(1, 5):
    j = 45 * (2 ** (i - 1))
    alfa = np.linspace(0., float(j), j + 1, endpoint=True)
    Si = radon(I, alfa)
    print(Si.shape)
    Ri = iradon(Si, alfa)
    if not show_subplots:
        plt.figure()
        plt.title("Sinogram, [0," + str(j) + "º], steps of 1º")
        plt.imshow(Si, cmap="gray")

        plt.figure()
        plt.title("Reconstructed image, [0," + str(j) + "º], steps of 1º")
        plt.imshow(Ri, cmap="gray")
        plt.show()
    else:
        Sinogram_tensor.append(Si)
        Reconstruction_tensor.append(Ri)

# Alternative plotting method
if show_subplots:
    fig_singrs = plt.figure()
    widths = [1, 91/46, 181/46, 361/46]
    heights = [1]
    grid_singrs = fig_singrs.add_gridspec(ncols=4, nrows=1, width_ratios=widths, height_ratios=heights)
    for i in range(0, 4):
        ax = fig_singrs.add_subplot(grid_singrs[0, i])
        plt.imshow(Sinogram_tensor[i], cmap="gray")
        plt.title("Sinogram, [0, {}º]".format(45 * (2 ** (i))), fontsize=10, rotation=30)
    plt.show()

    fig_reconstrs = plt.figure()
    grid_reconstrs = fig_reconstrs.add_gridspec(ncols=4, nrows=1, width_ratios=[1, 1, 1, 1], height_ratios=[1])
    for i in range(0, 4):
        ax = fig_reconstrs.add_subplot(grid_reconstrs[0, i])
        plt.imshow(Reconstruction_tensor[i], cmap="gray")
        plt.title("Reconstruction, [0, {}º]".format(45 * (2 ** (i))), fontsize=10, rotation=30)
    plt.show()

# In order to reconstruct an image, you need 180 degrees of data (* actually 180 + fan beam angle). 
# Why? The remaining 180 degrees are simply a mirror image of the first (because it does not matter
# which way a photon travels through tissue, it will be attenuated the same amount).
# (Because of the fan beam geometry, you need to measure an extra amount - equal to the fan angle
# - to actually get all of the data you need, but the concept is the same.)


# Ex.5 Repeat the simulations in 2. and 3. by covering [0;180]°, in steps of 0.25, 1, 5 and 10°

# Step = 0.25º
omega_025 = np.linspace(0., 180., 724, endpoint=True)  # 181%0.25
S_025 = radon(I, omega_025)
plt.figure()
plt.title("Sinogram, [0;180º], steps of 0.25º")
plt.imshow(S_025, cmap="gray")

R_025 = iradon(S_025, omega_025)
plt.figure()
plt.title("Reconstructed image, [0;180º], steps of 0.25º")
plt.imshow(R_025, cmap="gray")
plt.show()

# Step = 1º
omega_1 = np.linspace(0., 180., 181, endpoint=True)
S_1 = radon(I, omega_1)
plt.figure()
plt.title("Sinogram, [0;180º], steps of 1º")
plt.imshow(S, cmap="gray")

R_1 = iradon(S, omega_1)
plt.figure()
plt.title("Recontructed image, [0;180º], steps of 1º")
plt.imshow(R_1, cmap="gray")
plt.show()

# Step = 5º
omega_5 = np.linspace(0., 180., 36, endpoint=True)  # 181%5
S_5 = radon(I, omega_5)
plt.figure()
plt.title("Sinogram, [0;180º], steps of 5º")
plt.imshow(S_5, cmap="gray")

R_5 = iradon(S_5, omega_5)
plt.figure()
plt.title("Reconstructed image, [0;180º], steps of 5º")
plt.imshow(R_5, cmap="gray")
plt.show()

# Step = 10º
omega_10 = np.linspace(0., 180., 18, endpoint=True)  # 181/18
S_10 = radon(I, omega_10)
plt.figure()
plt.title("Sinogram, [0;180º], steps of 10º")
plt.imshow(S_10, cmap="gray")

R_10 = iradon(S_10, omega_10)
plt.figure()
plt.title("Reconstructed image, [0;180º], steps of 10º")
plt.imshow(R_10, cmap="gray")
plt.show()

# Ex.6 Repeat the simulations in 2. using the original angles, by adding noise to the projection
# data. For this purpose, first scale the sinogram (SS) using maximum number of counts per
# pixel of 103 photons, and then add the appropriate type of noise using the function
# random_noise

Intensity = 10 ** 3
S_scaled = (S * Intensity) / (np.max(np.max(S)))
a = 1 / next_power_of_2(len(np.unique(S_scaled)))
S_noise = util.random_noise(S_scaled * a, 'poisson')  # explicar poisson
plt.figure()
plt.title("Sinogram with noise")
plt.imshow(S_noise, cmap=plt.cm.gray)
# Confirmar se falta algo aqui

# Ex.7 Now reconstruct the image from the noisy projection data using iradon (with the
# original filter, i.e. the Ram-Lak filter)

theta = np.linspace(0., 180., 181, endpoint=True)
R_noise = iradon(S_noise, theta)
plt.figure()
plt.title("Recontructed image with noise, Ram-Lak filter")
plt.imshow(R_noise, cmap=plt.cm.gray)

# Ex.8 Repeat 7, by replacing the original Ram-Lak filter by modified filters (available in
# iradon), and explain the results as a function of their different frequency resp
# ramp, shepp-logan, cosine, hamming, hann

R_Hann = iradon(S_noise, theta, filter_name='hann')
plt.figure()
plt.title("Recontructed image with noise, Hann filter")
plt.imshow(R_Hann, cmap=plt.cm.gray)

R_Shepp_Logan = iradon(S_noise, theta, filter_name='shepp-logan')
plt.figure()
plt.title("Recontructed image with noise, Shepp-Logan filter")
plt.imshow(R_Shepp_Logan, cmap=plt.cm.gray)

R_Ramp = iradon(S_noise, theta, filter_name='ramp')
plt.figure()
plt.title("Recontructed image with noise, Ramp filter")
plt.imshow(R_Ramp, cmap=plt.cm.gray)

R_Cosine = iradon(S_noise, theta, filter_name='cosine')
plt.figure()
plt.title("Recontructed image with noise, Cosine filter")
plt.imshow(R_Cosine, cmap=plt.cm.gray)

R_Hamming = iradon(S_noise, theta, filter_name='hamming')
plt.figure()
plt.title("Recontructed image with noise, Hamming filter")
plt.imshow(R_Hamming, cmap=plt.cm.gray)
