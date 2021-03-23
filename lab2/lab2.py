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


# choose whether or not to display scaled Sinograms and images in the same figure
plots_same_figure = True

# Ex.1 Generate the modified Shepp-Logan phantom using the function
# shepp_logan_phantom, and then using the function rescale to get a 256x256
# dimension.

# generating the Shepp-logan phantom square image with Height x Width = n x n
I = data.shepp_logan_phantom()  # ndarray of the phantom
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
plt.title("Sinogram, [0;180]\N{DEGREE SIGN}, steps of 1\N{DEGREE SIGN}")
plt.imshow(S, cmap="gray")
plt.show()

# Ex.3 Simulate the associated reconstructed image using the inverse Radon transform (using
# the function iradon)

R = iradon(S, theta)
plt.figure()
plt.title("Recontructed image, [0;180]\N{DEGREE SIGN}, steps of 1\N{DEGREE SIGN}")
plt.imshow(R, cmap="gray")
plt.show()

# Ex.4 Repeat the simulations in 2. and 3. by covering: [0;45]°, [0;90]°, [0;180]° and [0;360]°,
# in steps of 1°
Sinogram_tensor = []
Reconstruction_tensor = []

for i in range(1, 5):
    j = 45 * (2 ** (i - 1))
    alfa = np.linspace(0., float(j), j + 1, endpoint=True)
    Si = radon(I, alfa)
    print(Si.shape)
    Ri = iradon(Si, alfa)
    if not plots_same_figure:
        plt.figure()
        plt.title("Sinogram, [0," + str(j) + "\N{DEGREE SIGN}], steps of 1\N{DEGREE SIGN}")
        plt.imshow(Si, cmap="gray")

        plt.figure()
        plt.title("Reconstructed image, [0," + str(j) + "\N{DEGREE SIGN}], steps of 1\N{DEGREE SIGN}")
        plt.imshow(Ri, cmap="gray")
        plt.show()
    else:
        Sinogram_tensor.append(Si)
        Reconstruction_tensor.append(Ri)

# Alternative plotting method
if plots_same_figure:
    fig_singrs = plt.figure()
    widths = [1, 91 / 46, 181 / 46, 361 / 46]
    heights = [1]
    grid_singrs = fig_singrs.add_gridspec(ncols=4, nrows=1, width_ratios=widths, height_ratios=heights)
    for i in range(0, 4):
        ax = fig_singrs.add_subplot(grid_singrs[0, i])
        plt.imshow(Sinogram_tensor[i], cmap="gray")
        plt.title("Sinogram, [0, {}\N{DEGREE SIGN}]".format(45 * (2 ** (i))), fontsize=10, rotation=30)
    plt.show()

    fig_reconstrs = plt.figure()
    grid_reconstrs = fig_reconstrs.add_gridspec(ncols=4, nrows=1, width_ratios=[1, 1, 1, 1], height_ratios=[1])
    for i in range(0, 4):
        ax = fig_reconstrs.add_subplot(grid_reconstrs[0, i])
        plt.imshow(Reconstruction_tensor[i], cmap="gray")
        plt.title("Reconstruction, [0, {}\N{DEGREE SIGN}]".format(45 * (2 ** (i))), fontsize=10, rotation=30)
    plt.show()

# In order to reconstruct an image, you need 180 degrees of data (* actually 180 + fan beam angle). 
# Why? The remaining 180 degrees are simply a mirror image of the first (because it does not matter
# which way a photon travels through tissue, it will be attenuated the same amount).
# (Because of the fan beam geometry, you need to measure an extra amount - equal to the fan angle
# - to actually get all of the data you need, but the concept is the same.)


# Ex.5 Repeat the simulations in 2. and 3. by covering [0;180]°, in steps of 0.25, 1, 5 and 10°

angle_steps = [0.25, 1., 5., 10.]
theta_angle_tensor, S_step_i, R_step_i = [], [], []

for i in range(len(angle_steps)):
    theta_angle_tensor.append(np.linspace(0., 180., int((180 / angle_steps[i]) + 1), endpoint=True))
    S_step_i.append(radon(I, theta_angle_tensor[i]))
    R_step_i.append(iradon(S_step_i[i], theta_angle_tensor[i]))


if not plots_same_figure:

    # Step = 0.25º
    plt.figure()
    plt.title("Sinogram, [0;180\N{DEGREE SIGN}], steps of 0.25\N{DEGREE SIGN}")
    plt.imshow(S_step_i[0], cmap="gray")

    plt.figure()
    plt.title("Reconstructed image, [0;180\N{DEGREE SIGN}], steps of 0.25\N{DEGREE SIGN}")
    plt.imshow(R_step_i[0], cmap="gray")
    plt.show()

    # Step = 1º
    plt.figure()
    plt.title("Sinogram, [0;180\N{DEGREE SIGN}], steps of 1\N{DEGREE SIGN}")
    plt.imshow(S_step_i[1], cmap="gray")

    plt.figure()
    plt.title("Recontructed image, [0;180\N{DEGREE SIGN}], steps of 1\N{DEGREE SIGN}")
    plt.imshow(R_step_i[1], cmap="gray")
    plt.show()

    # Step = 5º
    plt.figure()
    plt.title("Sinogram, [0;180\N{DEGREE SIGN}], steps of 5\N{DEGREE SIGN}")
    plt.imshow(S_step_i[2], cmap="gray")

    plt.figure()
    plt.title("Reconstructed image, [0;180\N{DEGREE SIGN}], steps of 5\N{DEGREE SIGN}")
    plt.imshow(R_step_i[2], cmap="gray")
    plt.show()

    # Step = 10º
    plt.figure()
    plt.title("Sinogram, [0;180\N{DEGREE SIGN}], steps of 10\N{DEGREE SIGN}")
    plt.imshow(S_step_i[3], cmap="gray")

    plt.figure()
    plt.title("Reconstructed image, [0;180\N{DEGREE SIGN}], steps of 10\N{DEGREE SIGN}")
    plt.imshow(R_step_i[3], cmap="gray")
    plt.show()

else:

    fig_singrs_steps = plt.figure()

    widths_steps_singrs = [1, 181 / 721, 37 / 721, 19 / 721]
    heights_steps_singrs = [1]
    grid_singrs_steps = fig_singrs_steps.add_gridspec(ncols=4, nrows=1, width_ratios=widths_steps_singrs,
                                                      height_ratios=heights_steps_singrs)
    fig_singrs_steps.suptitle('Sinograms using a range of angles within [0, 180\N{DEGREE SIGN}] with various steps',
                              fontsize=12)

    fig_reconstrs_steps = plt.figure()
    grid_reconstrs_steps = fig_reconstrs_steps.add_gridspec(ncols=4, nrows=1, width_ratios=[1, 1, 1, 1],
                                                            height_ratios=[1])
    fig_singrs_steps.suptitle(
        'Reconstruction images using a range of angles within [0, 180\N{DEGREE SIGN}] with various '
        'steps',
        fontsize=12)

    for i in range(len(angle_steps)):

        fig_singrs_steps.add_subplot(grid_singrs_steps[0, i])
        fig_singrs_steps.axes[i].imshow(S_step_i[i], cmap="gray")
        fig_singrs_steps.axes[i].set_title("Step of {}\N{DEGREE SIGN}".format(angle_steps[i]), fontsize=10, rotation=30)

        fig_reconstrs_steps.add_subplot(grid_reconstrs_steps[0, i])
        fig_reconstrs_steps.axes[i].imshow(R_step_i[i], cmap="gray")
        fig_reconstrs_steps.axes[i].set_title("Step of {}\N{DEGREE SIGN}".format(angle_steps[i]), fontsize=10,
                                              rotation=30)

    plt.show()

# Ex.6 Repeat the simulations in 2. using the original angles, by adding noise to the projection
# data. For this purpose, first scale the sinogram (SS) using maximum number of counts per
# pixel of 10^3 photons, and then add the appropriate type of noise using the function
# random_noise

Intensity = 10 ** 3
S_scaled = (S_step_i[1] * Intensity) / (np.max(np.max(S_step_i[1])))
a = 1 / next_power_of_2(len(np.unique(S_scaled)))
S_noise = util.random_noise(S_scaled * a, 'poisson')  # explicar poisson

plt.figure()
plt.title("Sinogram with noise")
plt.imshow(S_noise, cmap="gray")
plt.show()
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
