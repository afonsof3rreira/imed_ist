# matplotlib Qt
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.io import savemat
import os
import sys
import seaborn as sn
from skimage import data, util, measure
from skimage.transform import radon, rescale, iradon
from roipoly import RoiPoly, MultiRoi

"""A medical imaging script for LAB1's exercises
Authors:
    - Afonso Ferreira
    - Ana Lopes
    - Fábio Almeida
    - Madalena Cerqueira
"""
# Consider the SPECT imaging of the 2D phantom stored in activity.mat, containing a
# number of hot spots of different sizes, with FOV = 256 x 256 mm2
# , assuming a parallel beam
# geometry, with a rotation increment of 1º and maximum number of photon counts of 2500.
# Please note that, given that a 2D phantom is considered, only 1 of the 2 dimensions of the
# gamma camera is used in this simulation.

# choose whether or not to display scaled Sinograms and images in the same figure
plots_same_figure = True

def normalization(im_arr: np.ndarray):
    return (im_arr - np.min(im_arr)) / (np.max(im_arr) - np.min(im_arr))

# TODO: Exercise 1
"""Load the phantom, get its dimensions, and display it"""
print('-' * 10, 'Loading Exercise 1...')

# getting data path
script_dir = os.path.dirname(sys.argv[0])
spect_mat_dir = os.path.join(script_dir, "data")

# loading 2D SPECT phantom
activity = loadmat(os.path.join(spect_mat_dir, 'activity.mat'))
activity_arr = np.array(activity['activity'])

# printing phantom's shape
[rows, cols] = activity_arr.shape
print('        ->' + ' the phantom\'s dimensions are: (rows x columns) = ' + str(rows) + ' x ' + str(cols) + ' pixels')

# displaying phantom image
print(' ' * 10, 'loading plots...')
plt.figure()
plt.title("original SPECT image")
plt.imshow(activity_arr, cmap="gray")
plt.show()

# TODO: Exercise 2
"""Simulate the sinogram and reconstructed SPECT image of the phantom by filtered
back-projection including noise in your simulation. Make sure to constrain the
reconstructed image to have the same size as the phantom."""
print('-' * 10, 'Loading Exercise 2...')

# creating the set of projection angles (note: 0 = 360º, repeating this angle would lead to repetition)
theta = np.linspace(0., 359., 360, endpoint=True)

# creating the sinogram of the original image
sinogram = radon(activity_arr, theta)

# adding noise to the original image
a=1e-5
sinogram_noisy_arr = 1/a*util.random_noise(sinogram*a, mode='poisson', clip=False)

# scaling the original SPECT image to [0, 2500]
# 2500 = max. photon count (per sensor unit and time interval)
sinogram_noisy_arr = normalization(sinogram_noisy_arr)*2500

# adding poisson noise to the original SPECT image
activity_noise_arr = iradon(sinogram_noisy_arr, theta, filter="ramp") # default filter

print('        ->' + ' the noisy phantom has intensity values from ' + str(np.max(activity_noise_arr)) + ' to ' + str(
    np.min(activity_noise_arr)))

# filter_name (5 filters, ramp = default)
filter_list = ["shepp-logan", "cosine", "hamming", "hann"]

# list containing images for plotting
image_tensor = [activity_arr, activity_noise_arr]

# reconstructed images from noisy sinogram, using the available filters (filtered back-projection)
for i in range(len(filter_list)):
    image_tensor.append(iradon(sinogram_noisy_arr, theta, filter=filter_list[i]))

# list containing the sinograms of the original SPECT and the noisy SPECT
sinogram_tensor = [sinogram, sinogram_noisy_arr]

# creating a list containing titles for plotting
title_list = ["Original SPECT", "Reconstructed SPECT - ramp filter"]

for i in filter_list:
    title_list.append("Reconstructed SPECT - {} filter".format(i))

# rows = (images...)
# len(filters) + 1 (original) = 6

# plotting images and sinograms
if plots_same_figure:

    # images - only original + ramp filter...
    fig_ex2 = plt.figure()
    widths_ex2 = [1, 1]
    heights_ex2 = [1]
    grid_2 = fig_ex2.add_gridspec(ncols=2, nrows=1, width_ratios=widths_ex2, height_ratios=heights_ex2)

    # original
    fig_ex2.add_subplot(grid_2[0, 0])
    plt.imshow(image_tensor[0], cmap="gray")
    plt.title(title_list[0], fontsize=10)

    # reconstructed
    fig_ex2.add_subplot(grid_2[0, 1])
    plt.imshow(image_tensor[1], cmap="gray")
    plt.title(title_list[1], fontsize=10)
    plt.show()

    # images - original + all-filter reconstructions
    fig_ex2_1 = plt.figure()
    widths_ex2 = [1, 1]
    heights_ex2 = [1, 1, 1]
    grid_ex2 = fig_ex2_1.add_gridspec(ncols=2, nrows=3, width_ratios=widths_ex2, height_ratios=heights_ex2)
    im_count = 0

    for i in range(3):  # rows
        for j in range(2):  # columns
            fig_ex2_1.add_subplot(grid_ex2[i, j])
            plt.imshow(image_tensor[im_count], cmap="gray")
            plt.title(title_list[im_count], fontsize=10)
            # if im_count >
            im_count += 1

    plt.subplots_adjust(top=0.953,
                        bottom=0.05,
                        left=0.011,
                        right=0.989,
                        hspace=0.308,
                        wspace=0.0)
    plt.show()

    # sinograms - original and from noisy image
    fig_ex2_2 = plt.figure()
    widths_ex2_2 = [1]
    heights_ex2_2 = [1, 1]
    grid_ex2_2 = fig_ex2_2.add_gridspec(ncols=1, nrows=2, width_ratios=widths_ex2_2, height_ratios=heights_ex2_2)

    # original
    fig_ex2_2.add_subplot(grid_ex2_2[0, 0])
    plt.imshow(sinogram_tensor[0], cmap="gray")
    plt.title("Sinogram of the original SPECT phantom", fontsize=12)
    plt.xlabel('Degrees [\N{DEGREE SIGN}]', fontsize=10)
    plt.ylabel('1D section length [pixels]', fontsize=10)

    # noisy image
    fig_ex2_2.add_subplot(grid_ex2_2[1, 0])
    plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.4, wspace=0.4)
    plt.imshow(sinogram_tensor[1], cmap="gray")
    plt.title("Sinogram of the SPECT phantom after applying noise", fontsize=12)
    plt.xlabel('Degrees [\N{DEGREE SIGN}]', fontsize=10)
    plt.ylabel('1D section length [pixels]', fontsize=10)
    plt.show()

# plt.figure()
# plt.title("Sinogram with noise")
# plt.imshow(activity_noise_arr, cmap='gray')
# plt.show()


# sinogram_2500 = (sinogram*2500)/(np.max(np.max(sinogram)))
# sinogram_noise_2500 = util.random_noise(sinogram_2500, mode ='poisson', clip = False)

print(sinogram.shape)
print(sinogram_noisy_arr.shape)

# plt.figure()
# plt.title("Sinogram with noise")
# plt.imshow(sinogram_noise_2500, cmap='gray')
# plt.show()
#
# reconstructed_arr = iradon(sinogram_noise_2500, theta)
# plt.figure()
# plt.title("reconstructed SPECT image")
# plt.imshow(reconstructed_arr, cmap='gray')

# sinogram_5000=(sinogram*5000)/(np.max(np.max(sinogram)))
# sinogram_noise_5000 = util.random_noise(sinogram_5000, mode ='poisson', clip = False)

# plt.figure()
# plt.title("Sinogram with noise, maximum number of photon counts of 5000 ")
# plt.imshow(sinogram_noise_5000, cmap='gray')
# plt.show()

# reconstructed = iradon (sinogram_noise_5000, theta)
# plt.figure()
# plt.title("reconstructed SPECT image, maximum number of photon counts of 5000")
# plt.imshow(reconstructed, cmap=plt.cm.gray)

# TODO: Exercise 3
"""Define appropriate ROIs for the big, the medium and one of the small hotspots (using
roipoly)."""

# This code was adapted from roipoly's github example
print('-' * 10, 'Loading Exercise 3...')

# show original image
plt.figure()
plt.imshow(image_tensor[1], interpolation='nearest', cmap="gray")
plt.title("Click on the button to add a new ROI")

# enable multi-ROI plotting
multiroi_named = MultiRoi(roi_names=['Big ROI', 'Medium ROI', 'Small ROI'])

plt.imshow(image_tensor[1], interpolation='nearest', cmap="gray")
roi_names = []
roi_masks = []
for name, roi in multiroi_named.rois.items():
    roi.display_roi()
    roi.display_mean(image_tensor[1])
    roi_names.append(name)
    roi_masks.append(roi.get_mask(image_tensor[1]))
plt.legend(roi_names, bbox_to_anchor=(1.2, 1.05))
plt.show()

# TODO: Exercise 4
"""Illustrate and quantify the partial volume effects (PVE’s) suffered by each hot spot by:
a. plotting intensity profiles through the different hot spots; and
b. comparing their average intensities in the phantom (ground truth) and in
the reconstructed image.
c. showing the effects of changing the spatial resolution and/or the SNR.
Note: Make sure to normalize both the phantom and the reconstructed
image (to 1) so that the intensities are comparable between images."""

print('-' * 10, 'Loading Exercise 4...')
# normalizes both images (original and reconstructed)
# np.where(roi_masks[1])
# np.where(roi_masks[2])

norm_activity_arr = normalization(activity_arr)
norm_reconstructed_arr = normalization(image_tensor[1])

# a)
# to check grayscale values in the ROIS

y_1 = measure.profile_line(norm_reconstructed_arr, (27, 63), (59, 63))
x_1 = np.arange(0, len(y_1))

y_2= measure.profile_line(norm_reconstructed_arr, (67, 63), (72, 63))
x_2 = np.arange(0, len(y_2))

y_3 = measure.profile_line(norm_reconstructed_arr, (102, 56), (102, 61))
x_3 = np.arange(0, len(y_3))

# GT intensity profiles for the same ROIs (constant intensity along the axes)
y_1_gt = measure.profile_line(norm_activity_arr, (27, 63), (59, 63))
y_2_gt = measure.profile_line(norm_activity_arr, (67, 63), (72, 63))
y_3_gt = measure.profile_line(norm_activity_arr, (102, 56), (102, 61))

# Averages: noisy
avg_I_noisy = [np.mean(y_1), np.mean(y_2), np.mean(y_3)]

# Averages: original
avg_I_or = [np.mean(y_1_gt), np.mean(y_2_gt), np.mean(y_3_gt)]

plt.figure()
plt.plot(x_1, y_1)
plt.title("Intensity profile - Big ROI", fontsize=10)
plt.xlabel('Pixel index [-]', fontsize=10)
plt.ylabel('Intensity value [-]', fontsize=10)
plt.show()

plt.figure()
plt.plot(x_2, y_2)
plt.title("Intensity profile - Medium ROI", fontsize=10)
plt.xlabel('Pixel index [-]', fontsize=10)
plt.ylabel('Intensity value [-]', fontsize=10)
plt.show()

plt.figure()
plt.plot(x_3, y_3)
plt.title("Intensity profile - Small ROI", fontsize=10)
plt.xlabel('Pixel index [-]', fontsize=10)
plt.ylabel('Intensity value [-]', fontsize=10)
plt.show()

plt.figure()
plt.title("Annotated intensity profile axes for 3 ROIs - noisy phantom")
plt.imshow(norm_reconstructed_arr, cmap="gray")

plt.vlines(x=63, ymin=27, ymax=59, colors='red', ls=':', lw=2,
           label='Big ROI, AVG Intensity = ' + str(np.round(avg_I_noisy[0], 2)))

plt.vlines(x=63, ymin=67, ymax=72, colors='blue', ls=':', lw=2,
           label='Medium ROI, AVG Intensity = ' + str(np.round(avg_I_noisy[1], 2)))

plt.hlines(y=102, xmin=56, xmax=61, colors='green', ls=':', lw=2,
           label='Small ROI, AVG Intensity = ' + str(np.round(avg_I_noisy[2], 2)))

plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
plt.show()


plt.figure()
plt.title("Annotated intensity profile axes for 3 ROIs - original phantom")
plt.imshow(norm_activity_arr, cmap="gray")

plt.vlines(x=63, ymin=27, ymax=59, colors='red', ls=':', lw=2,
           label='Big ROI, AVG Intensity = ' + str(np.round(avg_I_or[0], 2)))

plt.vlines(x=63, ymin=67, ymax=72, colors='blue', ls=':', lw=2,
           label='Medium ROI, AVG Intensity = ' + str(np.round(avg_I_or[1], 2)))

plt.hlines(y=102, xmin=56, xmax=61, colors='green', ls=':', lw=2,
           label='Small ROI, AVG Intensity = ' + str(np.round(avg_I_or[2], 2)))

plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
plt.show()



# average intensities:
    # Noisy
print('        ->' + ' the noisy phantom has an average intensity value of:' + '\n')
print(' ' * 5 + '        ->' + str(avg_I_noisy[0]) + ' along the Big ROI' + '\n')
print(' ' * 5 + '        ->' + str(avg_I_noisy[1]) + ' along the Medium ROI' + '\n')
print(' ' * 5 + '        ->' + str(avg_I_noisy[2]) + ' along the Small ROI' + '\n')
    # Original
print('        ->' + ' the original phantom has an average intensity value of:' + '\n')
print(' ' * 5 + '        ->' + str(avg_I_or[0]) + ' along the Big ROI' + '\n')
print(' ' * 5 + '        ->' + str(avg_I_or[1]) + ' along the Medium ROI' + '\n')
print(' ' * 5 + '        ->' + str(avg_I_or[2]) + ' along the Small ROI' + '\n')
