from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.io import savemat
import os
import sys

"""A medical imaging script for LAB1's exercises

Authors:
    - Afonso Ferreira
    - Ana Lopes
    - Fábio Almeida
    - Madalena Cerqueira
"""

# Consider the 3D CT image stored in ct.mat, where the image intensity of each pixel (i,j,k)
# corresponds to the CT index, defined as (*) with FOV_xyz = 180 x 180 x 240 mm^3
#
# (*) CT(i,j,k) = (miu(i,j,k) - miu_water)*(1000 / miu_water), where miu_water = 0.206 cm^-1

# TODO: Exercise 1
"""Display a histogram of the CT image intensities (i.e. CT indexes), and then a histogram of the corresponding 
attenuation coefficients, by converting CT indexes to attenuation coefficients (miu) """

# Loading "ct.mat" and assigning to an ndarray
script_dir = os.path.dirname(sys.argv[0])
ct_mat_dir = os.path.join(script_dir, "data")
ct = loadmat(os.path.join(ct_mat_dir, "ct.mat"))
ct_arr = np.array(ct['ct'])

# configure and draw the histogram figure for the original CT image
histogram_ct, bin_edges_ct = np.histogram(ct_arr, bins=64)

plt.figure()
plt.plot(bin_edges_ct[0:-1], histogram_ct)
plt.title("CT 3D image - Grayscale Histogram")
plt.xlabel("grayscale intensity bins")
plt.ylabel("number of pixels")
plt.show()

# normalizing CT's intensities to [0,255]
ct_arr_normalized = ct_arr + np.abs(np.min(ct_arr))
ct_arr_normalized = np.divide(ct_arr_normalized, np.max(ct_arr_normalized))
ct_arr_normalized = np.multiply(ct_arr_normalized, 255)

# configure and draw the histogram figure for the normalized CT image
histogram_normalized, bin_edges_normalized = np.histogram(ct_arr_normalized, bins=64)

plt.figure()
plt.plot(bin_edges_normalized[0:-1], histogram_normalized)
plt.title("CT 3D image - Grayscale Histogram")
plt.xlabel("grayscale intensity bins")
plt.ylabel("number of pixels")
plt.show()

# initializing saving dir string
saving_processed_dir = os.path.join(script_dir, "processed_data")

# converting the CT image entries to attenuation coefficients
# comment/uncomment if there isn't/there's an already saved .mat file
arr_shape = ct_arr.shape
# ----------------------------------------------------------------
# ct_miu_arr = np.zeros(arr_shape)
# miu_water = 0.206
# for i in range(arr_shape[0]):
#     for j in range(arr_shape[1]):
#         for k in range(arr_shape[2]):
#             ct_miu_arr[i, j, k] = ct_arr[i, j, k] * miu_water / 1000 + miu_water
# mdic = {"ct_miu": ct_miu_arr, "label": "IMED"}
# savemat(os.path.join(saving_processed_dir, "ct_miu.mat"), mdic)
# ----------------------------------------------------------------

# loading attenuation CT image if previously saved
ct_miu = loadmat(os.path.join(saving_processed_dir, 'ct_miu.mat'))
ct_miu_arr = np.array(ct_miu['ct_miu'])

# configure and draw the histogram figure for the attenuation CT image
histogram_ct_miu, bin_edges_ct_miu = np.histogram(ct_miu_arr, bins=64)

plt.figure()
plt.plot(bin_edges_ct_miu[0:-1], histogram_ct_miu)
plt.title("CT 3D attenuation image - Grayscale Histogram")
plt.xlabel("grayscale intensity bins")
plt.ylabel("number of pixels")
plt.show()

# comparing histogram scales on the same figure
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(bin_edges_ct[0:-1], histogram_ct)
plt.title("Grayscale Histogram - CT 3D")
plt.subplot(1, 2, 2)
plt.plot(bin_edges_ct_miu[0:-1], histogram_ct_miu)
plt.title("Grayscale Histogram - CT 3D attenuation")
plt.xlabel("grayscale intensity bins")
plt.ylabel("number of pixels")
plt.show()

# TODO: Exercise 2
"""Display 16 representative slices of the image of attenuation coefficients for each
orientation - axial (x,y), sagittal (x,z) and coronal (y,z):
    a. applying rotations when appropriate;
    b. using an adequate intensity scale for tissue visualization;
    c. using an intensity scale that is matched across slices.
"""

# creating 16 evenly spaced values for x,y,z
values_z = np.arange(1, 257, 16)
values_xy = np.arange(1, 513, 32)

# creating a stack of 16 x 2D slices for all planes
axial = np.zeros((16, arr_shape[0], arr_shape[1]))
sagittal = np.zeros((16, arr_shape[0], arr_shape[2]))
coronal = np.zeros((16, arr_shape[1], arr_shape[2]))

# displaying 16 figures for each plane
plt.figure('Axial slices')

for i in range(1, 17):
    axial[i - 1, :, :] = ct_miu_arr[:, :, values_z[i - 1]]
    plt.subplot(4, 4, i)
    plt.imshow(axial[i - 1], cmap='gray', vmin=0, vmax=0.3)
plt.show()

plt.figure('Sagittal slices')

for i in range(1, 17):
    sagittal[i - 1, :, :] = ct_miu_arr[:, values_xy[i - 1], :]
    plt.subplot(4, 4, i)
    plt.imshow(np.rot90(sagittal[i - 1]), cmap='gray', vmin=0, vmax=0.35)
plt.show()

plt.figure('Coronal slices')

for i in range(1, 17):
    coronal[i - 1, :, :] = ct_miu_arr[values_xy[i - 1], :, :]
    plt.subplot(4, 4, i)
    plt.imshow(np.rot90(coronal[i - 1]), cmap='gray', vmin=0, vmax=0.35)
plt.show()

# TODO: Exercise 3
"""Simulate the planar X ray image that would be obtained by projection along x,
assuming that the incident X ray beam has an intensity I0 = 1.000 photons/pixel.
    a. Compute the voxel size along each direction
    b. Write down the attenuation equation, and apply it
    c. Display the resulting projection image

    attenuation equation: I = I_0 * exp(sum(-miu*x) for every voxel along the x axis)
"""

I_0 = 1000

# computing voxel size for all axes
FOV_mm = [18, 18, 24]  # cm x cm x cm
pixel_size = [FOV_mm[0] / ct_arr.shape[0], FOV_mm[1] / ct_arr.shape[1], FOV_mm[2] / ct_arr.shape[2]]

print(pixel_size)

# fix the yz plane and to go through the x axis
x, y, z = ct_arr.shape

# computing the CT attenuation image for the projection along the x axis
# comment/uncomment if there isn't/there's an already saved .mat file
# ----------------------------------------------------------------
# slice_proj_x = np.zeros((y, z))
# for j in range(y):
#     for k in range(z):
#         x_sum = np.sum(ct_miu_arr[:, j, k] * pixel_size[0])
#         slice_proj_x[j, k] = I_0 * math.exp(-x_sum)
#
# mdic_slices = {"proj": slice_proj_x, "label": "IMED"}
# savemat(os.path.join(saving_processed_dir,"slice_proj.mat"), mdic_slices)
# ----------------------------------------------------------------
slice_proj_x = loadmat(os.path.join(saving_processed_dir, 'slice_proj.mat'))
slice_proj_x_arr = np.array(slice_proj_x['proj'])

# displaying projection image along the x axis
plt.figure("X ray projection - x axis")
plt.imshow(np.rot90(slice_proj_x_arr), cmap="gray", vmin=np.min(slice_proj_x_arr),
           vmax=np.max(slice_proj_x_arr))  # , vmin=0, vmax=0.35)
plt.show()

# TODO: Exercise 4
"""Now simulate the planar X ray image that would be obtained by:
    a. using twice the X ray tube voltage: what changed? (hint: look at the histograms!)
    b. projection along y (adjust the image intensity scale in order to better visualize the internal organs)

    I varies with (tube voltage)^2  ==> new_tube_v = 2.previous_tube_v => new_I =4*previous_I
"""

# Getting the projection using 2 x initial tube voltage
# comment/uncomment if there isn't/there's an already saved .mat file
# ----------------------------------------------------------------
# slice_proj_x4 = np.zeros((y, z))
# for j in range(y):
#     for k in range(z):
#         slice_proj_x4[j, k] = slice_proj_x_arr[j, k] * 4
#
# mdic_slices_2 = {"proj": slice_proj_x4, "label": "IMED"}
# savemat(os.path.join(saving_processed_dir,"slice_proj_x4.mat"), mdic_slices_2)
# ----------------------------------------------------------------
slice_proj_x4 = loadmat(os.path.join(saving_processed_dir, 'slice_proj_x4.mat'))
slice_proj_x4_arr = np.array(slice_proj_x4['proj'])

# configure and draw the histogram figures for both images
histogram_slice_final, bin_edges_slice_final = np.histogram(slice_proj_x_arr, bins=64)
histogram_x4, bin_edges_x4 = np.histogram(slice_proj_x4_arr, bins=64)

# side-by-side histogram comparison
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(bin_edges_slice_final[0:-1], histogram_slice_final)
plt.title("Grayscale Histogram - original voltage")
plt.subplot(1, 2, 2)
plt.plot(bin_edges_x4[0:-1], histogram_x4)
plt.title("Grayscale Histogram - 2 x original voltage")
plt.xlabel("grayscale intensity bins")
plt.ylabel("number of pixels")
plt.show()

# overlap histogram comparison
plt.figure()
plt.plot(bin_edges_slice_final[0:-1], histogram_slice_final)
plt.plot(bin_edges_x4[0:-1], histogram_x4)
plt.title("Grayscale Histogram - voltage vs 2 x voltage")
plt.xlabel("grayscale intensity bins")
plt.ylabel("number of pixels")
plt.show()

# computing the CT projection along the y axis

# fix the xz plane
slice_final_along_y = np.zeros((x, z))

# displaying the CT projection along the y axis
for i in range(x):
    for k in range(z):
        y_sum = np.sum(ct_miu_arr[i, :, k] * pixel_size[0])  # certo
        slice_final_along_y[i, k] = I_0 * math.exp(-y_sum)

plt.figure()
plt.imshow(np.rot90(slice_final_along_y), cmap="gray", vmin=30, vmax=500)
plt.show()
