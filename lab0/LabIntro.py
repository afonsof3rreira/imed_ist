import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from roipoly import RoiPoly
from scipy.ndimage import gaussian_filter

# 1. Loading 3D image data and getting dimensions and basic stats
x = loadmat('data/brain3D.mat')

# select variable 'im' and convert it from dictionary to numpy array
im = np.array(x['im'])

# print image size
[rows, cols, slices] = im.shape
print(rows, cols, slices)

# 2. Showing images
#plt.figure
#plt.imshow(im)

# 3D images cannot be displayed, only 2D!
# 3. Selecting slices and reshaping matrices
ax_slice = np.reshape(im[:, :, slices//2], [rows, cols])
sag_slice = np.reshape(im[:, cols//2, :], [rows, slices])
cor_slice = np.reshape(im[rows//2, :, :], [cols, slices])

plt.figure('Mid Orthogonal Slices')
plt.subplot(2, 2, 1)
plt.imshow(ax_slice, cmap='gray')
plt.subplot(2, 2, 2)
plt.imshow(sag_slice, cmap='gray')
plt.subplot(2, 2, 3)
plt.imshow(cor_slice, cmap='gray')

# the display range does not seem to be appropriate for the saggital and coronal slices...
# 4. Exploring the image intensity and adjusting image display scale
# to choose a better range, it is useful to look at the histogram

# create the histogram
histogram, bin_edges = np.histogram(sag_slice, bins=64)
# configure and draw the histogram figure
plt.figure()
plt.title("Sagittal Slice - Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.plot(bin_edges[0:-1], histogram)

# Let's try again setting the intensity ranges to [0 - 125]
plt.figure('Saggital Slice - readjusting scale')
plt.subplot(2, 2, 1)
plt.imshow(sag_slice, cmap='gray')
plt.title('without specifying intensity range')
plt.subplot(2, 2, 2)
plt.imshow(sag_slice, cmap='gray', vmin=0, vmax=125)
plt.title('intensity range [0 - 125]')
plt.subplot(2, 2, 3)
plt.imshow(sag_slice, cmap='gray', vmin=0, vmax=10)
plt.title('intensity range [0 - 10]')

# Right, but this brain seems to be turned sideways...
# 5. Rotating 2D images
plt.figure('Saggital Slice - rotating 2D images')
plt.subplot(2, 1, 1)
plt.imshow(sag_slice, cmap='gray', vmin=0, vmax=125)
plt.title('without rotation')
plt.subplot(2, 1, 2)
plt.imshow(np.rot90(sag_slice), cmap='gray', vmin=0, vmax=125)

# That's more like it! What about the other orientations?
plt.figure('Mid Orthogonal Slices - adjusting range and rotating')
plt.subplot(2, 2, 1)
plt.imshow(ax_slice, cmap='gray', origin='lower', vmin=0, vmax=125)
plt.subplot(2, 2, 2)
plt.imshow(np.rot90(sag_slice), cmap='gray', vmin=0, vmax=125)
plt.subplot(2, 2, 3)
plt.imshow(np.rot90(cor_slice), cmap='gray', vmin=0, vmax=125)

# 6. Displaying multiple 2D images in one figure
slice_jump = slices//30

plt.figure('Showing Multiple Slices')
for n in range(1, 30):
    sl = 1 + (n-1)*slice_jump
    axsltmp = np.reshape(im[:, :, sl], [rows, cols])
    plt.subplot(5, 6, n)
    plt.imshow(axsltmp, cmap='gray', origin='lower', vmin=0, vmax=125)

# 7. Defining regions-of-interest (ROI)
plt.figure('Drawing an ROI')
plt.imshow(ax_slice, cmap='gray', origin='lower', vmin=0, vmax=125)
my_roi = RoiPoly(color='r')  # draw new ROI in red color
plt.imshow(ax_slice, cmap='gray', origin='lower', vmin=0, vmax=125)
my_roi.display_roi()
my_roi.display_mean(ax_slice)

# 8. Calculating ROI image statistics
mask = my_roi.get_mask(ax_slice)
plt.imshow(mask)
m = ax_slice[mask].mean()
s = ax_slice[mask].std()
print("mask mean = %.2f , mask stdev=%.2f" % (m, s))

# Let's assume, to simplify, that we can calculate the SNR using the same region as reference.
snr =  m / s
print("SNR = %.2f\" % (snr)")

# 9. Simulating a PSF and manipulating the spatial resolution
# application of a PSF as a Gaussian filter
deltaDirac = np.zeros([11, 11])
deltaDirac[5, 5] = 1

fig = plt.figure()
plt.subplot(1, 2, 1)
psf = gaussian_filter(deltaDirac, sigma=2.0)
plt.imshow(psf, vmin=0, vmax=0.05, cmap='hot')
plt.title('Gaussian Filter sigma=2')
plt.subplot(1, 2, 2)
psf = gaussian_filter(deltaDirac, sigma=4.0)
plt.imshow(psf, vmin=0, vmax=0.05, cmap='hot')
plt.title('Gaussian Filter sigma=4')

plt.figure()
ax1 = plt.subplot(2, 2, 1)
plt.imshow(ax_slice, cmap='gray', origin='lower', vmin=0, vmax=125)
plt.title('Original')
ax2 = plt.subplot(2, 2, 2)
ax_slice_flt = gaussian_filter(ax_slice, sigma=2.0)
plt.imshow(ax_slice_flt, cmap='gray', origin='lower', vmin=0, vmax=125)
plt.title('Gaussian Filtered sigma=2')
ax3 = plt.subplot(2, 2, 3)
ax_slice_flt4 = gaussian_filter(ax_slice, sigma=4.0)
plt.imshow(ax_slice_flt4, cmap='gray', origin='lower', vmin=0, vmax=125)
plt.title('Gaussian Filtered sigma=4')
ax4 = plt.subplot(2, 2, 4)
ax_slice_flt8 = gaussian_filter(ax_slice, sigma=8.0)
plt.imshow(ax_slice_flt8, cmap='gray', origin='lower', vmin=0, vmax=125)
plt.title('Gaussian Filtered sigma=8')

# removing ticks
all_axes = [ax1, ax2, ax3, ax4]
for ax in all_axes:
    ax.set_xticks([])
    ax.set_yticks([])
