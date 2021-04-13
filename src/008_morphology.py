# %% Import modules
from dip import *

# %% Example
img = cv2.imread(os.path.join(folder, 'text.tif'), cv2.IMREAD_GRAYSCALE)
kernel = np.zeros((3, 3), dtype='uint8')
kernel[:, 1] = 1
kernel[1, :] = 1
print(kernel)

img2 = cv2.dilate(img, kernel, iterations=1)

plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(img2)
plt.show()

# %% Some set operations
utk = cv2.imread(os.path.join(folder, 'utk.tif'), cv2.IMREAD_GRAYSCALE)
gt = cv2.imread(os.path.join(folder, 'gt.tif'), cv2.IMREAD_GRAYSCALE)

utkc = 255 - utk
utkORgt = cv2.bitwise_or(utk, gt)
utkANDgt = cv2.bitwise_and(utk, gt)
utkMinusgt = cv2.bitwise_and(utk, cv2.bitwise_not(gt))
# utkMinusgt = utk & ~gt

plt.subplot(231), plt.imshow(utk, cmap='gray')
plt.subplot(232), plt.imshow(gt, cmap='gray')
plt.subplot(233), plt.imshow(utkc, cmap='gray')
plt.subplot(234), plt.imshow(utkORgt, cmap='gray')
plt.subplot(235), plt.imshow(utkANDgt, cmap='gray')
plt.subplot(236), plt.imshow(utkMinusgt, cmap='gray')
plt.show()

# %% Dilation
j = cv2.imread(os.path.join(folder, 'j.png'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)
j_dilated = cv2.dilate(j, kernel, iterations=1)
plt.subplot(121), plt.imshow(j, cmap='gray')
plt.subplot(122), plt.imshow(j_dilated, cmap='gray')
plt.show()

# %% Erosion
j = cv2.imread(os.path.join(folder, 'j.png'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)
j_eroded = cv2.erode(j, kernel, iterations=1)
plt.subplot(131), plt.imshow(j, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(j_dilated, cmap='gray'), plt.title('Dilated')
plt.subplot(133), plt.imshow(j_eroded, cmap='gray'), plt.title('Eroded')
plt.show()

# %% Erosion2
wb = cv2.imread(os.path.join(folder, 'wirebond.tif'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)
wb_eroded = cv2.erode(wb, kernel, iterations=1)
plt.subplot(121), plt.imshow(wb, cmap='gray'), plt.title('Original')
plt.subplot(122), plt.imshow(wb_eroded, cmap='gray'), plt.title('Eroded')
plt.show()

# %% Opening
j_salt = cv2.imread(os.path.join(folder, 'j_salt.png'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)
j_cleaned = cv2.morphologyEx(j_salt, cv2.MORPH_OPEN, kernel)

plt.subplot(121), plt.imshow(j_salt, cmap='gray')
plt.subplot(122), plt.imshow(j_cleaned, cmap='gray')
plt.show()

# %% Closing
j_pepper = cv2.imread(os.path.join(folder, 'j_pepper.png'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)
j_cleaned = cv2.morphologyEx(j_pepper, cv2.MORPH_CLOSE, kernel)

plt.subplot(121), plt.imshow(j_pepper, cmap='gray')
plt.subplot(122), plt.imshow(j_cleaned, cmap='gray')
plt.show()

# %%===========================================================================
# MORPHOLOGY ALGORITHMS
# =============================================================================

# %% Noise filtering using opening and closing
noisy = cv2.imread(os.path.join(folder, 'noisy-fingerprint.tif'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)
noisy_o = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel)
noisy_oc = cv2.morphologyEx(noisy_o, cv2.MORPH_CLOSE, kernel)

plt.subplot(131), plt.imshow(noisy, cmap='gray')
plt.subplot(132), plt.imshow(noisy_o, cmap='gray')
plt.subplot(133), plt.imshow(noisy_oc, cmap='gray')
plt.show()

# %% Hit-or-Miss transform
input_image = np.array((
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 255, 255, 255, 0, 0, 0, 255],
    [0, 255, 255, 255, 0, 0, 0, 0],
    [0, 255, 255, 255, 0, 255, 0, 0],
    [0, 0, 255, 0, 0, 0, 0, 0],
    [0, 0, 255, 0, 0, 255, 255, 0],
    [0, 255, 0, 255, 0, 0, 255, 0],
    [0, 255, 255, 255, 0, 0, 0, 0]), dtype="uint8")

# kernel = np.array((
#        [0, 1, 0],
#        [1, -1, 1],
#        [0, 1, 0]), dtype="int")
k = np.array((
    [0, 1, 0],
    [-1, 1, 1],
    [-1, -1, 0]), dtype="int")

output_image = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, k)

plt.subplot(131), plt.imshow(input_image, cmap='gray'), plt.title('I')
plt.subplot(132), plt.imshow(k, cmap='gray'), plt.title('k')
plt.subplot(133), plt.imshow(output_image, cmap='gray'), plt.title('')
plt.show()

# %% BOUNDARY EXTRACTION
a = cv2.imread(os.path.join(folder, 'lincoln.tif'), cv2.IMREAD_GRAYSCALE)
b = np.ones((3, 3), np.uint8)
c = cv2.morphologyEx(a, cv2.MORPH_DILATE, b)
d = c & ~a

plt.subplot(131), plt.imshow(a, cmap='gray'), plt.title('A')
plt.subplot(132), plt.imshow(c, cmap='gray'), plt.title('$C = A \ominus B$')
plt.subplot(133), plt.imshow(d, cmap='gray'), plt.title('$D = A - (A \ominus B)$')
plt.show()

# %% Morphological Gradient - 'Border Detection'
j = cv2.imread(os.path.join(folder, 'j.png'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)
gradient = cv2.morphologyEx(j, cv2.MORPH_GRADIENT, kernel)

plt.subplot(121), plt.imshow(j, cmap='gray')
plt.subplot(122), plt.imshow(gradient, cmap='gray')
plt.show()

# %% HOLE FILLING
img = cv2.imread(os.path.join(folder, 'region-filling-reflections.tif'),
                 cv2.IMREAD_GRAYSCALE)
mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)

col = [0, 160, 180, 300]
row = [0, 250, 200, 240]
result = img.copy()
idx = 0
cv2.floodFill(result, mask, (row[idx], col[idx]), 255)
result_inv = cv2.bitwise_not(result)
saida = img | result_inv

plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('A')
plt.subplot(232), plt.imshow(~img, cmap='gray'), plt.title('$A^C$')
plt.subplot(233), plt.imshow(result, cmap='gray'), plt.title('$R = (X_{k-1} \oplus B ) \cap A^C$')
plt.subplot(234), plt.imshow(result_inv, cmap='gray'), plt.title('$R^C$')
plt.subplot(235), plt.imshow(saida, cmap='gray')
plt.show()

# %% Extraction of connected components
img = cv2.imread(os.path.join(folder, 'ten-objects.tif'), cv2.IMREAD_GRAYSCALE)
labels = []
_, img2 = cv2.connectedComponents(img)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.subplot(122), plt.imshow(img2, cmap='jet')
plt.show()

# %% Connected components
img = cv2.imread(os.path.join(folder, 'chickenfillet.tif'), cv2.IMREAD_GRAYSCALE)
_, img_bin = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY, cv2.CV_8U)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
img_erode = cv2.erode(img_bin, kernel, iterations=1)
n_cc, img_cc = cv2.connectedComponents(img_erode)
unique, count = np.unique(img_cc, return_counts=True)

plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(222), plt.imshow(img_bin, cmap='gray'), plt.title('Thresholded')
plt.subplot(223), plt.imshow(img_erode, cmap='gray'), plt.title('Eroded')
plt.subplot(224), plt.imshow(img_cc, cmap='jet'), plt.title('Connected components')
plt.show()

# %% Convex hull
img = cv2.imread(os.path.join(folder, 'j.png'), cv2.IMREAD_GRAYSCALE)
_, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

hull_list = []
drawn = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
for i in range(len(contours)):
    hull = cv2.convexHull(contours[i])
    hull_list.append(hull)

    cv2.drawContours(drawn, contours, i, 255)
    cv2.drawContours(drawn, hull_list, i, 255)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.subplot(122), plt.imshow(drawn, cmap='gray')
plt.show()

# %% Thinning
img = cv2.imread(os.path.join(folder, 'horse.png'), cv2.IMREAD_GRAYSCALE)
thinned_img = cv2.ximgproc.thinning(img, cv2.ximgproc.THINNING_GUOHALL)
cv2.imshow("Thinned Horse", thinned_img)
cv2.waitKey(0)


# %% Thickening


# %% Skeletons
def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy()  # don't clobber original
    skel = img.copy()

    skel[:, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:, :] = eroded[:, :]
        if cv2.countNonZero(img) == 0:
            break
    return skel


img = cv2.imread(os.path.join(folder, 'horse.png'), cv2.IMREAD_GRAYSCALE)
img2 = skeletonize(img)

plt.subplot(211), plt.imshow(img, cmap='gray')
plt.subplot(212), plt.imshow(img2, cmap='gray')
plt.show()

# %% Prunning


# %% Morphological Reconstruction
import skimage.morphology as skimorph

img = cv2.imread(os.path.join(folder, "balls.tif"), cv2.IMREAD_GRAYSCALE)

kernel = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
], dtype=img.dtype)

seeds = cv2.imread(os.path.join(folder, 'seeds.tif'), cv2.IMREAD_GRAYSCALE)
seeds = seeds > 200
#eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel, iterations=35)
reconstruct = skimorph.reconstruction(seeds, img)
plt.subplot("131"); plt.title("Original"); plt.imshow(img, 'gray')
plt.subplot("132"); plt.title("Seeds"); plt.imshow(seeds, 'gray')
plt.subplot("133"); plt.title("Reconstructed"); plt.imshow(reconstruct, 'gray')
plt.show()

# %% Opening by Reconstruction
mask = cv2.imread(os.path.join(folder, "text.tif"), cv2.IMREAD_GRAYSCALE)
marker = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1,15)))
seeds = np.copy(marker)

last_image = 0

while True:
    seeds = cv2.dilate(seeds, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    seeds = cv2.bitwise_and(seeds, mask)
    if (seeds == last_image).all():
        break
    last_image = seeds

plt.subplot("131"); plt.title("Mask"); plt.imshow(mask, 'gray')
plt.subplot("132"); plt.title("Marker"); plt.imshow(marker, 'gray')
plt.subplot("133"); plt.title("seeds"); plt.imshow(seeds, 'gray')
plt.show()

# %%===========================================================================
# GRAY-SCALE MORPHOLOGY
# =============================================================================

# %% GRAYSCALE DILATION
img = cv2.imread(os.path.join(folder, "dowels.tif"), cv2.IMREAD_GRAYSCALE)

kernel = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
], dtype=img.dtype)

dilated = cv2.dilate(img, kernel, iterations=4)

plt.subplot("121"); plt.title("Original"); plt.imshow(img, 'gray')
plt.subplot("122"); plt.title("Dilated"); plt.imshow(dilated, 'gray')
plt.show()

# %% GRAYSCALE EROSION
img = cv2.imread(os.path.join(folder, "dowels.tif"), cv2.IMREAD_GRAYSCALE)

kernel = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
], dtype=img.dtype)

eroded = cv2.erode(img, kernel, iterations=4)

plt.subplot("121"); plt.title("Original"); plt.imshow(img, 'gray')
plt.subplot("122"); plt.title("Eroded"); plt.imshow(eroded, 'gray')
plt.show()

# %% GRAYSCALE OPENING
img = cv2.imread(os.path.join(folder, "noised_ball.png"), cv2.IMREAD_GRAYSCALE)

kernel = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
], dtype=img.dtype)

opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=4)

plt.subplot("121"); plt.title("Original"); plt.imshow(img, 'gray')
plt.subplot("122"); plt.title("Opened"); plt.imshow(opened, 'gray')
plt.show()

# %% GRAYSCALE CLOSING
img = cv2.imread(os.path.join(folder, "sliced_ball.png"), cv2.IMREAD_GRAYSCALE)

kernel = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
], dtype=img.dtype)

closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=4)

plt.subplot("121"); plt.title("Original"); plt.imshow(img, 'gray')
plt.subplot("122"); plt.title("Closed"); plt.imshow(closed, 'gray')
plt.show()

# %% GRAYSCALE TOP-HAT
img = cv2.imread(os.path.join(folder, "balls.png"), cv2.IMREAD_GRAYSCALE)

kernel = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
], dtype=img.dtype)

top = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel, iterations=8)

plt.subplot("121"); plt.title("Original"); plt.imshow(img, 'gray')
plt.subplot("122"); plt.title("Top Hat"); plt.imshow(top, 'gray')
plt.show()

# %% GRAYSCALE BOTTOM-HAT
img = 255 - cv2.imread(os.path.join(folder, "balls.png"), cv2.IMREAD_GRAYSCALE)

kernel = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
], dtype=img.dtype)

black = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel, iterations=8)

plt.subplot("121"); plt.title("Original"); plt.imshow(img, 'gray')
plt.subplot("122"); plt.title("Black Hat"); plt.imshow(black, 'gray')
plt.show()

# %% GRANULOMETRY

# from https://www.scipy-lectures.org/advanced/image_processing/auto_examples/plot_granulo.html

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def disk_structure(n):
    struct = np.zeros((2 * n + 1, 2 * n + 1))
    x, y = np.indices((2 * n + 1, 2 * n + 1))
    mask = (x - n)**2 + (y - n)**2 <= n**2
    struct[mask] = 1
    return struct.astype(np.bool)


def granulometry(data, sizes=None):
    s = max(data.shape)
    if sizes is None:
        sizes = range(1, s/2, 2)
    granulo = [ndimage.binary_opening(data, \
            structure=disk_structure(n)).sum() for n in sizes]
    return granulo


np.random.seed(1)
n = 10
l = 256
im = np.zeros((l, l))
points = l*np.random.random((2, n**2))
im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
im = ndimage.gaussian_filter(im, sigma=l/(4.*n))

mask = im > im.mean()

granulo = granulometry(mask, sizes=np.arange(2, 19, 4))

plt.figure(figsize=(6, 2.2))

plt.subplot(121)
plt.imshow(mask, cmap=plt.cm.gray)
opened = ndimage.binary_opening(mask, structure=disk_structure(10))
opened_more = ndimage.binary_opening(mask, structure=disk_structure(14))
plt.contour(opened, [0.5], colors='b', linewidths=2)
plt.contour(opened_more, [0.5], colors='r', linewidths=2)
plt.axis('off')
plt.subplot(122)
plt.plot(np.arange(2, 19, 4), granulo, 'ok', ms=8)


plt.subplots_adjust(wspace=0.02, hspace=0.15, top=0.95, bottom=0.15, left=0, right=0.95)
plt.show()

# %% Texture Segmentation

# from http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_gabor.html

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel


def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i


# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)


shrink = (slice(0, None, 3), slice(0, None, 3))
brick = img_as_float(data.load('brick.png'))[shrink]
grass = img_as_float(data.load('grass.png'))[shrink]
wall = img_as_float(data.load('rough-wall.png'))[shrink]
image_names = ('brick', 'grass', 'wall')
images = (brick, grass, wall)

# prepare reference features
ref_feats = np.zeros((3, len(kernels), 2), dtype=np.double)
ref_feats[0, :, :] = compute_feats(brick, kernels)
ref_feats[1, :, :] = compute_feats(grass, kernels)
ref_feats[2, :, :] = compute_feats(wall, kernels)

print('Rotated images matched against references using Gabor filter banks:')

print('original: brick, rotated: 30deg, match result: ', end='')
feats = compute_feats(ndi.rotate(brick, angle=190, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: brick, rotated: 70deg, match result: ', end='')
feats = compute_feats(ndi.rotate(brick, angle=70, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: grass, rotated: 145deg, match result: ', end='')
feats = compute_feats(ndi.rotate(grass, angle=145, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])


def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

# Plot a selection of the filter bank kernels and their responses.
results = []
kernel_params = []
for theta in (0, 1):
    theta = theta / 4. * np.pi
    for frequency in (0.1, 0.4):
        kernel = gabor_kernel(frequency, theta=theta)
        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
        kernel_params.append(params)
        # Save kernel and the power image for each image
        results.append((kernel, [power(img, kernel) for img in images]))

fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(5, 6))
plt.gray()

fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

axes[0][0].axis('off')

# Plot original images
for label, img, ax in zip(image_names, images, axes[0][1:]):
    ax.imshow(img)
    ax.set_title(label, fontsize=9)
    ax.axis('off')

for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
    # Plot Gabor kernel
    ax = ax_row[0]
    ax.imshow(np.real(kernel), interpolation='nearest')
    ax.set_ylabel(label, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot Gabor responses with the contrast normalized for each filter
    vmin = np.min(powers)
    vmax = np.max(powers)
    for patch, ax in zip(powers, ax_row[1:]):
        ax.imshow(patch, vmin=vmin, vmax=vmax)
        ax.axis('off')

plt.show()
