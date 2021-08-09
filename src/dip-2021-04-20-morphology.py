# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 08:46:57 2018

@author: Vieira
"""

# %% Import modules
from utils import *

PATH_TO_IMAGES = '../img'


# %% Example
img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'text.tif'), cv2.IMREAD_GRAYSCALE)
kernel = np.zeros((3, 3), dtype='uint8')
kernel[:, 1] = 1
kernel[1, :] = 1
print(kernel)

img2 = cv2.dilate(img, kernel, iterations=5)

cv2.namedWindow("img", cv2.WINDOW_NORMAL  | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img2", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

while 0xFF & cv2.waitKey(1) != ord('q'):
    
    cv2.imshow("img", img)
    cv2.imshow("img2", img2)

cv2.destroyAllWindows()

# plt.figure(figsize=(64,32))
# plt.subplot(121), plt.imshow(img , cmap='gray')
# plt.subplot(122), plt.imshow(img2, cmap='gray')
# plt.show()

# %% Some set operations
utk = cv2.imread(os.path.join(PATH_TO_IMAGES, 'utk.tif'), cv2.IMREAD_GRAYSCALE)
gt = cv2.imread( os.path.join(PATH_TO_IMAGES, 'gt.tif'), cv2.IMREAD_GRAYSCALE)

utkc = 255 - utk
utkORgt = cv2.bitwise_or(utk, gt)
utkANDgt = cv2.bitwise_and(utk, gt)
utkMinusgt = cv2.bitwise_and(utk, cv2.bitwise_not(gt))

plt.figure(figsize=(32,24))
plt.subplot(231), plt.imshow(utk, cmap='gray'), plt.axis('off')
plt.subplot(232), plt.imshow(gt, cmap='gray'), plt.axis('off')
plt.subplot(233), plt.imshow(utkc, cmap='gray'), plt.axis('off')
plt.subplot(234), plt.imshow(utkORgt, cmap='gray'), plt.axis('off')
plt.subplot(235), plt.imshow(utkANDgt, cmap='gray'), plt.axis('off')
plt.subplot(236), plt.imshow(utkMinusgt, cmap='gray'), plt.axis('off')
plt.show()

# %% Dilation
j = cv2.imread(os.path.join(PATH_TO_IMAGES, 'j.png'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)
j_dilated = cv2.dilate(j, kernel, iterations=1)

plt.figure(figsize=(32,16))
plt.subplot(121), plt.imshow(j, cmap='gray')
plt.subplot(122), plt.imshow(j_dilated, cmap='gray')
plt.show()

# %% Erosion
j = cv2.imread(os.path.join(PATH_TO_IMAGES, 'j.png'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)
j_eroded = cv2.erode(j, kernel, iterations=1)
plt.figure(figsize=(32,16))
plt.subplot(131), plt.imshow(j, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(j_dilated, cmap='gray'), plt.title('Dilated')
plt.subplot(133), plt.imshow(j_eroded, cmap='gray'), plt.title('Eroded')
plt.show()

# %% Erosion2
wb = cv2.imread(os.path.join(PATH_TO_IMAGES, 'wirebond.tif'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)
wb_eroded = cv2.erode(wb, kernel, iterations=1)

plt.figure(figsize=(32,16))
plt.subplot(121), plt.imshow(wb, cmap='gray'), plt.title('Original')
plt.subplot(122), plt.imshow(wb_eroded, cmap='gray'), plt.title('Eroded')
plt.show()

# %% Opening
j_salt = cv2.imread(os.path.join(PATH_TO_IMAGES, 'j_salt.png'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)
j_cleaned = cv2.morphologyEx(j_salt, cv2.MORPH_OPEN, kernel)

plt.figure(figsize=(32,16))
plt.subplot(121), plt.imshow(j_salt, cmap='gray')
plt.subplot(122), plt.imshow(j_cleaned, cmap='gray')
plt.show()

# %% Closing
j_pepper = cv2.imread(os.path.join(PATH_TO_IMAGES, 'j_pepper.png'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)
j_cleaned = cv2.morphologyEx(j_pepper, cv2.MORPH_CLOSE, kernel)
plt.figure(figsize=(32,16))
plt.subplot(121), plt.imshow(j_pepper, cmap='gray')
plt.subplot(122), plt.imshow(j_cleaned, cmap='gray')
plt.show()

# %%===========================================================================
# MORPHOLOGY ALGORITHMS
# =============================================================================

# %% Noise filtering using opening and closing
noisy = cv2.imread(os.path.join(PATH_TO_IMAGES, 'noisy-fingerprint.tif'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)
noisy_o = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel)
noisy_oc = cv2.morphologyEx(noisy_o, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(32,16))
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
plt.figure(figsize=(32,16))
plt.subplot(131), plt.imshow(input_image, cmap='gray'), plt.title('I')
plt.subplot(132), plt.imshow(k, cmap='gray'), plt.title('k')
plt.subplot(133), plt.imshow(output_image, cmap='gray'), plt.title('')
plt.show()

# %% BOUNDARY EXTRACTION
a = cv2.imread(os.path.join(PATH_TO_IMAGES, 'lincoln.tif'), cv2.IMREAD_GRAYSCALE)
b = np.ones((3, 3), np.uint8)
c = cv2.morphologyEx(a, cv2.MORPH_DILATE, b)
d = c & ~a

plt.figure(figsize=(32,16))
plt.subplot(131), plt.imshow(a, cmap='gray'), plt.title('A')
plt.subplot(132), plt.imshow(c, cmap='gray'), plt.title('$C = A \ominus B$')
plt.subplot(133), plt.imshow(d, cmap='gray'), plt.title('$D = A - (A \ominus B)$')
plt.show()

# %% Morphological Gradient - 'Border Detection'
j = cv2.imread(os.path.join(PATH_TO_IMAGES, 'j.png'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)
gradient = cv2.morphologyEx(j, cv2.MORPH_GRADIENT, kernel)

plt.figure(figsize=(32,16))
plt.subplot(121), plt.imshow(j, cmap='gray')
plt.subplot(122), plt.imshow(gradient, cmap='gray')
plt.show()

# %% HOLE FILLING
img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'region-filling-reflections.tif'),
                 cv2.IMREAD_GRAYSCALE)
mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)

col = [0, 160, 180, 300]
row = [0, 250, 200, 240]
result = img.copy()
idx = 0
cv2.floodFill(result, mask, (row[idx], col[idx]), 255)
result_inv = cv2.bitwise_not(result)
saida = img | result_inv

plt.figure(figsize=(32,16))
plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('A')
plt.subplot(232), plt.imshow(~img, cmap='gray'), plt.title('$A^C$')
plt.subplot(233), plt.imshow(result, cmap='gray'), plt.title('$R = (X_{k-1} \oplus B ) \cap A^C$')
plt.subplot(234), plt.imshow(result_inv, cmap='gray'), plt.title('$R^C$')
plt.subplot(235), plt.imshow(saida, cmap='gray')
plt.show()

# %% Extraction of connected components
img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'ten-objects.tif'), cv2.IMREAD_GRAYSCALE)
labels = []
_, img2 = cv2.connectedComponents(img)

plt.figure(figsize=(32,16))
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.subplot(122), plt.imshow(img2, cmap='gray')
plt.show()

# %% Connected components
img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'chickenfillet.tif'), cv2.IMREAD_GRAYSCALE)
_, img_bin = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY, cv2.CV_8U)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
img_erode = cv2.erode(img_bin, kernel, iterations=1)
n_cc, img_cc = cv2.connectedComponents(img_erode)
unique, count = np.unique(img_cc, return_counts=True)

plt.figure(figsize=(32,16))
plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(222), plt.imshow(img_bin, cmap='gray'), plt.title('Thresholded')
plt.subplot(223), plt.imshow(img_erode, cmap='gray'), plt.title('Eroded')
plt.subplot(224), plt.imshow(img_cc, cmap='jet'), plt.title('Connected components')
plt.show()

# %% Convex hull
img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'j.png'), cv2.IMREAD_GRAYSCALE)
# img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'horse.png'), cv2.IMREAD_GRAYSCALE)
# img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'balls.tif'), cv2.IMREAD_GRAYSCALE)
# img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'elementsbw.png'), cv2.IMREAD_GRAYSCALE)
# img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'ten-objects.tif'), cv2.IMREAD_GRAYSCALE)
# img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'lincoln.tif'), cv2.IMREAD_GRAYSCALE)

contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

hull_list = []

draw_convhull = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
draw_contours = draw_convhull.copy()

for contour in contours:
    # contour = contours[i]
    hull = cv2.convexHull(contour)
    hull_list.append(hull)
    cv2.drawContours(draw_contours, [contour]  , 0, 255)
    cv2.drawContours(draw_convhull, [hull]     , 0, 255)

plt.figure(figsize=(32,16))
plt.subplot(131), plt.imshow(img          , cmap='gray')
plt.subplot(132), plt.imshow(draw_contours, cmap='gray')
plt.subplot(133), plt.imshow(draw_convhull, cmap='gray')
plt.show()

# %% Thinning
# img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'horse.png'), cv2.IMREAD_GRAYSCALE)
# img = 255 - cv2.imread(os.path.join(PATH_TO_IMAGES, 'horse.png'), cv2.IMREAD_GRAYSCALE)
# img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'horse2.png'), cv2.IMREAD_GRAYSCALE)
# img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'elementsbw.png'), cv2.IMREAD_GRAYSCALE)
# img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'j.png'), cv2.IMREAD_GRAYSCALE)
# img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'balls.tif'), cv2.IMREAD_GRAYSCALE)
img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'ten-objects.tif'), cv2.IMREAD_GRAYSCALE)
# img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'lincoln.tif'), cv2.IMREAD_GRAYSCALE)

thinned_img = cv2.ximgproc.thinning(img, cv2.ximgproc.THINNING_GUOHALL)

cv2.namedWindow("img" , cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img2", cv2.WINDOW_KEEPRATIO)
cv2.imshow("img" , img)
cv2.imshow("img2", thinned_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


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


img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'horse2.png'), cv2.IMREAD_GRAYSCALE)
img2 = skeletonize(img)

cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img2", cv2.WINDOW_KEEPRATIO)
cv2.imshow("img" , img)
cv2.imshow("img2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% Prunning


# %% Morphological Reconstruction
import skimage.morphology as skimorph

img = cv2.imread(os.path.join(PATH_TO_IMAGES, "balls.tif"), cv2.IMREAD_GRAYSCALE)

kernel = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
], dtype=img.dtype)

seeds = cv2.imread(os.path.join(PATH_TO_IMAGES, 'seeds.tif'), cv2.IMREAD_GRAYSCALE)
seeds = seeds > 200
#eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel, iterations=35)
reconstruct = skimorph.reconstruction(seeds, img)

plt.figure(figsize=(36,12))
plt.subplot("131"); plt.title("Original"); plt.imshow(img, 'gray')
plt.subplot("132"); plt.title("Seeds"); plt.imshow(seeds, 'gray')
plt.subplot("133"); plt.title("Reconstructed"); plt.imshow(reconstruct, 'gray')
plt.show()

# %% Opening by Reconstruction
mask = cv2.imread(os.path.join(PATH_TO_IMAGES, "text.tif"), cv2.IMREAD_GRAYSCALE)
marker = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1,15)))
seeds = np.copy(marker)

last_image = 0

while True:
    seeds = cv2.dilate(seeds, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    seeds = cv2.bitwise_and(seeds, mask)
    if (seeds == last_image).all():
        break
    last_image = seeds

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 64}
plt.rc('font', **font)

plt.figure(figsize=(69,23))
plt.subplot("131"); plt.title("Mask"); plt.imshow(mask, 'gray')
plt.subplot("132"); plt.title("Marker"); plt.imshow(marker, 'gray')
plt.subplot("133"); plt.title("seeds"); plt.imshow(seeds, 'gray')
plt.show()

# %%===========================================================================
# GRAY-SCALE MORPHOLOGY
# =============================================================================

# %% GRAYSCALE DILATION
img = cv2.imread(os.path.join(PATH_TO_IMAGES, "dowels.tif"), cv2.IMREAD_GRAYSCALE)

kernel = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
], dtype=img.dtype)

dilated = cv2.dilate(img, kernel, iterations=4)

plt.figure(figsize=(69,23))
plt.subplot("121"); plt.title("Original"); plt.imshow(img, 'gray')
plt.subplot("122"); plt.title("Dilated"); plt.imshow(dilated, 'gray')
plt.show()

# %% GRAYSCALE EROSION
img = cv2.imread(os.path.join(PATH_TO_IMAGES, "dowels.tif"), cv2.IMREAD_GRAYSCALE)

kernel = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
], dtype=img.dtype)

eroded = cv2.erode(img, kernel, iterations=4)

plt.figure(figsize=(69,23))
plt.subplot("121"); plt.title("Original"); plt.imshow(img, 'gray')
plt.subplot("122"); plt.title("Eroded"); plt.imshow(eroded, 'gray')
plt.show()

# %% GRAYSCALE OPENING
img = cv2.imread(os.path.join(PATH_TO_IMAGES, "dowels.tif"), cv2.IMREAD_GRAYSCALE)

kernel = np.ones((5,5), dtype=img.dtype)

opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=4)

plt.figure(figsize=(69,23))
plt.subplot("121"); plt.title("Original"); plt.imshow(img, 'gray')
plt.subplot("122"); plt.title("Opened"); plt.imshow(opened, 'gray')
plt.show()

# %% GRAYSCALE CLOSING
img = cv2.imread(os.path.join(PATH_TO_IMAGES, "dowels.tif"), cv2.IMREAD_GRAYSCALE)

kernel = np.ones((5,5), dtype=img.dtype)

closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=4)

plt.figure(figsize=(69,23))
plt.subplot("121"); plt.title("Original"); plt.imshow(img, 'gray')
plt.subplot("122"); plt.title("Closed"); plt.imshow(closed, 'gray')
plt.show()

# %% GRAYSCALE TOP-HAT
img = cv2.imread(os.path.join(PATH_TO_IMAGES, "dowels.tif"), cv2.IMREAD_GRAYSCALE)

kernel = np.ones((5,5), dtype=img.dtype)

top = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel, iterations=8)

plt.figure(figsize=(69,23))
plt.subplot("121"); plt.title("Original"); plt.imshow(img, 'gray')
plt.subplot("122"); plt.title("Top Hat"); plt.imshow(top, 'gray')
plt.show()

# %% GRAYSCALE BOTTOM-HAT
img = cv2.imread(os.path.join(PATH_TO_IMAGES, "dowels.tif"), cv2.IMREAD_GRAYSCALE)

kernel = np.ones((5,5), dtype=img.dtype)

black = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel, iterations=8)

plt.figure(figsize=(69,23))
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

plt.figure(figsize=(69,23))

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


