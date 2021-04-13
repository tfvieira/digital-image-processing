#%% Import modules
from dip import *

#%% Example
img = cv2.imread(os.path.join(folder, 'text.tif'), cv2.IMREAD_GRAYSCALE)
kernel = np.zeros((3,3), dtype = 'uint8')
kernel[:, 1] = 1
kernel[1,:] = 1
print(kernel)

img2 = cv2.dilate(img, kernel, iterations = 1)

plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(img2)
plt.show()

#%% Some set operations
utk = cv2.imread(os.path.join(folder, 'utk.tif'), cv2.IMREAD_GRAYSCALE)
gt = cv2.imread(os.path.join(folder, 'gt.tif'), cv2.IMREAD_GRAYSCALE)

utkc = 255 - utk
utkORgt = cv2.bitwise_or(utk, gt)
utkANDgt = cv2.bitwise_and(utk, gt)
utkMinusgt = cv2.bitwise_and(utk, cv2.bitwise_not(gt))
#utkMinusgt = utk & ~gt

plt.subplot(231), plt.imshow(utk, cmap='gray')
plt.subplot(232), plt.imshow(gt, cmap='gray')
plt.subplot(233), plt.imshow(utkc, cmap='gray')
plt.subplot(234), plt.imshow(utkORgt, cmap='gray')
plt.subplot(235), plt.imshow(utkANDgt, cmap='gray')
plt.subplot(236), plt.imshow(utkMinusgt, cmap='gray')
plt.show()

#%% Dilation
j = cv2.imread(os.path.join(folder, 'j.png'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3,3), np.uint8)
j_dilated = cv2.dilate(j, kernel, iterations=1)
plt.subplot(121), plt.imshow(j, cmap='gray')
plt.subplot(122), plt.imshow(j_dilated, cmap='gray')
plt.show()

#%% Erosion
j = cv2.imread(os.path.join(folder, 'j.png'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3,3), np.uint8)
j_eroded = cv2.erode(j, kernel, iterations=1)
plt.subplot(131), plt.imshow(j, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(j_dilated, cmap='gray'), plt.title('Dilated')
plt.subplot(133), plt.imshow(j_eroded, cmap='gray'), plt.title('Eroded')
plt.show()

#%% Erosion2
wb = cv2.imread(os.path.join(folder, 'wirebond.tif'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5,5), np.uint8)
wb_eroded = cv2.erode(wb, kernel, iterations=1)
plt.subplot(121), plt.imshow(wb, cmap='gray'), plt.title('Original')
plt.subplot(122), plt.imshow(wb_eroded, cmap='gray'), plt.title('Eroded')
plt.show()


#%% Opening
j_salt = cv2.imread(os.path.join(folder, 'j_salt.png'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)
j_cleaned = cv2.morphologyEx(j_salt, cv2.MORPH_OPEN, kernel)

plt.subplot(121), plt.imshow(j_salt, cmap='gray')
plt.subplot(122), plt.imshow(j_cleaned, cmap='gray')
plt.show()

#%% Closing
j_pepper = cv2.imread(os.path.join(folder, 'j_pepper.png'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)
j_cleaned = cv2.morphologyEx(j_pepper, cv2.MORPH_CLOSE, kernel)

plt.subplot(121), plt.imshow(j_pepper, cmap='gray')
plt.subplot(122), plt.imshow(j_cleaned, cmap='gray')
plt.show()

# %%===========================================================================
# MORPHOLOGY ALGORITHMS
# =============================================================================

#%% Noise filtering using opening and closing
noisy = cv2.imread(os.path.join(folder, 'noisy-fingerprint.tif'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)
noisy_o =  cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel)
noisy_oc = cv2.morphologyEx(noisy_o, cv2.MORPH_CLOSE, kernel)

plt.subplot(131), plt.imshow(noisy, cmap='gray')
plt.subplot(132), plt.imshow(noisy_o, cmap='gray')
plt.subplot(133), plt.imshow(noisy_oc, cmap='gray')
plt.show()

#%% Hit-or-Miss transform
input_image = np.array((
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 255, 255, 255, 0, 0, 0, 255],
    [0, 255, 255, 255, 0, 0, 0, 0],
    [0, 255, 255, 255, 0, 255, 0, 0],
    [0, 0, 255, 0, 0, 0, 0, 0],
    [0, 0, 255, 0, 0, 255, 255, 0],
    [0,255, 0, 255, 0, 0, 255, 0],
    [0, 255, 255, 255, 0, 0, 0, 0]), dtype="uint8")

#kernel = np.array((
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

#%% BOUNDARY EXTRACTION
a = cv2.imread(os.path.join(folder, 'lincoln.tif'), cv2.IMREAD_GRAYSCALE)
b = np.ones((3,3), np.uint8)
c = cv2.morphologyEx(a, cv2.MORPH_DILATE, b)
d = c & ~a

plt.subplot(221), plt.imshow(a, cmap='gray'), plt.title('A')
plt.subplot(222), plt.imshow(b, cmap='gray'), plt.title('B')
plt.subplot(223), plt.imshow(c, cmap='gray'), plt.title('C')
plt.subplot(224), plt.imshow(d, cmap='gray'), plt.title('D')
plt.show()

#%% Morphological Gradient - 'Border Detection'
j = cv2.imread(os.path.join(folder, 'j.png'), cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3,3), np.uint8)
gradient = cv2.morphologyEx(j, cv2.MORPH_GRADIENT, kernel)

plt.subplot(121), plt.imshow(j, cmap='gray')
plt.subplot(122), plt.imshow(gradient, cmap='gray')
plt.show()

#%% HOLE FILLING
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

#%% Extraction of connected components
img = cv2.imread(os.path.join(folder, 'ten-objects.tif'), cv2.IMREAD_GRAYSCALE)
labels = []
_, labels = cv2.connectedComponents(img)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.subplot(122), plt.imshow(img2, cmap='jet')
plt.show()

#%% Connected components
img = cv2.imread(os.path.join(folder, 'chickenfillet.tif'), cv2.IMREAD_GRAYSCALE)
_, img_bin = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY, cv2.CV_8U)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4,4))
img_erode = cv2.erode(img_bin, kernel, iterations=1)
n_cc, img_cc = cv2.connectedComponents(img_erode)
unique, count = np.unique(img_cc, return_counts=True)

plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(222), plt.imshow(img_bin, cmap='gray'), plt.title('Thresholded')
plt.subplot(223), plt.imshow(img_erode, cmap='gray'), plt.title('Eroded')
plt.subplot(224), plt.imshow(img_cc, cmap='jet'), plt.title('Connected components')
plt.show()

#%% Convex hull
img = cv2.imread(os.path.join(folder, 'j.png'), cv2.IMREAD_GRAYSCALE)
_, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

hull_list = []
drawn = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)
for i in range(len(contours)):
    hull = cv2.convexHull(contours[i])
    hull_list.append(hull)

    cv2.drawContours(drawn, contours, i, 255)
    cv2.drawContours(drawn, hull_list, i, 255)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.subplot(122), plt.imshow(drawn, cmap='gray')
plt.show()

#%% Thinning


#%% Thickening


#%% Skeletons
def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break
    return skel

img = cv2.imread(os.path.join(folder, 'horse.png'), cv2.IMREAD_GRAYSCALE)
img2 = skeletonize(img)

plt.subplot(211), plt.imshow(img, cmap='gray')
plt.subplot(212), plt.imshow(img2, cmap='gray')
plt.show()

#%% Prunning


#%% Morphological Reconstruction


#%% Opening by Reconstruction























# %%===========================================================================
# GRAY-SCALE MORPHOLOGY
# =============================================================================

#%% GRAYSCALE DILATION

#%% GRAYSCALE EROSION

#%% GRAYSCALE OPENING

#%% GRAYSCALE CLOSING

#%% GRAYSCALE TOP-HAT

#%% GRAYSCALE BOTTOM-HAT

#%% GRANULOMETRY

#%% Texture Segmentation