# %%===========================================================================
# IMPORT MODULES
# 
# =============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_histogram
path = "../img/"

# %%===========================================================================
# CREATING IMAGES
#
# =============================================================================

#%% Creating an image filled with zeros
img = np.zeros((5,5), dtype=np.int16)
plt.imshow(img, cmap='gray')
print(img)

#%% Creating an image filled with ones
img = np.ones((5, 5))
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
print(img)

#%% Creating an image filled with a scalar
img = 127*np.ones((50, 50))
plt.imshow(img, cmap='gray')
plt.imshow(img, cmap='gray', vmin=0, vmax=255)

#%% Initializing a grayscale image with random values, uniformly distributed
img = np.ones((250, 250), dtype=np.uint8)
cv2.randu(img, 0, 255)
plt.subplot("221"); plt.title("Original"); plt.imshow(img, cmap='gray')
plt.subplot("222"); plt.title("Histogram"); plt.hist(img.ravel(), 256, [0, 256])
plt.show()

#%% Initializing a color image with random values, uniformly distributed
img = np.ones((250, 250, 3), dtype=np.uint8)
bgr = cv2.split(img)
cv2.randu(bgr[0], 0, 255)
cv2.randu(bgr[1], 0, 255)
cv2.randu(bgr[2], 0, 255)
img = cv2.merge(bgr)
cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(16,4))
plt.subplot("121"); plt.title("Original"); plt.imshow(img)
plt.subplot("122"); plt.title("Histogram"); plot_histogram(img)
plt.show()

#%% Initializing a grayscale image with random values, normally distributed
img = np.ones((250, 250), dtype=np.uint8)
cv2.randn(img, 127, 40)

plt.figure(figsize=(16,4))
plt.subplot("221"); plt.title("Original"); plt.imshow(img, cmap='gray')
plt.subplot("222"); plt.title("Histogram"); plt.hist(img.ravel(), 256, [0, 256])
plt.show()

#%% Initializing a color image with random values, normally distributed
img = np.zeros((50, 50, 3), dtype=np.uint8)
cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)
while 0xFF & cv2.waitKey(1) != ord('q'):
    bgr = cv2.split(img)
    cv2.randn(bgr[0], 127, 40)
    cv2.randn(bgr[1], 127, 40)
    cv2.randn(bgr[2], 127, 40)
    img = cv2.merge(bgr)
    cv2.imshow("img", img)
cv2.destroyAllWindows()

#%%
plt.figure(figsize=(16,4))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot("121"); plt.title("Original"); plt.imshow(img)
plt.subplot("122"); plt.title("Histogram"); plot_histogram(img)
plt.show()


#%% Scale image
img = np.ones((3, 3), dtype=np.float32)

cv2.randn(img, 0, 1)
print("Normally distributed random values = \n", img, "\n\n")

img2 = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX)
print("Normalized = \n", img2, "\n\n")

img2 = np.asarray(img2, dtype=np.uint8)
print("Converted to uint8 = \n", img2, "\n\n")

img = (255. * cv2.normalize(img, None, 1.0, 0, cv2.NORM_MINMAX)).astype(np.uint8)
print(img, "\n\n")

print("img == img2?")
print((img == img2).all())

# Normally distributed random values = 
#  [[-0.10685379 -0.3343182   0.6172743 ]
#  [-1.0519822   2.1705427   0.2486184 ]
#  [-0.12854138 -1.2465241  -0.5537428 ]] 


# Normalized = 
#  [[ 8.5048355e+01  6.8073730e+01  1.3908670e+02]
#  [ 1.4517767e+01  2.5499998e+02  1.1157561e+02]
#  [ 8.3429909e+01 -2.9158273e-06  5.1699078e+01]] 


# Converted to uint8 = 
#  [[ 85  68 139]
#  [ 14 254 111]
#  [ 83   0  51]] 


# [[ 85  68 139]
#  [ 14 255 111]
#  [ 83   0  51]] 


# img == img2?
# False