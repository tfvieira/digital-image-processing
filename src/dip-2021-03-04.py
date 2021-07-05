# %%===========================================================================
# IMPORT MODULES
# 
# =============================================================================
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
path = "../img/"

# %%===========================================================================
# TRANSLATIONS
#==============================================================================
img = cv2.imread(os.path.join(path, 'lena.png'))
cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%store height and width of the image
img = cv2.imread(os.path.join(path, 'lena.png'))
height, width, = img.shape[:2]
quarter_height, quarter_width = height/4, width/4
#%
# T = | 1 0 Tx |
#     | 0 1 Ty |
# T is our transformation matrix
T = np.float32([[1, 0, quarter_width],
                [0, 1, quarter_height]])
#% We use warpaffine to transform the image using matrix 
img_translation = cv2.warpAffine(img, T, (width,height))
plt.figure(figsize=(16,4))
plt.subplot(121); plt.title("Original"); plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.subplot(122); plt.title("Translated"); plt.imshow(cv2.cvtColor(img_translation,cv2.COLOR_BGR2RGB))

#%%============================================================================
# ROTATIONS
#==============================================================================

### Affine rotation
### cv2.warpAffine(image, Translation matrix, image_shape)
# M = | cos theta   -sin theta |
#     | sin theta    cos theta |
# Where theta is the angle of rotation
#cv2.getRotationMatrix2D(rotation_center_x, rotation_center_y, angle_of_rotation, scale)
image = cv2.imread(os.path.join(path, 'lena.png'))
height, width = image.shape[:2]
#% divided by 2 to rototate the image around its centre
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 45, .5)
img_rotated = cv2.warpAffine(image, rotation_matrix,(width, height))
plt.figure(figsize=(16,4))
plt.subplot(121); plt.title("Original"); plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.subplot(122); plt.title("Rotation"); plt.imshow(cv2.cvtColor(img_rotated,cv2.COLOR_BGR2RGB))

#%%============================================================================
# IMAGE PYRAMIDS
#==============================================================================

### Re-Sizing, scaling and interpolation
### cv2.INTER_AREA - good for shrinking or down sampling
### cv2.INTER_NEAREST - fastest
### cv2.INTER_LINEAR - Good for zooming or up scaling(default)
### cv2.INTER_CUBIC - Better
### cv2.INTER_LANCZOS4 - Best

### Good comparison of interpolation methods http://tanbakuchi.com/posts/comparison-of-openv-interpolation-algorithms/
### cv2.resize(image, dsize(output image size), x scale, y scale, interpolation)
#########################################################################################################
image = cv2.imread(os.path.join(path, 'lena.png'))
# Let's make our image 3/4 of it's original size
image_scaled = cv2.resize(image, None, fx = 0.75, fy = 0.75)  ## fx and fy are the factors
cv2.imshow('Scaling - Linear interpolation', image_scaled)
cv2.waitKey()
## make the image double it's size
img_scaled = cv2.resize(image, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('Scaling - Cubic interpolation', img_scaled)
cv2.waitKey()
## skew the re-sizing by setting exact dimensions
img_scaled = cv2.resize(image, (900, 400), interpolation = cv2.INTER_AREA)
cv2.imshow('Scaling - Skewed Size', img_scaled)
cv2.waitKey()
cv2.destroyAllWindows()

#%%============================================================================
# BITWISE OPERATIONS
# =============================================================================
image = cv2.imread(os.path.join(path, 'lena.png'))

#making a square
square = np.zeros((300,300), np.uint8)
cv2.rectangle(square, (50,50), (250,250), 255, -2)
cv2.imshow('Square', square)

#making ellipse
ellipse = np.zeros((300,300), np.uint8)
cv2.ellipse(ellipse, (150,150), (150,150), 30, 0, 180, 255, -1)
cv2.imshow('Ellipse', ellipse)

#show only where they intersectd 
And = cv2.bitwise_and(square, ellipse)
cv2.imshow('And',And)

bitwiseOr = cv2.bitwise_or(square, ellipse)
cv2.imshow('Or',bitwiseOr)

bitwiseXor = cv2.bitwise_xor(square, ellipse)
cv2.imshow('Xor',bitwiseXor)

bitwiseNot_sq = cv2.bitwise_not(square)
cv2.imshow('NOT - square',bitwiseNot_sq)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%%============================================================================
# CROPPING
# =============================================================================
image = cv2.imread(os.path.join(path, 'lena.png'))
height, width = image.shape[:2]
# lets get starting pixel coordinates (top left of cropping rectangle)
start_row, start_col = int(height * .25), int(width * .25)
# lets get the ending pixel coordinates (bottom right)
end_row, end_col = int(height * .75), int(height * .75)
# simply use indexing to crop out the rectangle we desire 
cropped = image[start_row:end_row, start_col:end_col]
while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow("Original Image",image)
    cv2.imshow('cropped image',cropped)
cv2.destroyAllWindows()
