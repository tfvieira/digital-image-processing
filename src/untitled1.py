#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:38:36 2021

@author: vieira
"""

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
img = 127 * np.ones((50, 50))

plt.imshow(img, cmap='gray')
plt.imshow(img, cmap='gray', vmin=0, vmax=255)

#%% Initializing a grayscale image with random values, uniformly distributed
img = np.ones((250, 250), dtype=np.uint8)
cv2.randu(img, 0, 255)

plt.figure(figsize=(32,16))
plt.subplot(121); plt.title("Original"); plt.imshow(img, cmap='gray')
plt.subplot(122); plt.title("Histogram"); plt.hist(img.ravel(), 256, [0, 256])
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
plt.subplot(121); plt.title("Original"); plt.imshow(img)
plt.subplot(122); plt.title("Histogram"); plot_histogram(img)
plt.show()

#%% Initializing a grayscale image with random values, normally distributed
img = np.ones((250, 250), dtype=np.uint8)
cv2.randn(img, 127, 40)

plt.figure(figsize=(32, 16))
plt.subplot("221"); plt.title("Original"); plt.imshow(img, cmap='gray')
plt.subplot("222"); plt.title("Histogram"); plt.hist(img.ravel(), 256, [0, 256])
plt.show()

#%% highgui
cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)

while True:

    # Process
    cv2.randn(img, 127, 40)
    
    cv2.imshow("img", img)

    key = 0xFF & cv2.waitKey(1)

    if key == ord('q'):
        break
    
    elif key == ord('s'):
        cv2.imwrite("img.png", img)


cv2.destroyAllWindows()


#%%
img = np.zeros((50, 50, 3), dtype=np.uint8)
cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)

while True:

    # Process
    bgr = cv2.split(img)
    cv2.randn(bgr[0], 127, 40)
    cv2.randn(bgr[1], 127, 40)
    cv2.randn(bgr[2], 127, 40)
    img = cv2.merge(bgr)

    
    cv2.imshow("img", img)

    key = 0xFF & cv2.waitKey(1)

    if key == ord('q'):
        break
    
    elif key == ord('s'):
        cv2.imwrite("img.png", img)










