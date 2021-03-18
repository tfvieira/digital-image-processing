# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:33:47 2021

@author: Vieira
"""

#%%
import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt
from utils import do_nothing
from utils import im_info
path = "../img/"

# =============================================================================
# Intensity Transformations - Part I
# 
# =============================================================================

#%% Image negative
img = cv2.imread(os.path.join(path,'lena.png'), cv2.IMREAD_GRAYSCALE)

cv2.namedWindow("img", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img2", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
while cv2.waitKey(1) != ord('q'):
    cv2.imshow("img", img)
    cv2.imshow("img2", 255 - img)
cv2.destroyAllWindows()

im_info(img)

#%% Log Transform
img = cv2.imread(os.path.join(path,'spectrum.tif'), cv2.IMREAD_GRAYSCALE)
img2 = np.ones(img.shape, np.float64)
c = 1

cv2.namedWindow("img", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img2", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img3", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

tic = cv2.getTickCount()
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        intensity = img[x][y]
        intensity_new = c * np.log(1 + intensity)
        img2[x][y] = intensity_new
cv2.normalize(img2, img2, 1, 0, cv2.NORM_MINMAX)
toc = cv2.getTickCount()
et = (toc - tic)/cv2.getTickFrequency()
print(f"Elapsed time using for loop: {1000*et:.2f}ms")

tic = cv2.getTickCount()
img3 = c * np.log(1 + img.astype('float'))
cv2.normalize(img3, img3, 1, 0,  cv2.NORM_MINMAX)
toc = cv2.getTickCount()
et = (toc - tic)/cv2.getTickFrequency()
print(f"Elapsed time using for Numpy Array: {1000*et:.2f}ms")

while cv2.waitKey(1) != ord('q'):
    cv2.imshow("img", img)
    cv2.imshow("img2", img2)
    cv2.imshow("img3", img3)
cv2.destroyAllWindows()

#%% Interactive Intensity transform
img = cv2.imread(os.path.join(path, 'spectrum.tif'), cv2.IMREAD_GRAYSCALE)
img2 = np.ones(img.shape, np.uint8)
cv2.namedWindow("img", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img2", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

val_max = 1.0
n = 1000
n_max = 1000

cv2.createTrackbar("n", "img2", n, n_max, do_nothing)
while 0xFF & cv2.waitKey(1) != ord('q'):
    n = cv2.getTrackbarPos("n", "img2")
    # for x in range(img.shape[0]):
    #     for y in range(img.shape[1]):
    #         intensity = img[x][y]
    #         intensity_new = np.power(intensity, n)
    img2 = np.power(img.astype('float'), val_max * n/n_max)
    cv2.normalize(img2, img2, 1, 0, cv2.NORM_MINMAX)
    cv2.imshow("img", img)
    cv2.imshow("img2", img2)
cv2.destroyAllWindows()