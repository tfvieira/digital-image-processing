#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:24:49 2019

@author: tvieira
"""

#%%
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread(os.path.join('../img','water_coins.jpg'))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

ws = cv2.watershed(img,markers)

res = img.copy()
res[markers == -1] = [255,0,0]

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

#%%
plt.figure(1, figsize=(56,64)), plt.clf()
plt.subplot(331), plt.imshow(rgb), plt.title('Original', fontsize=64), plt.axis('off')
plt.subplot(332), plt.imshow(thresh, cmap='gray'), plt.title('Threshold', fontsize=64), plt.axis('off')
plt.subplot(333), plt.imshow(sure_bg, cmap='gray'), plt.title('Sure_bg', fontsize=64), plt.axis('off')
plt.subplot(334), plt.imshow(dist_transform, cmap='gray'), plt.title('Distance Transform', fontsize=64), plt.axis('off')
plt.subplot(335), plt.imshow(sure_fg, cmap='gray'), plt.title('Sure_fg', fontsize=64), plt.axis('off')
plt.subplot(336), plt.imshow(unknown, cmap='gray'), plt.title('Unknown', fontsize=64), plt.axis('off')
plt.subplot(337), plt.imshow(markers, cmap='jet'), plt.title('Markers', fontsize=64), plt.axis('off')
plt.subplot(338), plt.imshow(ws, cmap='jet'), plt.title('Watershed result', fontsize=64), plt.axis('off')
plt.subplot(339), plt.imshow(res_rgb, cmap='jet'), plt.title('Segmentation result', fontsize=64), plt.axis('off')
plt.show()

