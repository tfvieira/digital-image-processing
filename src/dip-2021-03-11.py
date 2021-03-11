# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 13:19:27 2018

@author: Vieira
"""

#%%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_histogram
from utils import create_noisy_img
path = "../img/"

# %%===========================================================================
# LINEAR AND NONLINEAR OPERATIONS
# =============================================================================

#%% Adding 'uint8' scalars
x = np.uint8([250])
y = np.uint8([10])

print(x+y)
print( cv2.add(x,y) )

#%% Add a scalar to an image
img = cv2.imread(os.path.join(path,"lena.png"), cv2.IMREAD_COLOR)
val = 50

img2 = img + val
# img2 = img.astype('float') + val
# img2[img2 > 255] = 255
# img2 = img2.astype('uint8')

img3 = cv2.add(img, val)

plt.figure(figsize=(32,32))
plt.subplot("221"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot("222"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.subplot("223"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot("224"); plt.title("IMG 3"); plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.show()

#%% Add two images
img = cv2.imread(os.path.join(path,"lena.png"), cv2.IMREAD_COLOR)
img2 = cv2.imread(os.path.join(path,"baboon.png"), cv2.IMREAD_COLOR)
img3 = cv2.add(img, img2)

plt.subplot("131"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot("132"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.subplot("133"); plt.title("IMG 3"); plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.show()

#%% Max operator
img = cv2.imread(os.path.join(path,"lena.png"), cv2.IMREAD_COLOR)
img2 = cv2.imread(os.path.join(path,"baboon.png"), cv2.IMREAD_COLOR)
img3 = cv2.max(img, img2)

plt.subplot("131"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot("132"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.subplot("133"); plt.title("IMG 3"); plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.show()

#%% Absolute image diferencing
img = cv2.imread(os.path.join(path,"lena.png"), cv2.IMREAD_COLOR)
img2 = cv2.imread(os.path.join(path,"baboon.png"), cv2.IMREAD_COLOR)
img3 = cv2.absdiff(img, img2)

plt.subplot("131"); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot("132"); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.subplot("133"); plt.title("IMG 3"); plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.show()

#%% Image difference
cap = cv2.VideoCapture(0)

while cv2.waitKey(1) != ord('q'):
    _, frame1 = cap.read()
    _, frame2 = cap.read()

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray2, gray1)
    cv2.imshow('Gray 1', gray1)
    # cv2.imshow('Gray 2', gray2)
    cv2.imshow('DIFF', diff)

cap.release()
cv2.destroyAllWindows()

#%% Adding noise to an image
img = cv2.imread(os.path.join(path,"lena.png"), cv2.IMREAD_COLOR)
noise = create_noisy_img(img.shape, noise_type='uniform', a=20, b = 60)
img2 = cv2.add(img, noise)

plt.figure(figsize=(120,60))
plt.subplot(131); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(132); plt.imshow(cv2.cvtColor(noise, cv2.COLOR_BGR2RGB))
plt.subplot(133); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.show()

plot_histogram(img)
plot_histogram(noise)
plot_histogram(img2)


#%% Adding noise to an image containing negative values
img = cv2.imread(os.path.join(path,"lena.png"), cv2.IMREAD_COLOR)
mu, std = 0, 200
noise = mu + std*np.random.normal(0, 1, img.shape)
img2 = img.astype('float') + noise
img2 = img2.clip(min=0, max=255).astype('uint8')

fig = plt.figure(figsize=(120,60))
plt.subplot(131)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(132)
plt.title('Noise')
plt.imshow(cv2.cvtColor(noise.clip(min=0,max=255).astype('uint8'), cv2.COLOR_BGR2RGB))
plt.subplot(133)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.show()


#%% Adding salt & pepper noise to an image
img = cv2.imread(os.path.join(path,"lena.png"), cv2.IMREAD_COLOR)
noise = np.zeros((img.shape[0], img.shape[1]), img.dtype)
cv2.randu(noise, 0, 255)
salt = noise > 250
pepper = noise < 5
img2 = img.copy()
img2[salt == True] = np.array([255, 255, 255])
img2[pepper == True] = np.array([0, 0, 0])

plt.figure(figsize=(16,9))
plt.subplot("121"); plt.title("IMG 1");   plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot("122"); plt.title("HIST 1");  plot_histogram(img)
plt.show()

plt.figure(figsize=(16,9))
plt.subplot("121"); plt.title("IMG 2");   plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.subplot("122"); plt.title("HIST 2");  plot_histogram(img2)
plt.show()

#%% Set operations
img = cv2.imread(os.path.join(path,"utk.tif"), cv2.IMREAD_COLOR)
img2 = cv2.imread(os.path.join(path,"gt.tif"), cv2.IMREAD_COLOR)
and_img = img & img2
or_img = img | img2
not_img = ~img
plt.figure(figsize=(32,32))
plt.subplot("151"); plt.title("IMG 1"); plt.imshow(img, 'gray')
plt.subplot("152"); plt.title("IMG 2"); plt.imshow(img2, 'gray')
plt.subplot("153"); plt.title("AND"); plt.imshow(and_img, 'gray')
plt.subplot("154"); plt.title("OR"); plt.imshow(or_img, 'gray')
plt.subplot("155"); plt.title("NOT"); plt.imshow(not_img, 'gray')
plt.show()
