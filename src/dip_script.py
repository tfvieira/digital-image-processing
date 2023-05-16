#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 10:48:19 2018

@author: tvieira
"""
#%% Import modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import PIL
import skimage
import os
from numpy.linalg import norm
import cv2
import tensorflow as tf
# import keras

#folder = '/home/tvieira/Dropbox/db/img/'
folder = '../img'
plt.rcParams['figure.figsize'] = (45.0, 32.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#%%
import sys
print('Python interpreter path: ' + sys.executable)

import os
print('Current working directory: ' + os.getcwd())

print(cv2.__name__ + "-" + str(cv2.__version__))
print(tf.__name__ + "-" + str(tf.__version__))
# print(keras.__name__ + "-" + str(keras.__version__))
print(matplotlib.__name__ + "-" + str(matplotlib.__version__))
print(sklearn.__name__ + "-" + str(sklearn.__version__))
print(skimage.__name__ + "-" + str(skimage.__version__))
print(PIL.__name__ + "-" + str(PIL.__version__))
print(pd.__name__ + "-" + str(pd.__version__))

#%% Define functions
bgr2rgb = lambda x : cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
bgr2hsv = lambda x : cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
bgr2gray = lambda x : cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
gray2bgr = lambda x : cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
bgr2lab = lambda x : cv2.cvtColor(x, cv2.COLOR_BGR2Lab)

#%% Define functions
def doNothing(x):
    pass



def printImgDims(img):
    print ('[Height, Width, Channels] = [' + 
           str(img.shape[0]) + ', ' + 
           str(img.shape[1]) + ', ' + 
           str(img.ndim) + ']')

def iminfo(img):
    print('################################################################################')
    print('# Image info:')
    print('Image type = ', img.dtype.name)
    printImgDims(img)
    print('Max value = ', img.max())
    print('Mean value = ', img.mean())
    print('Min value = ', img.min())
    print('\n\n')
    return None
    
def plotMultipleImgs(imlist, titles, rc = '22'):
    for i in range(len(imlist)):
        img = imlist[i]
        plt.subplot(rc + str(i+1))
        if img.ndim > 2:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(titles[i])
            
def plotContour(cont):
    b = np.array(cont)
    x = b[:, 0, 0]
    y = b[:, 0, 1]
    return x, y

def compute_piecewise_linear_val(val, r1, s1, r2, s2):
    output = 0
    if (0 <= val) and (val <= r1):
        output = (s1 / r1) * val
    if (r1 <= val) and (val <= r2):
        output = ((s2 - s1) / (r2 - r1)) * (val - r1) + s1
    if (r2 <= val) and (val <= 1):
        output = ((1 - s2) / (1 - r2)) * (val - r2) + s2
    return output

def createSaltAndPepperNoise(height = 100, width = 100, 
                             salt_prob = 0.05, pepper_prob = 0.05):
    """ Returns an image \in [-1, 1] containing salt (I = 1.0) and 
    pepper (I = -1.0) noise with respective probability distributions
    equal salt_prob and pepper_prob. Pixels without noise have values of 0.5.
    """
    img = np.zeros((height, width), np.float64)
    noise = np.random.rand(img.shape[0], img.shape[1])
    img[noise > 1 - salt_prob] = 1
    img[noise < pepper_prob] = -1
    return img

def addSaltAndPepperNoise2Img(img, salt_prob = 0.05, pepper_prob = 0.05):
    noise = createSaltAndPepperNoise(img.shape[0], img.shape[1],
                                     salt_prob, pepper_prob)
    img[noise == 1] = 255
    img[noise == -1] = 0    
    return img

def colorGrad(img):
    # Get image's BGR channels
    B, G, R = cv2.split(img)
    # Compute color derivatives
    dBx = cv2.Sobel(B, cv2.CV_32F, 1, 0)
    dBy = cv2.Sobel(B, cv2.CV_32F, 0, 1)
    dGx = cv2.Sobel(G, cv2.CV_32F, 1, 0)
    dGy = cv2.Sobel(G, cv2.CV_32F, 0, 1)
    dRx = cv2.Sobel(R, cv2.CV_32F, 1, 0)
    dRy = cv2.Sobel(R, cv2.CV_32F, 0, 1)
    # Compute inner products
    Gxx = np.multiply(dBx, dBx) + np.multiply(dGx, dGx) + np.multiply(dRx, dRx)
    Gyy = np.multiply(dBy, dBy) + np.multiply(dGy, dGy) + np.multiply(dRy, dRy)
    Gxy = np.multiply(dBx, dBy) + np.multiply(dGx, dGy) + np.multiply(dRx, dRy)
    
    Theta = 0.5 * np.arctan((2 * Gxy) / (Gxx - Gyy))
    F = np.nan_to_num(
            np.sqrt(0.5 * 
                    (Gxx + Gyy) + (Gxx - Gyy) * np.cos(2 * Theta) + 
                    2 * Gxy * np.sin(2 * Theta))
            )
    cv2.normalize(F, F, 0, 1, cv2.NORM_MINMAX)
    return F
    
def scaleImage2_uchar(src):
    if src.dtype != np.float32:
        src = np.float32(src)
    cv2.normalize(src, src, 1, 0, cv2.NORM_MINMAX)
    src = np.uint8(255 * src)
    return src

def createWhiteDisk(height = 100, width = 100, xc = 50, yc = 50, rc = 20):
    disk = np.zeros((height, width), np.float64)
    for x in range(disk.shape[0]):
        for y in range(disk.shape[1]):
            if (x - xc) * (x - xc) + (y - yc) * (y - yc) <= rc * rc:
                disk[x][y] = 1.0
    return disk

def createWhiteDisk2(height = 100, width = 100, xc = 50, yc = 50, rc = 20):
    xx, yy = np.meshgrid(range(height), range(width))
    img = np.array(
            ( (xx - xc)**2 + (yy - yc)**2 - rc**2  ) < 0).astype('float64')
    return img

def createCosineImage(height, width, freq, theta):
    img = np.zeros((height, width), dtype=np.float64)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img[x][y] = np.cos(
                    2 * np.pi * freq * (x * np.cos(theta) - y * np.sin(theta)))
    return img

def createCosineImage2(height, width, freq, theta):
    img = np.zeros((height, width), dtype=np.float64)
    xx, yy = np.meshgrid(range(height), range(width))
    theta = np.deg2rad(theta)
    rho = (xx * np.cos(theta) - yy * np.sin(theta))
    img[:] = np.cos(2 * np.pi * freq * rho)
    return img

def createSineImage2(height, width, freq, theta):
    img = np.zeros((height, width), dtype=np.float64)
    xx, yy = np.meshgrid(range(height), range(width))
    theta = np.deg2rad(theta)
    rho = (xx * np.cos(theta) - yy * np.sin(theta))
    img[:] = np.sin(2 * np.pi * freq * rho)
    return img

def applyLogTransform(img):
    img2 = np.copy(img)
    img2 += 1
    img2 = np.log(img2)
    return img2

def create2DGaussian(rows = 100, 
                     cols = 100, 
                     mx = 50, 
                     my = 50, 
                     sx = 10, 
                     sy = 100,
                     theta = 0):
    """
    Create an image (rows x cols) with a 2D Gaussian with
    mx, my means in the x and y directions and standard deviations
    sx, sy respectively. The Gaussian can also be rotate of theta
    radians in clockwise direction.
    """
    
    xx0, yy0 = np.meshgrid(range(cols), range(rows))
    xx0 -= mx
    yy0 -= my
    theta = np.deg2rad(theta)
    xx = xx0 * np.cos(theta) - yy0 * np.sin(theta)
    yy = xx0 * np.sin(theta) + yy0 * np.cos(theta)
    try:
        img = np.exp( - ((xx**2)/(2*sx**2) + 
                         (yy**2)/(2*sy**2)) )
    except ZeroDivisionError:
        img = np.zeros((rows, cols), dtype='float64')
            
    cv2.normalize(img, img, 1, 0, cv2.NORM_MINMAX)
    return img

def compute_histogram_1C(src):
    # Compute the histograms:
    b_hist = cv2.calcHist([src], [0], None, [256], [0, 256], True, False)

    # Draw the histograms for B, G and R
    hist_w = 512
    hist_h = 400
    bin_w = np.round(hist_w / 256)

    histImage = np.ones((hist_h, hist_w), np.uint8)

    # Normalize the result to [ 0, histImage.rows ]
    cv2.normalize(b_hist, b_hist, 0, histImage.shape[0], cv2.NORM_MINMAX)

    # Draw for each channel
    for i in range(1, 256):
        cv2.line(histImage, (int(bin_w * (i - 1)), int(hist_h - np.round(b_hist[i - 1]))),
                 (int(bin_w * i), int(hist_h - np.round(b_hist[i]))), 255, 2, cv2.LINE_8, 0)
    return histImage, b_hist

def compute_histogram_3C(src):

    b, g, r = cv2.split(src)

    # Compute the histograms:
    b_hist = cv2.calcHist([b], [0], None, [256], [0, 256], True, False)
    g_hist = cv2.calcHist([g], [0], None, [256], [0, 256], True, False)
    r_hist = cv2.calcHist([r], [0], None, [256], [0, 256], True, False)

    # Draw the histograms for B, G and R
    hist_w = 512
    hist_h = 200
    bin_w = np.round(hist_w / 256)

    histImage = np.zeros((hist_h, hist_w, 3), np.uint8)

    # Normalize the result to [ 0, histImage.rows ]
    cv2.normalize(b_hist, b_hist, 0, histImage.shape[0], cv2.NORM_MINMAX)
    cv2.normalize(g_hist, g_hist, 0, histImage.shape[0], cv2.NORM_MINMAX)
    cv2.normalize(r_hist, r_hist, 0, histImage.shape[0], cv2.NORM_MINMAX)

    # Draw for each channel
    for i in range(1, 256):
        cv2.line(histImage, (int(bin_w * (i - 1)), int(hist_h - np.round(b_hist[i - 1]))),
                 (int(bin_w * i), int(hist_h - np.round(b_hist[i]))), (255,0,0), 2, cv2.LINE_8, 0)
        cv2.line(histImage, (int(bin_w * (i - 1)), int(hist_h - np.round(g_hist[i - 1]))),
                 (int(bin_w * i), int(hist_h - np.round(g_hist[i]))), (0,255,0), 2, cv2.LINE_8, 0)
        cv2.line(histImage, (int(bin_w * (i - 1)), int(hist_h - np.round(r_hist[i - 1]))),
                 (int(bin_w * i), int(hist_h - np.round(r_hist[i]))), (0,0,255), 2, cv2.LINE_8, 0)
        
    return histImage#, b_hist, g_hist, r_hist

def show_imgs(imgs):
    i = 0
    for img in imgs:
        cv2.namedWindow(str(i), cv2.WINDOW_KEEPRATIO)
        cv2.imshow(str(i),img)
        i += 1
    while cv2.waitKey(0) != ord('q'):
        pass
    cv2.destroyAllWindows()

def skeletonize(src):
    size = np.size(src)
    skel = np.zeros(src.shape, np.uint8)
    sel = np.ones((3,3))
    done = False
    while (not done):
        eroded = cv2.erode(src, sel)
        tmp = cv2.dilate(eroded, sel)
        tmp = cv2.subtract(src, tmp)
        skel = cv2.bitwise_or(skel, tmp)
        src = eroded.copy()
        zeros = size - cv2.countNonZero(src)
        if zeros == size:
            done = True
    return skel

def thin(src):
    """
    Thin a binary image 'SRC' using a set of predefined kernels 'B'
    """
    b =  np.array([[[-1, -1, -1], [0, 1, 0], [1, 1, 1]],
               [[0, -1, -1], [1, 1, -1], [1, 1, 0]],
               [[1, 0, -1], [1, 1, -1], [1, 0, -1]],
               [[1, 1, 0], [1, 1, -1], [0, -1, -1]],
               [[1, 1, 1], [0, 1, 0], [-1, -1, -1]],
               [[0, 1, 1], [-1, 1, 1], [-1, -1, 0]],
               [[-1, 0, 1], [-1, 1, 1], [-1, 0, 1]],
               [[-1, -1, 0], [-1, 1, 1], [0, 1, 1]]
               ])
    f = src
    tmp = np.zeros(src.shape)
    while not (tmp == f).all():
        tmp = f
        bw = np.zeros(f.shape).astype(np.uint8)
        for kernel in b:
            bw = bw | cv2.morphologyEx(tmp, cv2.MORPH_HITMISS, kernel)
        f = tmp & ~bw
    return f

def ind2sub (i, rows = 5, cols = 3, axis = 1):
    """
    Returns row and column indexes corresponding to numerical value ind within 
    an array size limit of rows*cols
    
    Arguments:
    i -- Integer value must be ind < rows * cols
    rows -- Number of rows in 2D array.
    cols -- Number of cols in 2D array.
    
    Returns:
    r -- row index of position "i", counting from left-right and up-down.
    c -- column index of position "i", counting from left-right and up-down.
    
    Example usage:
    import numpy as np
    np.random.seed(0)
    a = np.random.randn(5,3)
    rows, cols = a.shape
    for i in range(0,rows*cols):
        r, c = ind2sub(i, rows, cols)
        print("i = ",i,"; r = ",r, "; c = ",c)
    """
    assert type(i) == int and type(rows) == int and type(cols) == int
    assert i < rows * cols
    
    return (i // cols, i % cols)

def sub2ind (r, c, rows = 5, cols = 3, axis = 1):
    """
    Returns index "i" corresponding to "row" r and "column" c within array size
    limits.
    
    Arguments:
    r -- row index.
    c -- column index.
    
    Returns:
    i -- index counting along axis "axis".
    
    Example usage:
    import numpy as np
    np.random.seed(0)
    a = np.random.randn(5,3)
    rows, cols = a.shape
    for r in range(rows):
    for c in range(cols):
        i = sub2ind (r, c, rows = 5, cols = 3)
        print("r = ",r, ", c = ",c, ", i = ",i)
    """
    assert type(r) == int and type(c) == int and type(rows) == int and type(cols) == int
    assert r < rows and c < cols
    return r * cols + c