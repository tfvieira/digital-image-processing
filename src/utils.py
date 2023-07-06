#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:56:09 2019

@author: tvieira
"""
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def normalize_img(img):
    return cv2.normalize(img.astype('float64'), None, 1, 0, cv2.NORM_MINMAX)

def isgray(img):
    return len(img.shape) < 3


def do_nothing(x):
    pass


def plot_grayscale_histogram(gray_img):
    plt.hist(gray_img.ravel(), 256, [0, 256])
    plt.show()

def plot_rgb_histogram(rgb_img):
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        hist = cv2.calcHist([rgb_img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.show()
    return None
    
def plot_histogram(img, ax=[]):
    # Grayscale image
    if isgray(img):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        if ax != []:
            ax.plot(hist, color='k')
            ax.show()
        else:
            plt.plot(hist, color='k')
            plt.show()
    # Color image
    else:
        color = ('r', 'g', 'b')
        for i, col in enumerate(color):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            if ax != []:
                ax.plot(hist, color=col)
                ax.set_xlim([0, 256])                
            else:
                plt.plot(hist, color=col)
                plt.xlim([0, 256])
    if ax == []:
        plt.show()

    return None

def create_noisy_img(shape, noise_type = 'uniform', a=127, b=40):
    tmp = np.zeros(shape, dtype=np.uint8)
    if noise_type == 'uniform':
        noise = cv2.merge([cv2.randu(x, a, b) for x in cv2.split(tmp)])
    elif noise_type == 'normal':
        noise = cv2.merge([cv2.randn(x, a, b) for x in cv2.split(tmp)])
    return noise

def im_info(img):
    print('# Image info:')
    print(f'Image type = {img.dtype.name}')
    h, w = img.shape
    d = img.ndim
    print(f'[Height, Width, Dimensions] = [{h}, {w}, {d}]')
    print(f'Max value = {img.max()}')
    print(f'Mean value = {img.mean():.2f}')
    print(f'Min value = {img.min()}')
    print('\n\n')
    return None

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
    return histImage#, b_hist

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

def compute_piecewise_linear_val(val, r1, s1, r2, s2):
    output = 0
    if (0 <= val) and (val <= r1):
        output = (s1 / r1) * val
    elif (r1 <= val) and (val <= r2):
        output = ((s2 - s1) / (r2 - r1)) * (val - r1) + s1
    elif (r2 <= val) and (val <= 1):
        output = ((1 - s2) / (1 - r2)) * (val - r2) + s2

    return output

def get_piecewise_transformed_img (img, r1, s1, r2, s2):
    
    if img.dtype == 'uint8':
        out = img/255.
    else:
        img.copy()
    
    eps = 1e-16
    
    mask1 = out <= r1
    mask2 = np.bitwise_and(r1 < out, out <= r2)
    mask3 = out > r2
    #Opitionally check whether masks are mutually exclusive:
    mask = np.bitwise_xor(mask1, np.bitwise_xor(mask2, mask3))
    print(f"Mutually exclusive masks? {mask.all()}")
    
    out[mask1] = out[mask1] * (s1 / (r1 + eps))
    out[mask2] = s1 + (out[mask2] - r1) * ((s2 - s1)/(r2 - r1 + eps))
    out[mask3] = s2 + (out[mask3] - r2) / (1 - r2 + eps)
    
    out = np.clip(out, 0, 1.)
    out = 255*out
    out = out.astype('uint8')
    
    return out

def log_transform(img):
    img2 = np.copy(img)
    img2 = np.log(1 + img2)
    return img2


def create_white_disk(height = 100, width = 100, xc = 50, yc = 50, rc = 20):
    xx, yy = np.meshgrid(range(height), range(width))
    img = np.array(
            ( (xx - xc)**2 + (yy - yc)**2 - rc**2  ) < 0).astype('float64')
    return img

def bgr2rgb(img):
    """
    Convert image color from BGR to RGB
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def color_gradient(img):
    
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
    F = np.sqrt(0.5 * (Gxx + Gyy) + (Gxx - Gyy) * np.cos(2 * Theta) + 2 * Gxy * np.sin(2 * Theta))
    
    F = cv2.normalize(F.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
    return F

def plot_multiple_images(imlist, titles=[], ncols = 2):

    plt.figure(figsize=(16,16))
    # a = np.random.random((5,3))

    # imlist = [a, a, a, a, a]

    # ncols = 7
    nimgs = len(imlist)
    nrows = math.ceil(nimgs/ncols)

    # titles = [str(k+1) for k in range(nimgs)]

    # print(f'nimgs = {nimgs}')
    # print(f'ncols = {ncols}')
    # print(f'nrows = {nrows}')

    for r in range(nrows):

        for c in range(ncols):

            i = min(r*ncols + c, nimgs - 1)

            img = imlist[i]

            s = str(nrows) + str(ncols) + str(i+1)
            plt.subplot(int(s))

            if isgray(img):
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(bgr2rgb(img))
            
            if titles != []:
                plt.title(titles[i])





































def createCosineImage2(height, width, freq, theta):
    img = np.zeros((height, width), dtype=np.float64)
    xx, yy = np.meshgrid(range(height), range(width))
    theta = np.deg2rad(theta)
    rho = (xx * np.cos(theta) - yy * np.sin(theta))
    img[:] = np.cos(2 * np.pi * freq * rho)
    return img

def createWhiteDisk2(height = 100, width = 100, xc = 50, yc = 50, rc = 20):
    xx, yy = np.meshgrid(range(height), range(width))
    img = np.array(
            ( (xx - xc)**2 + (yy - yc)**2 - rc**2  ) < 0).astype('float64')
    return img

def applyLogTransform(img):
    img2 = np.copy(img)
    img2 += 1
    img2 = np.log(img2)
    return img2

def scaleImage2_uchar(src):
    if src.dtype != np.float32:
        src = np.float32(src)
    cv2.normalize(src, src, 1, 0, cv2.NORM_MINMAX)
    src = np.uint8(255 * src)
    return src


def create_2D_gaussian(
    shape = (100, 100), 
    mx = 50, 
    my = 50, 
    sx = 10, 
    sy = 10,
    theta = 0):
    """
    Create an image with shape = (rows x cols) with a 2D Gaussian with
    mx, my means in the x and y directions and standard deviations
    sx, sy respectively. The Gaussian can also be rotate of theta
    radians in clockwise direction.

    Example usage:
    g = create_2D_gaussian(
        shape = (500, 1000), 
        mx = 5000, 
        my = 250, 
        sx = 60, 
        sy = 20,
        theta = -30
        )
    """
    
    xx0, yy0 = np.meshgrid(range(shape[1]), range(shape[0]))
    xx0 -= mx
    yy0 -= my
    theta = np.deg2rad(theta)
    xx = xx0 * np.cos(theta) - yy0 * np.sin(theta)
    yy = xx0 * np.sin(theta) + yy0 * np.cos(theta)
    try:
        img = np.exp( - ((xx**2)/(2*sx**2) + 
                         (yy**2)/(2*sy**2)) )
    except ZeroDivisionError:
        img = np.zeros((shape[0], shape[1]), dtype='float64')

    return cv2.normalize(img.astype('float'), None, 1, 0, cv2.NORM_MINMAX)

