# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 13:49:47 2021

@author: Vieira
"""
# %%===========================================================================
# Intensity Transformations - Part II
# 
# =============================================================================
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import im_info
from utils import compute_histogram_1C
from utils import compute_piecewise_linear_val
from utils import get_piecewise_transformed_img
from utils import do_nothing
from utils import norm_img
path = "../img/"


#%% Pice-wise linear transformation
# img = cv2.imread(os.path.join(path,"kidney.tif"), cv2.IMREAD_GRAYSCALE)
# img = cv2.imread(os.path.join(path,"aerial.tif"), cv2.IMREAD_GRAYSCALE)
img = cv2.imread(os.path.join(path,"pollen_washedout.tif"), cv2.IMREAD_GRAYSCALE)
img2 = np.copy(img)

T0 = 255 * np.ones((100,100), np.uint8)

r1 = 0
s1 = 0
r2 = 100
s2 = 100

cv2.namedWindow("Transformation", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img2", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("hist", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("hist2", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Transformation", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar("r1", "Transformation", r1, T0.shape[0] - 1, do_nothing)
cv2.createTrackbar("s1", "Transformation", s1, T0.shape[0] - 1, do_nothing)
cv2.createTrackbar("r2", "Transformation", r2, T0.shape[0] - 1, do_nothing)
cv2.createTrackbar("s2", "Transformation", s2, T0.shape[0] - 1, do_nothing)

while 0xFF & cv2.waitKey(1) != ord('q'):

    r1 = cv2.getTrackbarPos("r1", "Transformation")
    s1 = cv2.getTrackbarPos("s1", "Transformation")
    r2 = cv2.getTrackbarPos("r2", "Transformation")
    s2 = cv2.getTrackbarPos("s2", "Transformation")

    # Draw points and lines of the intensity transformation function
    T = np.copy(T0)
    p1 = (r1, T.shape[1] - 1 - s1)
    p2 = (r2, T.shape[1] - 1 - s2)
    cv2.line(T, (0, T.shape[0] - 1), p1, (0, 0, 0), 2, cv2.LINE_8, 0)
    cv2.circle(T, p1, 4, 0, 2, cv2.LINE_8, 0)
    cv2.line(T, p1, p2, 0, 2, cv2.LINE_8, 0)
    cv2.circle(T, p2, 4, 0, 2, cv2.LINE_8, 0)
    cv2.line(T, p2, (T.shape[0] - 1, 0), 0, 2, cv2.LINE_8, 0)

    img2 = get_piecewise_transformed_img (img, r1/100, s1/100, r2/100, s2/100)

    hist = compute_histogram_1C(img)
    hist2 = compute_histogram_1C(img2)
    
    cv2.imshow("img", img)
    cv2.imshow("img2", img2)
    cv2.imshow("hist", hist)
    cv2.imshow("hist2", hist2)
    cv2.imshow("Transformation", T)

cv2.destroyAllWindows()

# %%===========================================================================
# Spatial filtering
# =============================================================================

#%% Average Blurring
img = cv2.imread(os.path.join(path, 'lena.png'), cv2.IMREAD_GRAYSCALE)

cv2.namedWindow("Original", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("New", cv2.WINDOW_KEEPRATIO)

ksizex = 0
ksizey = 0

cv2.createTrackbar("ksizex", "New", ksizex, 63, do_nothing)
cv2.createTrackbar("ksizey", "New", ksizey, 63, do_nothing)

img2 = np.zeros(img.shape, dtype=np.float64)

while 0xFF & cv2.waitKey(1) != ord('q'):

    ksizey = cv2.getTrackbarPos("ksizey", "New")
    ksizex = cv2.getTrackbarPos("ksizex", "New")

    if ksizex < 1:
        ksizex = 1
    if ksizey < 1:
        ksizey = 1

    img2 = cv2.blur(img, (ksizex, ksizey), img2, (-1, -1), cv2.BORDER_DEFAULT)

    #cv2.imshow("Original", img)
    cv2.imshow("New", img2)

cv2.destroyAllWindows()


#%% Adding salt & pepper noise to an image and cleaning it using the median
img = cv2.imread(os.path.join(path, "lena.png"), cv2.IMREAD_GRAYSCALE)

noise = np.zeros(img.shape, np.uint8)
img2 = np.zeros(img.shape, np.uint8)
img3 = np.zeros(img.shape, np.uint8)
salt = np.zeros(img.shape, np.uint8)
pepper = np.zeros(img.shape, np.uint8)

ksize = 0
amount = 5
cv2.namedWindow("img3", cv2.WINDOW_KEEPRATIO);
cv2.namedWindow("img2", cv2.WINDOW_KEEPRATIO);
cv2.createTrackbar("ksize", "img3", ksize, 15, do_nothing)
cv2.createTrackbar("amount", "img2", amount, 120, do_nothing)

cv2.randu(noise, 0, 255)

while 0xFF & cv2.waitKey(1) != ord('q'):
    amount = cv2.getTrackbarPos("amount", "img2")
    ksize = cv2.getTrackbarPos("ksize", "img3")

    img2 = np.copy(img)

    salt = noise > 255 - amount
    pepper = noise < amount

    img2[salt == True] = 255
    img2[pepper == True] = 0

    img3 = cv2.medianBlur(img2, (ksize + 1) * 2 - 1)

    #cv2.imshow("img", img)
    cv2.imshow("img2", img2)
    cv2.imshow("img3", img3)

cv2.destroyAllWindows()

#%% First derivative operators - Sobel masks - Part I
img = cv2.imread(os.path.join(path, 'lena.png'), cv2.IMREAD_GRAYSCALE)
img2 = img.astype('float')

cv2.namedWindow("Original", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("New", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Gx", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Gy", cv2.WINDOW_KEEPRATIO)

kx = [[-1,  0,  1],
      [-2,  0,  2],
      [-1,  0,  1]]
kx = np.array(kx)

ky = [[-1, -2, -1],
      [ 0,  0,  0],
      [ 1,  2,  1]]
ky = np.array(ky)

gx = cv2.filter2D(img2, -1, kx, cv2.BORDER_DEFAULT)
gy = cv2.filter2D(img2, -1, ky, cv2.BORDER_DEFAULT)
g = np.abs(gx) + np.abs(gy)

gx = norm_img(gx)
gy = norm_img(gy)
g = norm_img(g)

cv2.namedWindow("Original", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("New", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Gx", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Gy", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow("Original", img)
    cv2.imshow("New", norm_img(g))
    cv2.imshow("Gx", norm_img(gx))
    cv2.imshow("Gy", norm_img(gy))

cv2.destroyAllWindows()

#%% First derivative operators - Sobel masks - Part II
img = cv2.imread(os.path.join(path, 'lena.png'), cv2.IMREAD_GRAYSCALE)

gx, gy = cv2.spatialGradient(img, ksize=3, borderType=cv2.BORDER_DEFAULT)
g = np.abs(gx) + np.abs(gy)

gx = norm_img(gx)
gy = norm_img(gy)
g = norm_img(g)

cv2.namedWindow("Original", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("New", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Gx", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Gy", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow("Original", img)
    cv2.imshow("New", g)
    cv2.imshow("Gx", gx)
    cv2.imshow("Gy", gy)

cv2.destroyAllWindows()

#%% First derivative operators - Sobel masks - Part III
img = cv2.imread(os.path.join(path, 'lena.png'), cv2.IMREAD_GRAYSCALE)

gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
g = np.abs(gx) + np.abs(gy)

gx = norm_img(gx)
gy = norm_img(gy)
g = norm_img(g)

while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow("Original", img)
    cv2.imshow("New", g)
    cv2.imshow("Gx", gx)
    cv2.imshow("Gy", gy)

cv2.destroyAllWindows()

#%% Image sharpening using the Laplacian operator - Part I
img = cv2.imread(os.path.join(path, 'lena.png'), cv2.IMREAD_GRAYSCALE)
img = img.astype('float')
kernel = np.array([[1.0,  1.0, 1.0],
                   [1.0, -8.0, 1.0],
                   [1.0,  1.0, 1.0]],
                   dtype='float')
factor = 0
cv2.namedWindow("img", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img2", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img3", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar("factor", "img3", factor, 1000, do_nothing)
while cv2.waitKey(1) != ord('q'):
    factor = cv2.getTrackbarPos("factor", "img3")
    img2 = cv2.filter2D(img, -1, kernel, cv2.BORDER_DEFAULT)
    img3 = img - (factor/1000.0) * img2
    np.clip(img3, 0, 255)
    cv2.imshow("img", norm_img(img.copy()))
    cv2.imshow("img2", norm_img(img2.copy()))
    cv2.imshow("img3", img3.copy()/255)
cv2.destroyAllWindows()

#%% Image sharpening using the Laplacian operator - Part II
img = cv2.imread(os.path.join(path, 'lena.png'), cv2.IMREAD_GRAYSCALE)
lap = cv2.Laplacian(img, ddepth=cv2.CV_32F, ksize=1, scale=1, delta=0)

cv2.namedWindow("img", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("lap", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
while cv2.waitKey(1) != ord('q'):
    cv2.imshow("img", img)
    cv2.imshow("lap", norm_img(lap.copy()))
cv2.destroyAllWindows()














