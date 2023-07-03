#%% Import libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from numpy.fft import fft2, fftshift
from utils import  do_nothing
from utils import create_white_disk
# from utils import norm_img
from utils import compute_piecewise_linear_val
# from utils import color_gradient
PATH_TO_IMAGES = '../img'

#%% Load a color image and visualize each channel separately
img = cv2.imread(os.path.join(PATH_TO_IMAGES,'baboon.png'), cv2.IMREAD_COLOR)
bgr = cv2.split(img)

plt.figure(figsize=(32,32))
plt.subplot('221'); plt.title('B'); plt.imshow(bgr[0], 'gray')
plt.subplot('222'); plt.title('G'); plt.imshow(bgr[1], 'gray')
plt.subplot('223'); plt.title('R'); plt.imshow(bgr[2], 'gray')
plt.subplot('224'); plt.title('Original'); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

#%% Load a color image and visualize each channel separately
img = cv2.imread(os.path.join(PATH_TO_IMAGES,'baboon.png'), cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

r = [1, 0, 0] # red
g = [0, 1, 0] # green
b = [0, 0, 1] # blue

c = [0, 1, 1] # cyan
m = [1, 0, 1] # magenta
y = [1, 1, 0] # yellow

img_r = r * rgb
img_g = g * rgb
img_b = b * rgb

img_c = c * rgb
img_m = m * rgb
img_y = y * rgb

plt.figure(figsize=(32,32))
plt.subplot(241); plt.title('RGB'); plt.imshow(rgb)
plt.subplot(242); plt.title('R'); plt.imshow(img_r)
plt.subplot(243); plt.title('G'); plt.imshow(img_g)
plt.subplot(244); plt.title('B'); plt.imshow(img_b)
plt.subplot(246); plt.title('C'); plt.imshow(img_c)
plt.subplot(247); plt.title('M'); plt.imshow(img_m)
plt.subplot(248); plt.title('Y'); plt.imshow(img_y)
plt.show()

#%% Load a color image and visualize each channel separately
img = cv2.imread(os.path.join(PATH_TO_IMAGES, 'baboon.png'), cv2.IMREAD_COLOR)
bgr = cv2.split(img)
colormap = cv2.COLORMAP_JET
#
plt.figure(figsize=(32,32))
plt.subplot(221); plt.title('B'); plt.imshow(cv2.applyColorMap(bgr[0], colormap))
plt.subplot(222); plt.title('G'); plt.imshow(cv2.applyColorMap(bgr[1], colormap))
plt.subplot(223); plt.title('R'); plt.imshow(cv2.applyColorMap(bgr[2], colormap))
plt.subplot(224); plt.title('Original'); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

#%% Load a color image and visualize each channel separately
img = cv2.imread(os.path.join(PATH_TO_IMAGES,'rgbcube_kBKG.png'), cv2.IMREAD_COLOR)
bgr = cv2.split(img)
#colormap = cv2.COLORMAP_JET
colormap = 1

plt.figure(figsize=(32,32))
plt.subplot(221); plt.title('B'); plt.imshow(cv2.applyColorMap(bgr[0], colormap))
plt.subplot(222); plt.title('G'); plt.imshow(cv2.applyColorMap(bgr[1], colormap))
plt.subplot(223); plt.title('R'); plt.imshow(cv2.applyColorMap(bgr[2], colormap))
plt.subplot(224); plt.title('Original'); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

#%% Load a color image and visualize each channel separately
img = cv2.imread(os.path.join(PATH_TO_IMAGES,'rgbcube_kBKG.png'), cv2.IMREAD_COLOR)
img = img
bgr = cv2.split(img)
colormap = cv2.COLORMAP_JET
#
plt.figure(figsize=(32,32))
plt.subplot(221); plt.title('B'); plt.imshow(cv2.applyColorMap(bgr[0], colormap))
plt.subplot(222); plt.title('G'); plt.imshow(cv2.applyColorMap(bgr[1], colormap))
plt.subplot(223); plt.title('R'); plt.imshow(cv2.applyColorMap(bgr[2], colormap))
plt.subplot(224); plt.title('Original'); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

#%% Converting between color spaces - BGR to YRB - Part I
# NTSC colorspace - Part I
img = cv2.imread(os.path.join(PATH_TO_IMAGES,'baboon.png'), cv2.IMREAD_COLOR)

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
yrb = cv2.split(img2)

while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow('img', img)
    cv2.imshow('img2', img2)
    cv2.imshow('y', yrb[0])
    cv2.imshow('r', yrb[1])
    cv2.imshow('b', yrb[2])
    
cv2.destroyAllWindows()

#%% Converting between color spaces - BGR to YRB - Part II
# NTSC colorspace
img = cv2.imread(os.path.join(PATH_TO_IMAGES,'baboon.png'), cv2.IMREAD_COLOR)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
yrb  = cv2.split(img2)
diff = cv2.absdiff(gray, yrb[0])
#
while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow('img', img)
    cv2.imshow('gray', gray)
    cv2.imshow('diff', diff)
    cv2.imshow('y', yrb[0])
    cv2.imshow('r', yrb[1])
    cv2.imshow('b', yrb[2])

cv2.destroyAllWindows()

#%% The HSV colorspace - Part I
img = cv2.imread(os.path.join(PATH_TO_IMAGES,'rgbcube_kBKG.png'), cv2.IMREAD_COLOR)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
v = cv2.normalize(v.astype('float'), None, 1, 0, cv2.NORM_MINMAX)

while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow('img', img)
    cv2.imshow('hsv', hsv)
    cv2.imshow('h', h)
    cv2.imshow('s', s)
    cv2.imshow('v', v)
    
cv2.destroyAllWindows()

#%% The HSV colorspace - Part II
img = cv2.imread(os.path.join(PATH_TO_IMAGES,'baboon.png'), cv2.IMREAD_COLOR)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
h, s, v = cv2.split(hsv)

colormap = 1 #cv2.COLORMAP_JET

while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow('img', img)
    cv2.imshow('hsv', hsv)
    cv2.imshow('h', cv2.applyColorMap(h, colormap))
    cv2.imshow('s', cv2.applyColorMap(s, colormap))
    cv2.imshow('v', cv2.applyColorMap(v, colormap))

cv2.destroyAllWindows()

#%% Converting between color spaces - BGR to HSV - Part II
# HSV colorspace
img = cv2.imread(os.path.join(PATH_TO_IMAGES,'chips.png'), cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
h, s, v = cv2.split(hsv)
while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow('img', img)
    cv2.imshow('hsv', hsv)
    cv2.imshow('h', h)
    cv2.imshow('s', s)
    cv2.imshow('v', cv2.normalize(v.astype('float'), None, 1, 0, cv2.NORM_MINMAX))

cv2.destroyAllWindows()

#%%
plt.figure(figsize=(16,9))
plt.hist(h.ravel(), 256, [0, 256], histtype='step', linestyle=('solid'), color=('black'))
plt.hist(s.ravel(), 256, [0, 256], histtype='step', linestyle=('solid'), color=('blue' ))
plt.legend(['Hue', 'Saturation'])
plt.show()

#%% The CMYK colorspace
img = cv2.imread(os.path.join(PATH_TO_IMAGES,'baboon.png'), cv2.IMREAD_COLOR)
img2 = 255 - img
ymc = cv2.split(img2)
colormap = cv2.COLORMAP_JET
while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow('img', img)
    cv2.imshow('img2', img2)
    cv2.imshow('y', ymc[0])
    cv2.imshow('m', ymc[1])
    cv2.imshow('c', ymc[2])
    
cv2.destroyAllWindows()

#%% Creating BGR color disks
rows = 1e3
radius = rows/4
bx = rows/2
by = rows/2 - radius/2
gx = rows/2 - radius/2
gy = rows/2 + radius/2
rx = rows/2 + radius/2
ry = rows/2 + radius/2
#
bgr = [
       create_white_disk(int(rows), int(rows), int(bx), int(by), int(radius)),
       create_white_disk(int(rows), int(rows), int(gx), int(gy), int(radius)),
       create_white_disk(int(rows), int(rows), int(rx), int(ry), int(radius))
      ]

# bgr = cv2.merge(bgr)
# plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
# plt.show()

cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
while 0xFF & cv2.waitKey(1) != ord('q'):
    img = cv2.merge(bgr)
    img = cv2.normalize(img.astype('float'), None, 1, 0, cv2.NORM_MINMAX)
    cv2.imshow('img', img)
cv2.destroyAllWindows()

#%% Create CMYK color disks
rows = 1e3
radius = rows/4
bx = rows/2
by = rows/2 - radius/2
gx = rows/2 - radius/2
gy = rows/2 + radius/2
rx = rows/2 + radius/2
ry = rows/2 + radius/2

bgr = [
       create_white_disk(int(rows), int(rows), int(bx), int(by), int(radius)),
       create_white_disk(int(rows), int(rows), int(gx), int(gy), int(radius)),
       create_white_disk(int(rows), int(rows), int(rx), int(ry), int(radius))
      ]
img = cv2.merge(bgr)
img = cv2.normalize(img.astype('float'), None, 1, 0, cv2.NORM_MINMAX)
cmy = 1.0 - img

cv2.namedWindow('cmy', cv2.WINDOW_KEEPRATIO)
while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow('img', img)
    cv2.imshow('cmy', cmy)
cv2.destroyAllWindows()

#%%============================================================================
# Intensity transformation of color images.
# Pice-wise linear transformation of color channels
# 
# =============================================================================
img = cv2.imread(os.path.join(PATH_TO_IMAGES,'baboon.png'), cv2.IMREAD_COLOR)

cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('img2', cv2.WINDOW_KEEPRATIO)

# img = cv2.resize(img, (128, 128))

T0 = 255 * np.ones((256, 256), np.uint8)

b_x1 = 65
b_y1 = 65
b_x2 = 195
b_y2 = 195

g_x1 = 65
g_y1 = 65
g_x2 = 195
g_y2 = 195

r_x1 = 65
r_y1 = 65
r_x2 = 195
r_y2 = 195

cv2.namedWindow('B_transform', cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('b_x1', 'B_transform', b_x1, T0.shape[1] - 1, do_nothing)
cv2.createTrackbar('b_y1', 'B_transform', b_y1, T0.shape[1] - 1, do_nothing)
cv2.createTrackbar('b_x2', 'B_transform', b_x2, T0.shape[1] - 1, do_nothing)
cv2.createTrackbar('b_y2', 'B_transform', b_y2, T0.shape[1] - 1, do_nothing)

cv2.namedWindow('G_transform', cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('g_x1', 'G_transform', g_x1, T0.shape[1] - 1, do_nothing)
cv2.createTrackbar('g_y1', 'G_transform', g_y1, T0.shape[1] - 1, do_nothing)
cv2.createTrackbar('g_x2', 'G_transform', g_x2, T0.shape[1] - 1, do_nothing)
cv2.createTrackbar('g_y2', 'G_transform', g_y2, T0.shape[1] - 1, do_nothing)

cv2.namedWindow('R_transform', cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('r_x1', 'R_transform', r_x1, T0.shape[1] - 1, do_nothing)
cv2.createTrackbar('r_y1', 'R_transform', r_y1, T0.shape[1] - 1, do_nothing)
cv2.createTrackbar('r_x2', 'R_transform', r_x2, T0.shape[1] - 1, do_nothing)
cv2.createTrackbar('r_y2', 'R_transform', r_y2, T0.shape[1] - 1, do_nothing)

while cv2.waitKey(1) != ord('q'):

    b_x1 = cv2.getTrackbarPos('b_x1', 'B_transform')
    b_y1 = cv2.getTrackbarPos('b_y1', 'B_transform')
    b_x2 = cv2.getTrackbarPos('b_x2', 'B_transform')
    b_y2 = cv2.getTrackbarPos('b_y2', 'B_transform')

    g_x1 = cv2.getTrackbarPos('g_x1', 'G_transform')
    g_y1 = cv2.getTrackbarPos('g_y1', 'G_transform')
    g_x2 = cv2.getTrackbarPos('g_x2', 'G_transform')
    g_y2 = cv2.getTrackbarPos('g_y2', 'G_transform')

    r_x1 = cv2.getTrackbarPos('r_x1', 'R_transform')
    r_y1 = cv2.getTrackbarPos('r_y1', 'R_transform')
    r_x2 = cv2.getTrackbarPos('r_x2', 'R_transform')
    r_y2 = cv2.getTrackbarPos('r_y2', 'R_transform')

    #Draw the transformation function for B channel
    T_B = np.copy(T0)
    p1 = (b_x1, T_B.shape[0] - 1 - b_y1)
    p2 = (b_x2, T_B.shape[0] - 1 - b_y2)
    cv2.line(T_B, (0, T_B.shape[1] - 1), p1, (0,0,0), 2, cv2.LINE_8, 0)
    cv2.circle(T_B, p1, 4, 0, 2, cv2.LINE_8, 0)
    cv2.line(T_B, p1, p2, 0, 2, cv2.LINE_8, 0)
    cv2.circle(T_B, p2, 4, 0, 2, cv2.LINE_8, 0)
    cv2.line(T_B, p2, (T_B.shape[1] - 1, 0), 0, 2, cv2.LINE_8, 0)
    #Draw the transformation function for G channel
    T_G = np.copy(T0)
    p1 = (g_x1, T_G.shape[0] - 1 - g_y1)
    p2 = (g_x2, T_G.shape[0] - 1 - g_y2)
    cv2.line(T_G, (0, T_G.shape[1] - 1), p1, (0,0,0), 2, cv2.LINE_8, 0)
    cv2.circle(T_G, p1, 4, 0, 2, cv2.LINE_8, 0)
    cv2.line(T_G, p1, p2, 0, 2, cv2.LINE_8, 0)
    cv2.circle(T_G, p2, 4, 0, 2, cv2.LINE_8, 0)
    cv2.line(T_G, p2, (T_G.shape[1] - 1, 0), 0, 2, cv2.LINE_8, 0)
    #Draw the transformation function for R channel
    T_R = np.copy(T0)
    p1 = (r_x1, T_R.shape[0] - 1 - r_y1)
    p2 = (r_x2, T_R.shape[0] - 1 - r_y2)
    cv2.line(T_R, (0, T_R.shape[1] - 1), p1, (0,0,0), 2, cv2.LINE_8, 0)
    cv2.circle(T_R, p1, 4, 0, 2, cv2.LINE_8, 0)
    cv2.line(T_R, p1, p2, 0, 2, cv2.LINE_8, 0)
    cv2.circle(T_R, p2, 4, 0, 2, cv2.LINE_8, 0)
    cv2.line(T_R, p2, (T_R.shape[1] - 1, 0), 0, 2, cv2.LINE_8, 0)

    #Clone the original image
    img2 = np.copy(img)

    #Split its channels
    bgr = cv2.split(img2)
    B = bgr[0]
    G = bgr[1]
    R = bgr[2]

    for x in range(0, img2.shape[1]):
        for y in range(0, img2.shape[0]):
            B[x, y] = 255 * compute_piecewise_linear_val(B[x, y] / 255.0,
                                                                  (b_x1 / 255.0),
                                                                  (b_y1 / 255.0),
                                                                  (b_x2 / 255.0),
                                                                  (b_y2 / 255.0))

            G[x, y] = 255 * compute_piecewise_linear_val(G[x, y] / 255.0,
                                                                  (g_x1 / 255.0),
                                                                  (g_y1 / 255.0),
                                                                  (g_x2 / 255.0),
                                                                  (g_y2 / 255.0))

            R[x, y] = 255 * compute_piecewise_linear_val(R[x, y] / 255.0,
                                                                  (r_x1 / 255.0),
                                                                  (r_y1 / 255.0),
                                                                  (r_x2 / 255.0),
                                                                  (r_y2 / 255.0))

    img2 = cv2.merge(bgr)

    cv2.imshow('img', img)
    cv2.imshow('img2', img2)
    cv2.imshow('B_transform', T_B)
    cv2.imshow('G_transform', T_G)
    cv2.imshow('R_transform', T_R)

cv2.destroyAllWindows()

#%% Spatial transformation - Blurring
img = cv2.imread(os.path.join(PATH_TO_IMAGES,'baboon.png'), cv2.IMREAD_COLOR)
xsize = 3
ysize = 3
cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('img2', cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('xsize', 'img2', xsize, 50, do_nothing)
cv2.createTrackbar('ysize', 'img2', ysize, 50, do_nothing)

while cv2.waitKey(1) != ord('q'):
    xsize = cv2.getTrackbarPos('xsize', 'img2')
    ysize = cv2.getTrackbarPos('ysize', 'img2')
    
    img2 = cv2.blur(img, (xsize + 1, ysize + 1))
    cv2.imshow('img', img)
    cv2.imshow('img2', img2)

cv2.destroyAllWindows()

#%% Spatial transformation - Sharpening
img = cv2.imread(os.path.join(PATH_TO_IMAGES,'baboon.png'), cv2.IMREAD_COLOR)
wsize = 3

cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('img2', cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('wsize', 'img2', wsize, 10, do_nothing)

while cv2.waitKey(1) != ord('q'):
    wsize = cv2.getTrackbarPos('wsize', 'img2')

    img2 = cv2.Laplacian(img, cv2.CV_16S, ksize=2*wsize + 1, 
                         scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    cv2.imshow('img', img)
    cv2.imshow('img2', cv2.normalize(img2.astype('float'), None, 1, 0, cv2.NORM_MINMAX))
cv2.destroyAllWindows()

#%%
size = 0
kx = cv2.getDerivKernels(1, 0, size)
ky = cv2.getDerivKernels(0, 1, size)
kx2 = np.outer(kx[0], kx[1])
ky2 = np.outer(ky[0], ky[1])
print('\nkx2 = ')
print(kx2)
print('\nky2 = ')
print(ky2)

# =============================================================================
# 
# =============================================================================

#%% Working directly in BGR colorspace
#Using a prepared image
rows = 100
cols = 200

R = np.zeros((rows, cols), np.float32)
B = np.copy(R);
for col in range(int(cols / 2), cols):
    for row in range(0, rows):
        R[row, col] = 1

G = np.copy(R)

for col in range(0, cols):
    for row in range(0, int(rows / 2)):
        B[row, col] = 1

bgr = [B, G, R]
img = cv2.merge(bgr)

#%% Color Gradient
side = int(400)
rows, cols = 400, 200

r = 255*np.concatenate((np.zeros((rows, cols)), np.ones((rows, cols))), axis=1).astype(np.uint8)
g = r.copy()
b = np.flipud(r.transpose())
rgb = cv2.merge([r, g, b])
img = rgb.copy()


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

Theta = 0.5 * np.arctan2((2 * Gxy) , (Gxx - Gyy))
F = np.sqrt(0.5 * (Gxx + Gyy) + (Gxx - Gyy) * np.cos(2 * Theta) + 2 * Gxy * np.sin(2 * Theta))
F = cv2.normalize(F.astype('float'), None, 0, 1, cv2.NORM_MINMAX)

rgb_grad = cv2.add(cv2.Sobel(r, cv2.CV_8U, 1, 0, ksize=5), 
            cv2.Sobel(g, cv2.CV_8U, 1, 0, ksize=5),
            cv2.Sobel(b, cv2.CV_8U, 1, 0, ksize=5))


plt.figure(1, figsize = (32, 16)), plt.clf()
plt.subplot(231), plt.imshow(r, cmap = 'gray'), plt.title('R')
plt.subplot(232), plt.imshow(g, cmap = 'gray'), plt.title('G')
plt.subplot(233), plt.imshow(b, cmap = 'gray'), plt.title('B')
plt.subplot(234), plt.imshow(rgb)
plt.subplot(235), plt.imshow(F, cmap = 'gray'), plt.title('F')
plt.subplot(236), plt.imshow(rgb_grad, cmap = 'gray'), plt.title('RGB Gradient')
plt.show()

#%% Using the 'baboon' image
from utils import color_gradient

img = cv2.imread(os.path.join(PATH_TO_IMAGES,'lena.png'), cv2.IMREAD_COLOR)
img = np.float32(img)

F = color_gradient(img)

cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('F', cv2.WINDOW_KEEPRATIO)
while 0xFF & cv2.waitKey(1) != ord('q'):

    img = cv2.normalize(img.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
    F   = cv2.normalize(F.astype('float'),   None, 0, 1, cv2.NORM_MINMAX)

    cv2.imshow('img', img)
    cv2.imshow('F', F)

cv2.destroyAllWindows()

#%% Image Segmentation in the BGR colorspace - Part I
img = cv2.imread(os.path.join(PATH_TO_IMAGES,'baboon.png'), cv2.IMREAD_COLOR)

sp = 10
sr = 100
maxLevel = 1

img2 = cv2.pyrMeanShiftFiltering(img, sp, sr, 
                                 maxLevel, 
                                 cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS)

while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow('img', img)
    cv2.imshow('img2', img2)
cv2.destroyAllWindows()

#%% Creating a gaussian image
img = np.zeros((100, 100), np.float32)
mx = 50
my = 50
sx = 20
sy = 10

colormap = cv2.COLORMAP_JET
theta_slider = 20

cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('mx', 'img', mx, img.shape[1], do_nothing)
cv2.createTrackbar('my', 'img', my, img.shape[1], do_nothing)
cv2.createTrackbar('sx', 'img', sx, img.shape[1], do_nothing)
cv2.createTrackbar('sy', 'img', sy, img.shape[1], do_nothing)
cv2.createTrackbar('theta', 'img', theta_slider, 100, do_nothing)
cv2.createTrackbar('colormap', 'img', colormap, 7, do_nothing)

while cv2.waitKey(1) != ord('q'):
    mx = cv2.getTrackbarPos('mx', 'img')
    my = cv2.getTrackbarPos('my', 'img')
    sx = cv2.getTrackbarPos('sx', 'img')
    sy = cv2.getTrackbarPos('sy', 'img')
    theta_slider = cv2.getTrackbarPos('theta', 'img')
    colormap = cv2.getTrackbarPos('colormap', 'img')

    theta = theta_slider * 2 * np.pi / 100
    img = create2DGaussian(100, 100, mx, my, sx, sy, theta)
    cv2.imshow('img', cv2.applyColorMap(scaleImage2_uchar(img), colormap))

cv2.destroyAllWindows()

#%%
img = cv2.imread(os.path.join(PATH_TO_IMAGES,'baboon.png'), cv2.IMREAD_COLOR)
img = np.float32(img)
B, G, R = cv2.split(img)
#planes = cv2.split(img)
#B = planes[0]
#G = planes[1]
#R = planes[2]

dBx = cv2.Sobel(B, cv2.CV_32F, 1, 0)
dBy = cv2.Sobel(B, cv2.CV_32F, 0, 1)
dGx = cv2.Sobel(G, cv2.CV_32F, 1, 0)
dGy = cv2.Sobel(G, cv2.CV_32F, 0, 1)
dRx = cv2.Sobel(R, cv2.CV_32F, 1, 0)
dRy = cv2.Sobel(R, cv2.CV_32F, 0, 1)

plt.rcParams['figure.figsize'] = [10, 10]
plt.subplot('221'); plt.title('Original'); plt.imshow(bgr2rgb(scaleImage2_uchar(img)), 'gray')
plt.subplot('222'); plt.title('B'); plt.imshow(scaleImage2_uchar(B), 'gray')
plt.subplot('223'); plt.title('dBx'); plt.imshow(scaleImage2_uchar(dBx), 'gray')
plt.subplot('224'); plt.title('dBy'); plt.imshow(scaleImage2_uchar(dBy), 'gray')
plt.show()

#%%
# loading image
img0 = cv2.imread(os.path.join(PATH_TO_IMAGES, 'SanFrancisco.jpg'),)
#img0 = cv2.imread(os.path.join(PATH_TO_IMAGES, 'windows.jpg'),)

# converting to gray scale
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# remove noise
img = cv2.GaussianBlur(gray,(3,3),0)

# convolute with proper kernels
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()
