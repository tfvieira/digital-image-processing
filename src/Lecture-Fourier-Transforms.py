#%% Import libraries
import os
import cv2
import numpy as np
from numpy.fft import fft2, fftshift
from utils import  do_nothing
from utils import normalize_img
from utils import createCosineImage2
from utils import createWhiteDisk2
from utils import applyLogTransform
from utils import scaleImage2_uchar
path = '../img'

#%% Define functions
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

def scaleImage2_uchar(src):
    tmp = np.copy(src)
    if src.dtype != np.float32:
        tmp = np.float32(tmp)
    cv2.normalize(tmp, tmp, 1, 0, cv2.NORM_MINMAX)
    tmp = 255 * tmp
    tmp = np.uint8(tmp)
    return tmp

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
    
    xx0, yy0 = np.meshgrid(range(rows), range(cols))
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

#%% Demonstrate the creation of a Gaussian filter
rows = 100
cols = 100

xc = 50
yc = 50
sx = 30
sy = 10
theta = 0

cv2.namedWindow('img', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('sliders', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('xc', 'sliders', xc, int(rows), do_nothing)
cv2.createTrackbar('yc', 'sliders', yc, int(cols), do_nothing)
cv2.createTrackbar('sx', 'sliders', sx, int(rows), do_nothing)
cv2.createTrackbar('sy', 'sliders', sy, int(cols), do_nothing)
cv2.createTrackbar('theta', 'sliders', theta, 360, do_nothing)

while 0xFF & cv2.waitKey(1) != ord('q'):
    xc = cv2.getTrackbarPos('xc', 'sliders')
    yc = cv2.getTrackbarPos('yc', 'sliders')
    sx = cv2.getTrackbarPos('sx', 'sliders')
    sy = cv2.getTrackbarPos('sy', 'sliders')
    theta = cv2.getTrackbarPos('theta', 'sliders')
    img = create2DGaussian(rows, cols, xc, yc, sx, sy, theta)
    cv2.imshow('img', cv2.applyColorMap(scaleImage2_uchar(img), 
                                        cv2.COLORMAP_JET))
cv2.destroyAllWindows()

#%% The Discrete Fourier Transform - Part I - Obtaining real and imaginary 
#   parts of the DFT
img = cv2.imread(os.path.join(path, 'rectangle.jpg'), cv2.IMREAD_GRAYSCALE)


cv2.namedWindow('Original', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Plane 0 - Real', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Plane 1 - Imaginary', cv2.WINDOW_KEEPRATIO)

planes = [np.zeros(img.shape, dtype=np.float64), 
          np.zeros(img.shape, dtype=np.float64)]
planes[0][:] = np.float64(img[:])

img2 = cv2.merge(planes)
img2 = cv2.dft(img2)

planes = cv2.split(img2)

# cv2.normalize(planes[0], planes[0], 1,0, cv2.NORM_MINMAX)
# cv2.normalize(planes[1], planes[1], 1,0, cv2.NORM_MINMAX)

planes[0] = normalize_img(planes[0])
planes[1] = normalize_img(planes[1])

while 0xFF & cv2.waitKey(1) != ord('q'):
    cv2.imshow('Original', img)
    cv2.imshow('Plane 0 - Real', planes[0])
    cv2.imshow('Plane 1 - Imaginary', planes[1])
cv2.destroyAllWindows()


#%% DFT - Part II -> Applying the log transform
img = cv2.imread(os.path.join(path,'rectangle.jpg'), cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('Original', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Plane 0 - Real', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Plane 1 - Imaginary', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Mag', cv2.WINDOW_KEEPRATIO)

planes = [np.zeros(img.shape, dtype=np.float64), 
          np.zeros(img.shape, dtype=np.float64)]

planes[0][:] = np.float64(img[:])
planes[1][:] = np.float64(img[:])

cv2.normalize(planes[0], planes[0], 1, 0, cv2.NORM_MINMAX)
cv2.normalize(planes[1], planes[1], 1, 0, cv2.NORM_MINMAX)

img2 = cv2.merge(planes)
img2 = cv2.dft(img2)
planes = cv2.split(img2)

mag = cv2.magnitude(planes[0], planes[1])
mag += 1
mag = np.log(mag)

cv2.normalize(mag,mag, 1,0, cv2.NORM_MINMAX)

while cv2.waitKey(1) != ord('q'):
    cv2.imshow('Original', img)
    cv2.imshow('Plane 0 - Real', planes[0])
    cv2.imshow('Plane 1 - Imaginary', planes[1])
    cv2.imshow('Mag', mag)
cv2.destroyAllWindows()

#%% DFT - Part III -> Shifting the Transform
# img = cv2.imread(os.path.join(path,'lena.png'), cv2.IMREAD_GRAYSCALE)
img = cv2.imread(os.path.join(path,'rectangle.jpg'), cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('Original', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Mag', cv2.WINDOW_KEEPRATIO)

planes = [np.zeros(img.shape, dtype=np.float64), 
          np.zeros(img.shape, dtype=np.float64)]

planes[0][:] = np.float64(img[:])
planes[1][:] = np.float64(img[:])
cv2.normalize(planes[0], planes[0], 1, 0, cv2.NORM_MINMAX)
cv2.normalize(planes[1], planes[1], 1, 0, cv2.NORM_MINMAX)

img2 = cv2.merge(planes)
img2 = cv2.dft(img2)
planes = cv2.split(img2)

mag = cv2.magnitude(planes[0], planes[1])
mag += 1
mag = np.log(mag)

cv2.normalize(mag, mag, 1, 0, cv2.NORM_MINMAX)

while cv2.waitKey(1) != ord('q'):
    #print(mag)
    cv2.imshow('Original', img)
    cv2.imshow('Mag', np.fft.fftshift(mag))
cv2.destroyAllWindows()

#%% The Discrete Fourier Transform
rows = 200
cols = 200
disk = np.zeros((rows, cols), np.float32)

cv2.namedWindow('disk', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('sliders', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

xc = 100
yc = 100
radius = 20

cv2.createTrackbar('xc', 'sliders', xc, disk.shape[0], do_nothing)
cv2.createTrackbar('yc', 'sliders', yc, disk.shape[1], do_nothing)
cv2.createTrackbar('radius', 'sliders', radius, int(disk.shape[1]/2), do_nothing)

while cv2.waitKey(1) != ord('q'):
    xc = cv2.getTrackbarPos('xc', 'sliders')
    yc = cv2.getTrackbarPos('yc', 'sliders')
    radius = cv2.getTrackbarPos('radius', 'sliders')

    e1 = cv2.getTickCount()
    # disk = createWhiteDisk(rows, cols, xc, yc, radius) # fps ~100
    disk = createWhiteDisk2(200, 200, xc, yc, radius) # fps ~1500
    e2 = cv2.getTickCount()
    fps = cv2.getTickFrequency()/(e2 - e1)
    print(f'time = {1/fps:.4e}, fps = {fps:.4e}')
    
    cv2.imshow('disk', disk)
cv2.destroyAllWindows()

#%% The Discrete Fourier Transform - Part III - Lowpass Filtering
# Ressaltar o surgimento de 'falseamento', isto é, frequencia notáveis
# quando é feita a transformada inversa. Isto ocorre pq o filtro é IDEAL.
# Comparar o resultado da filtragem usando uma Gaussiana como filtro.
img = cv2.imread(os.path.join(path, 'lena.png'), cv2.IMREAD_GRAYSCALE)

fft_complex = fft2(img)
ifft_complex = np.fft.ifft2(fft_complex)
ifft_real = np.real(ifft_complex)

radius = 50

cv2.namedWindow('mask', cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar('radius', 'mask', radius, 4 * np.max(img.shape), do_nothing)
cv2.namedWindow('ifft_real', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('ifft_real_filtered', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

while 0xFF & cv2.waitKey(1) != ord('q'):

    radius = cv2.getTrackbarPos('radius', 'mask')
    mask = create2DGaussian(img.shape[0],
                            img.shape[1],
                            int(img.shape[0] / 2),
                            int(img.shape[1] / 2),
                            radius,
                            radius,
                            theta = 0)
    # mask[mask >  .5] = 1.0
    # mask[mask <= .5] = 0
    
    fft_complex_filtered = fft_complex * fftshift(mask)
    
    ifft_complex_filtered = np.fft.ifft2(fft_complex_filtered)
    ifft_real_filtered = np.real(ifft_complex_filtered)
    
    # Scale images for visualization
    fft_abs = np.absolute(fft_complex_filtered)
    fft_abs = np.log(1 + fft_abs)
    fft_abs = normalize_img(fft_abs)
    fft_abs = fftshift(fft_abs)

    cv2.imshow("img", img)
    cv2.imshow("fft_abs", fft_abs)
    cv2.imshow("ifft_real", ifft_real.astype('uint8'))
    cv2.imshow("ifft_real_filtered", ifft_real_filtered.astype('uint8'))
    cv2.imshow('mask', mask)

cv2.destroyAllWindows()


#%% The Discrete Fourier Transform - Part IV - Highpass Filtering
img = cv2.imread(os.path.join(path,'lena.png'), cv2.IMREAD_GRAYSCALE)

fft_complex = fft2(img)
ifft_complex = np.fft.ifft2(fft_complex)
ifft_real = np.real(ifft_complex)

radius = 50

cv2.namedWindow('mask', cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar('radius', 'mask', radius, np.max(img.shape), do_nothing)
cv2.namedWindow('ifft_real', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('ifft_real_filtered', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

while 0xFF & cv2.waitKey(1) != ord('q'):

    radius = cv2.getTrackbarPos('radius', 'mask')
    mask = create2DGaussian(img.shape[0],
                            img.shape[1],
                            int(img.shape[0] / 2),
                            int(img.shape[1] / 2),
                            radius,
                            radius,
                            theta = 0)
    mask[mask >  .5] = 1.0
    mask[mask <= .5] = 0.0
    mask = 1.0 - mask
    if radius == 0:
        mask = np.ones(mask.shape, mask.dtype)
        
    fft_complex_filtered = fft_complex * fftshift(mask)
    
    ifft_complex_filtered = np.fft.ifft2(fft_complex_filtered)
    ifft_real_filtered = np.real(ifft_complex_filtered)
    ifft_real_filtered = 255 * normalize_img(ifft_real_filtered)
    
    # Scale images for visualization
    fft_abs = np.absolute(fft_complex_filtered)
    fft_abs = np.log(1 + fft_abs)
    fft_abs = normalize_img(fft_abs)
    fft_abs = fftshift(fft_abs)

    cv2.imshow("img", img)
    cv2.imshow("fft_abs", fft_abs)
    cv2.imshow("ifft_real", ifft_real.astype('uint8'))
    cv2.imshow("ifft_real_filtered", ifft_real_filtered.astype('uint8'))
    cv2.imshow('mask', mask)

cv2.destroyAllWindows()

#%% The Discrete Fourier Transform - Visualizing sinusoidal images - Part I
rows = 250
cols = 250
freq = 1
theta = 0

cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('Freq', 'img', freq, 500, do_nothing)
cv2.createTrackbar('Theta', 'img', theta, 360, do_nothing)

while cv2.waitKey(1) != ord('q'):
    freq = cv2.getTrackbarPos('Freq', 'img')
    theta = cv2.getTrackbarPos('Theta', 'img')

    e1 = cv2.getTickCount()
    # img = createCosineImage(rows, 
    #                        cols, 
    #                        float(freq/1e3), 
    #                        float(2 * np.pi * theta/100.0)) # ~0.2 second
    img = createCosineImage2(rows,
                             cols,
                             float(freq/1e3),
                             theta) # ~0.001 second
    e2 = cv2.getTickCount()
    t = (e2 - e1)/cv2.getTickFrequency()
    print(f'Elapsed time = {t:.3e} seconds.')
    cv2.imshow('img', scaleImage2_uchar(img))
cv2.destroyAllWindows()

#%% The Discrete Fourier Transform - Visualizing sinusoidal images - Part II
rows = 250
cols = 250
freq = 1
theta = 2

cv2.namedWindow('mag', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('Freq', 'img', freq, 500, do_nothing)
cv2.createTrackbar('Theta', 'img', theta, 360, do_nothing)

while cv2.waitKey(1) != ord('q'):
    freq = cv2.getTrackbarPos('Freq', 'img')
    theta = cv2.getTrackbarPos('Theta', 'img')

    img = createCosineImage2(rows, cols, float(freq / 1e3), theta)
    img3 = np.copy(img)
    planes = [img3, np.zeros(img3.shape, np.float64)]
    img2 = cv2.merge(planes)
    img2 = cv2.dft(img2)
    planes = cv2.split(img2)
    mag = cv2.magnitude(planes[0], planes[1])
    mag = applyLogTransform(mag)

    cv2.imshow('img', cv2.applyColorMap((255 * normalize_img(img)).astype(np.uint8), 
                                        cv2.COLORMAP_COOL))
    cv2.imshow('mag', cv2.applyColorMap(np.fft.fftshift((255 * normalize_img(mag)).astype(np.uint8)), 
                                        cv2.COLORMAP_COOL))
cv2.destroyAllWindows()

#%% The Discrete Fourier Transform - Adding sinusoidal noise to images - Part I
cv2.namedWindow('mag', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)

img = cv2.imread(os.path.join(path,'lena.png'), cv2.IMREAD_GRAYSCALE)

img = np.float32(img)
img = img / 255.0

rows = img.shape[0]
cols = img.shape[1]

freq = 90
theta = 10
gain = 30

cv2.createTrackbar('Freq', 'img', freq, 500, do_nothing)
cv2.createTrackbar('Theta', 'img', theta, 100, do_nothing)
cv2.createTrackbar('Gain', 'img', gain, 100, do_nothing)

while cv2.waitKey(1) != ord('q'):
    freq = cv2.getTrackbarPos('Freq', 'img')
    theta = cv2.getTrackbarPos('Theta', 'img')
    gain = cv2.getTrackbarPos('Gain', 'img')

    noise = createCosineImage2(rows, cols, float(freq/1e3), theta)
    noise = img + float(gain/100.0) * noise

    img3 = np.copy(noise)
    planes = [img3, np.zeros(img3.shape, np.float64)]
    img2 = cv2.merge(planes)
    img2 = cv2.dft(img2)
    planes = cv2.split(img2)
    mag = cv2.magnitude(planes[0], planes[1])
    mag = applyLogTransform(mag)

    cv2.imshow('img', scaleImage2_uchar(noise))
    cv2.imshow('mag', cv2.applyColorMap(
            np.fft.fftshift(scaleImage2_uchar(mag)), 
            cv2.COLORMAP_OCEAN))
cv2.destroyAllWindows()

#%% The Discrete Fourier Transform - Adding sinusoidal noise to images - Part II
cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('mask', cv2.WINDOW_KEEPRATIO)

img = cv2.imread(os.path.join(path, 'lena.png'), cv2.IMREAD_GRAYSCALE)
img = np.float32(img)
img = img / 255.0;

rows = img.shape[0]
cols = img.shape[1]

freq = 90
theta = 10
gain = 30

cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('mag', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('mask', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('tmp', cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('Freq', 'img', freq, 500, do_nothing)
cv2.createTrackbar('Theta', 'img', theta, 360, do_nothing)
cv2.createTrackbar('Gain', 'img', gain, 100, do_nothing)

bandwidth = 2
outer_radius = 256 - 210 + bandwidth
inner_radius = 256 - 210 - bandwidth
cv2.createTrackbar('in_radius', 'mask', inner_radius, img.shape[1], do_nothing)
cv2.createTrackbar('out_radius', 'mask', outer_radius, img.shape[1], do_nothing)

while cv2.waitKey(1) != ord('q'):
    freq = cv2.getTrackbarPos('Freq', 'img')
    theta = cv2.getTrackbarPos('Theta', 'img')
    gain = cv2.getTrackbarPos('Gain', 'img')

    outer_radius = cv2.getTrackbarPos('in_radius', 'mask')
    inner_radius = cv2.getTrackbarPos('out_radius', 'mask')

    noise = img + float(gain / 100.0) * createCosineImage2(
            rows, cols, float(freq / 1e3), theta)

    mask = 1 - (createWhiteDisk2(rows, cols, int(cols / 2), 
                                 int(rows / 2), outer_radius) - createWhiteDisk2(rows, cols,  int(cols/2), int(rows/2), inner_radius))

    planes = [np.copy(noise), np.zeros(noise.shape, np.float64)]
    img2 = cv2.merge(planes)
    img2 = cv2.dft(img2)
    planes = cv2.split(img2)
    mag = cv2.magnitude(planes[0], planes[1])
    mag = applyLogTransform(mag)
    planes[0] = np.multiply(np.fft.fftshift(mask), planes[0])
    planes[1] = np.multiply(np.fft.fftshift(mask), planes[1])
    tmp = cv2.merge(planes)
    tmp = cv2.idft(tmp)

    cv2.imshow('img', scaleImage2_uchar(noise))
    cv2.imshow('mag', cv2.applyColorMap(np.fft.fftshift(scaleImage2_uchar(mag)), cv2.COLORMAP_OCEAN))
    cv2.imshow('mask', scaleImage2_uchar(mask))
    cv2.imshow('tmp', scaleImage2_uchar(tmp[:, :, 0]))
cv2.destroyAllWindows()
