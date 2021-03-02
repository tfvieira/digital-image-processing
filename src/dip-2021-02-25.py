#%%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
path = "../img/"

# %%
imfile = "ctskull.tif"
img = cv2.imread(os.path.join(path,imfile), cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.show()

#%%
imfile = "baboon.png"
img = cv2.imread(os.path.join(path,imfile), cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

#%%
imfile = "baboon.png"
img = cv2.imread(os.path.join(path,imfile), cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rgb = cv2.split(img)
r, g, b = cv2.split(img)

# plt.subplot("221"); plt.title("original"); plt.imshow(img)
# plt.subplot("222"); plt.title("r"); plt.imshow(r, cmap='gray')
# plt.subplot("223"); plt.title("g"); plt.imshow(g, cmap='gray')
# plt.subplot("224"); plt.title("b"); plt.imshow(b, cmap='gray')
# plt.show()

plt.subplot("221"); plt.title("original"); plt.imshow(img)
plt.subplot("222"); plt.title("r"); plt.imshow(rgb[0], cmap='gray')
plt.subplot("223"); plt.title("g"); plt.imshow(rgb[1], cmap='gray')
plt.subplot("224"); plt.title("b"); plt.imshow(rgb[2], cmap='gray')
plt.show()

#%%
imfile = "ctskull.tif"
img = cv2.imread(os.path.join(path,imfile), cv2.IMREAD_GRAYSCALE)
plt.subplot("221"); plt.title("Original"); plt.imshow(img, cmap='gray')
plt.subplot("222"); plt.title("Histogram"); plt.hist(img.ravel(), 256, [0, 256])
plt.show()

#%%
imfile = "baboon.png"
img = cv2.imread(os.path.join(path,imfile), cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

color = ('r', 'g', 'b')

for i, col in enumerate(color):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.show()