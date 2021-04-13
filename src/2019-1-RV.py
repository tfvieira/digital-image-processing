#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 20:16:16 2019

@author: tvieira
"""

#%%
runfile('dip.py')

#%%
np.random.seed(0)
img = np.random.random((3,4))
idx = img > .5
img[ idx] = 0
img[~idx] = 1
img

#%%
img = cv2.imread('moon.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (8,8))
a = sum([2 ** i for (i, v) in enumerate(img.flatten()) if v > 60])
print('\na = ')
print(a)

img2 = cv2.imread('moon_result.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.resize(img2, (8,8))
b = sum([2 ** i for (i, v) in enumerate(img2.flatten()) if v > 60])
print('\nb = ')
print(b)

img3 = cv2.imread('ckbd2.png', cv2.IMREAD_GRAYSCALE)
img3 = cv2.resize(img3, (8,8))
c = sum([2 ** i for (i, v) in enumerate(img3.flatten()) if v > 60])
print('\nb = ')
print(c)

# 115792089237158226986068149228641030811744479017162221396356760341031663173631
# 114885696693032228116255894290642557386037123911622252097122726389930027603903
#a = '115792089237158226986068149228641030811744479017162221396356760341031663173631'
#b = '114885696693032228116255894290642557386037123911622252097122726389930027603903'
#np.count_nonzero(a!=b)
