#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:56:09 2019

@author: tvieira
"""

import cv2
import matplotlib.pyplot as plt

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