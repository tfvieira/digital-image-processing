#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:56:09 2019

@author: tvieira
"""
import cv2
import numpy as np
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
    
def plot_histogram(img):
    # Grayscale image
    if len(img.shape) == 2:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist, color='k')
        plt.show()
    # Color image
    elif len(img.shape) == 3:
        color = ('r', 'g', 'b')
        for i, col in enumerate(color):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
    plt.show()
    
def create_noisy_img(shape, noise_type = 'uniform', a=127, b=40):
    tmp = np.zeros(shape, dtype=np.uint8)
    if noise_type == 'uniform':
        noise = cv2.merge([cv2.randu(x, a, b) for x in cv2.split(tmp)])
    elif noise_type == 'normal':
        noise = cv2.merge([cv2.randn(x, a, b) for x in cv2.split(tmp)])
    return noise