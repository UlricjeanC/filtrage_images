# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:49:16 2024

@author: UlricJeanC
"""

from PIL import Image
import numpy as np


def normMatImage(Mat):
    Mat = np.where(Mat > 0, Mat, 0)
    m = np.max(Mat)
    Mat = (Mat / m) * 255
    Mat = np.int_(np.ceil(Mat))
    return Mat

def get_concat_h(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def DFT_helper(f):
    M, N = f.shape
    E = np.zeros((M,N))
    for m in range(M-1):
        for n in range(N-1):
            E[m,n] = p*m/M + q*n/N
    return E

MEMO_DFT = {}

def DFT(f):
    M, N = f.shape
    F = np.zeros((M,N))
    E = np.zeros((M,N))
    for p in range(M-1):
        for q in range(N-1):
            res = MEMO_DFT.get((p,q))
            for n in range(M-1):
                for m in range(N-1):
                    E[n,p] = p*n/N + q*m/M
            F[p,q] = np.sum ( f*np.exp(-1j*2*np.pi*E) ) 
    return F

        
image = Image.open("C:/Users/lupus/Desktop/images_python/pillow.png")

image = image.convert('L')

Mat = np.asarray(image)

F = DFT(Mat)

dim = Mat.shape

image.show()