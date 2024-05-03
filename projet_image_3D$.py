# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:46:15 2024

@author: UlricJeanC
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image = Image.open("C:/Users/lupus/Desktop/images_python/pillow.png")
image = image.convert('L')
Mat = np.asarray(image)

def match_im(f1,f2,alpha,k=5):
    dim = f1.shape
    troncx = dim[0]//k
    addx = dim[0]%k
    troncy = dim[1]//k
    addy = dim[1]%k
    
    print(addx)
    print(addy)
    
    return f1.shape

f1 = f2 = Mat ; alpha = 3
match_im(f1,f2,alpha,k=5)