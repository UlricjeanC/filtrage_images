# -*- coding: utf-8 -*-
"""
Created on Fri May 10 08:32:41 2024

@author: UlricJeanC
"""

#%% IMPORTS/FONCTIONS

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def PSFronebius(A,B):
    return np.sum(A*B)

def NormFronebius(A):
    return np.sqrt(PSFronebius(A,A))

def creationMatSigma(U,V,sigma):
    
    dimU = U.shape[0]
    dimV = V.shape[0]
    
    if dimU < dimV:
        S = np.diag(sigma,k = (dimU-dimV))
        S = S[dimV-dimU:S.shape[0],:]
    elif dimU > dimV:
        S = np.diag(sigma,k = (dimU-dimV))
        S = S[:,(dimU-dimV):S.shape[0]]
    return S

def SVD(f,k):
    
    U, sigma, V = np.linalg.svd(f)
    S = creationMatSigma(U,V,sigma)
    
    u = U[:,0:k]
    v = V[0:k,:]
    s = S[0:k,0:k]
    return u@s@v

#%% TEST IMAGE 

image = Image.open("C:/Users/lupus/Desktop/images_python/passiflora.jpg")
image = image.convert('L')


image = Image.open("C:/Users/lupus/Desktop/images_python/passiflora.jpg")
r, g, image = image.split()

"""
image = Image.open("C:/Users/lupus/Desktop/images_python/pont.jpg")
image = image.convert('L')
"""

veck = np.array((1,5,10,20,40,80,160,220,440,880))
f = np.asarray(image) ; image_s = image ; image_err = image
U, sigma, V = np.linalg.svd(f)
vecDiff = np.empty(shape=0)

for k in veck:
    imk = SVD(f,k)
    imk = Image.fromarray(imk.astype(np.uint8))
    image_s = get_concat_h(image_s, imk)
    
    fk = np.asarray(imk)
    diff = NormFronebius(f-fk)
    vecDiff = np.append(vecDiff, diff)
    
    err = f - fk
    err = Image.fromarray(err.astype(np.uint8))

    image_err = get_concat_h(image_err, err)

image_s = get_concat_v(image_s,image_err)
image_s.show()

#%% PERFORMANCES

plt.figure()
plt.plot(veck,vecDiff)

#%% IMAGE RGB

image = Image.open("C:/Users/lupus/Desktop/images_python/passiflora.jpg")
r, g, b = image.split()

veck = np.array((1,5,10,20,40,80,160,220,440,880))
fr = np.asarray(r) ; fg = np.asarray(g) ; fb = np.asarray(b) ; image_s = image

for k in veck:
    
    imkr = SVD(fr,k)
    imkg = SVD(fg,k)
    imkb = SVD(fb,k)
    imkr = Image.fromarray(imkr.astype(np.uint8))
    imkg = Image.fromarray(imkg.astype(np.uint8))
    imkb = Image.fromarray(imkb.astype(np.uint8))
    
    imrgb = Image.merge("RGB", (imkr, imkg, imkb))
    image_s = get_concat_h(image_s, imrgb)

image_s.show()









