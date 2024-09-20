# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:37:23 2024

@author: lupus
"""

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

image = Image.open("C:/Users/lupus/Downloads/lys_2.jpg")

# r,g,b = image.split()

s = image.convert('L')
f = np.asarray(image.convert('L'))
n, m = f.shape


# plt.hist(g,255,density=True)
# plt.hist(np.linspace(0,255,n*m),density=True)
# plt.hist(np.linspace(0,255,n*m)[np.argsort(f.flatten())],density=True)

#ker = 255 * np.exp(-np.linspace(0,2,n*m))
ker = np.linspace(0,255,n*m)

h = ker[np.argsort(ker)[np.argsort(np.argsort(f.flatten()))]].reshape(n,m)
#h = np.linspace(0,255,n*m)[np.argsort(np.linspace(0,255,n*m))[np.argsort(np.argsort(f.flatten()))]].reshape(n,m)

im = Image.fromarray(h.astype(np.uint8))


image_s = get_concat_h(s, im)
image_s.show()

#%% La couleur est mal faite comme ca faut utiliser même méthode autrement


r,g,b = image.split()

r_s = r.convert('L') ; g_s = g.convert('L') ; b_s = b.convert('L')

f = np.asarray(r_s.convert('L'))
h = ker[np.argsort(ker)[np.argsort(np.argsort(f.flatten()))]].reshape(n,m)
imkr = Image.fromarray(h.astype(np.uint8))

f = np.asarray(g_s.convert('L'))
h = ker[np.argsort(ker)[np.argsort(np.argsort(f.flatten()))]].reshape(n,m)
imkg = Image.fromarray(h.astype(np.uint8))

f = np.asarray(b_s.convert('L'))
h = ker[np.argsort(ker)[np.argsort(np.argsort(f.flatten()))]].reshape(n,m)
imkb = Image.fromarray(h.astype(np.uint8))


im_recomp = Image.merge("RGB", (imkr, imkg, imkb))

image_s = get_concat_h(image, im_recomp)
image_s.show()








