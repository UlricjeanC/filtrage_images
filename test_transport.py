# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:37:23 2024

@author: lupus
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation

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
# image = Image.open("C:/Users/lupus/Downloads/lys_4.jpg")

r,g,b = image.split()

s = image.convert('L')
f = np.asarray(image.convert('L'))
n, m = f.shape


# plt.hist(g,255,density=True)
# plt.hist(np.linspace(0,255,n*m),density=True)
# plt.hist(np.linspace(0,255,n*m)[np.argsort(f.flatten())],density=True)

#ker = 255 * np.exp(-np.linspace(0,2,n*m))
ker = np.linspace(0,255,n*m)
#ker = np.array((0*np.ones(int(n*m/2)),255*np.ones(int(n*m/2)))).flatten()

h = ker[np.argsort(ker)[np.argsort(np.argsort(f.flatten()))]].reshape(n,m)
#h = np.linspace(0,255,n*m)[np.argsort(np.linspace(0,255,n*m))[np.argsort(np.argsort(f.flatten()))]].reshape(n,m)

im = Image.fromarray(h.astype(np.uint8))


image_s = get_concat_h(s, im)
image_s.show()

#%% avec transition

nb = 25
temp = np.linspace(0,1,nb)
image_s = image.convert('L')
for t in temp:
    if t != 0:
        ker = np.linspace(0,255,n*m)*t + (1-t)*f.flatten()
        h = ker[np.argsort(ker)[np.argsort(np.argsort(f.flatten()))]].reshape(n,m)
        im = Image.fromarray(h.astype(np.uint8))
        image_s = get_concat_h(image_s, im)
image_s.show()

#%% Refaire en 1 seule frame

nb = 25
temp = np.linspace(0,1,nb)
snapshots = []
for t in temp:
    ker = np.linspace(0,255,n*m)*t + (1-t)*f.flatten()
    #ker = np.array((0*np.ones(int(n*m/2)),255*np.ones(int(n*m/2)))).flatten()*t + (1-t)*f.flatten()
    h = ker[np.argsort(ker)[np.argsort(np.argsort(f.flatten()))]].reshape(n,m)
    snapshots.append(255-h)


fig = plt.figure( figsize=(8,8) )

fps = 5
nSeconds = 5
a = snapshots[0]
im = plt.imshow(a, cmap='Greys')
im.axes.get_xaxis().set_visible(False)
im.axes.get_yaxis().set_visible(False)

def animate_func(i):
    if i % fps == 0:
        print( '.', end ='' )

    im.set_array(snapshots[i])
    return [im]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps,
                               )
    

#%% La couleur est mal faite comme ca faut utiliser même méthode autrement

"""
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

"""

#%% représentation

def creat_color(r_vec,g_vec,b_vec):
    l = []
    for i in range(len(r_vec)):
        l.append('#%02x%02x%02x' % (r_vec[i],g_vec[i],b_vec[i]))
    return l

r_vec = np.asarray(r.convert('L')).flatten()
g_vec = np.asarray(g.convert('L')).flatten()
b_vec = np.asarray(b.convert('L')).flatten()

list_col = creat_color(r_vec,g_vec,b_vec)

# Select Sub Group
sub = np.random.randint(0,n*m,10000)


# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

# Creating plot
ax.scatter3D(r_vec[sub], g_vec[sub], b_vec[sub], 
             color = np.asarray(list_col)[sub], marker='.')
plt.title("Plot des Couleurs")
 
# show plot
plt.show()

#%%
























