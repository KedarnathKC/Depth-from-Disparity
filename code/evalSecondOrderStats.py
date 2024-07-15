import numpy as np
import skimage.io as sio
import os
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import convolve

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#

def derivative_x(im):
    sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    return convolve(im, sobel_x, mode='constant', cval=0.0)

def derivative_y(im):    
    sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
    return convolve(im, sobel_y, mode='constant', cval=0.0)


def second_order_statistics(ix, iy, ws):
    ixx = ix*ix
    ixy = ix*iy
    iyy = iy*iy
    k=0.03
    f = np.ones((ws,ws))
    ixxsum = convolve(ixx, f, mode='constant', cval=0.0)
    ixysum = convolve(ixy, f, mode='constant', cval=0.0)
    iyysum = convolve(iyy, f, mode='constant', cval=0.0)
    R=np.zeros(ix.shape)
    for i in range(ix.shape[0]):
        for j in range(ix.shape[1]):
            R[i,j] = ixxsum[i,j]*iyysum[i,j] - ixysum[i,j]*ixysum[i,j] -k*(ixxsum[i,j]+iyysum[i,j])**2
    return R

# read image
im_dir = "./data/disparity"
image_file = "cones_im2.png"
image = sio.imread(os.path.join(im_dir, image_file))
# convert image to gray
image = rgb2gray(image)

# compute differentiations along x and y axis respectively
# x-diff
#--------- add your code here ------------------#
Ix = derivative_x(image)

# y-diff
#--------- add your code here ------------------#
Iy = derivative_y(image)

# set window size
#--------- modify this accordingly ------------------#
ws = 5

heatMapImg = second_order_statistics(Ix, Iy, ws)

plt.imshow(heatMapImg,cmap='hot')
plt.colorbar()
savedir = "./output/harris/"
savefile = "cones_im2.png"
if not os.path.isdir(savedir):
    os.makedirs(savedir)
plt.imsave(os.path.join(savedir, 'ws-'+str(ws)+'_'+savefile), heatMapImg)
plt.show()