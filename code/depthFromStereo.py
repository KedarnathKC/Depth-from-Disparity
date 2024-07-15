# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#

import math
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter

def SSD(img1,img2):
    ssd=np.sum(np.square(img1*img2),axis=1)
    return ssd 

def CosineSimilarity(img1,img2):
    norm = np.linalg.norm(img1,axis=1)*np.linalg.norm(img2,axis=1)
    norm=norm[:,np.newaxis]
    return 1-np.sum((img1*img2)/norm,axis=1)

def getPatchFromImg(Img,winSize):
    patchesImg=dict()
    for i in range(Img.shape[0]-winSize+1):
        patches = list()
        for j in range(Img.shape[1]-winSize+1):
            patches.append(np.ndarray.flatten(Img[i:i+winSize,j:j+winSize]))
        patchesImg[i]=np.array(patches)
    return patchesImg

def depthFromStereo(img1, img2, ws):
    img1=gaussian_filter(color.rgb2gray(img1),2)
    img2=gaussian_filter(color.rgb2gray(img2),2)
    h,w= img1.shape
    img1=np.pad(img1,ws//2)
    img2=np.pad(img2,ws//2)
    patches=getPatchFromImg(img2,ws)
    depth = findDepth(img1,patches,ws,h,w)
    depth[depth>0.08]=0
    # depthGaussian = gaussian_filter(depth,2)
    return depth

def findDepth(img,patches,ws,h,w):
    depth=np.zeros((h,w))
    for y in range(h):
        for x in range(w):
            patchX = np.ndarray.flatten(img[y:y+ws, x:x+ws])[np.newaxis,:]
            patchX = np.repeat(patchX,repeats=w,axis=0)
            ssd=CosineSimilarity(patchX,patches[y])
            ind=np.where(ssd==np.min(ssd))[0][0]
            depth[y,x]=1/(abs(ind-x)+0.01)
    return depth
