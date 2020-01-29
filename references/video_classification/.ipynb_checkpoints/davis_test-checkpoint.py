from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data
import random

# from utils.imutils2 import *
# from utils.transforms import *
import torchvision.transforms as transforms

import scipy.io as sio
import scipy.misc
import cv2

# get the video frames
# two patches in the future frame, one is center, the other is one of the 8 patches around

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    return img

def resize(img, owidth, oheight):
    img = im_to_numpy(img)
    img = cv2.resize( img, (owidth, oheight) )
    img = im_to_torch(img)
    return img

def load_image(img_path):
    # H x W x C => C x H x W
    img = cv2.imread(img_path)
#     print(img_path)
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:,:,::-1]
    img = img.copy()
    return im_to_torch(img)

def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x

class DavisSet(data.Dataset):
    def __init__(self, params, is_train=True):

        self.filelist = params['filelist']
        self.imgSize = params['imgSize']
        self.videoLen = params['videoLen']


        f = open(self.filelist, 'r')
        self.jpgfiles = []
        self.lblfiles = []

        for line in f:
            rows = line.split()
            jpgfile = rows[0]
            lblfile = rows[1]

            self.jpgfiles.append(jpgfile)
            self.lblfiles.append(lblfile)

        f.close()
    
    def get_onehot_lbl(self, lbl_path):
        name = '/' + '/'.join(lbl_path.split('.')[:-1]) + '_onehot.npy'
        if os.path.exists(name):
            return np.load(name)
        else:
            return None
    
    def __getitem__(self, index):

        folder_path = self.jpgfiles[index]
        label_path = self.lblfiles[index]

        imgs = []
        lbls = []
        lbls_onehot = []
        patches = []
        target_imgs = []

        frame_num = len(os.listdir(folder_path)) + self.videoLen

        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        
        lbl_paths = []
        img_paths = []
        for i in range(frame_num):
            if i < self.videoLen:
                img_path = folder_path + "/{:05d}.jpg".format(0)
                lbl_path = label_path + "/{:05d}.png".format(0)
            else:
                img_path = folder_path + "/{:05d}.jpg".format(i - self.videoLen)
                lbl_path = label_path + "/{:05d}.png".format(i - self.videoLen)
            
            img = load_image(img_path)  # CxHxW
            ht, wd = img.size(1), img.size(2)
            newh, neww = ht, wd

            if ht <= wd:
                ratio  = 1.0 #float(wd) / float(ht)
                # width, height
                img = resize(img, int(self.imgSize * ratio), self.imgSize)
                newh = self.imgSize
                neww = int(self.imgSize * ratio)
            else:
                ratio  = 1.0 #float(ht) / float(wd)
                # width, height
                img = resize(img, self.imgSize, int(self.imgSize * ratio))
                newh = int(self.imgSize * ratio)
                neww = self.imgSize

            if i == 0:
                imgs = torch.Tensor(frame_num, 3, newh, neww)

            img = color_normalize(img, mean, std)
            imgs[i] = img
            lblimg  = cv2.imread(lbl_path)
            lblimg  = cv2.resize(lblimg, (newh, neww), cv2.INTER_NEAREST)

            lbls.append(lblimg.copy())
            
            lbl_onehot = self.get_onehot_lbl(lbl_path)
            if lbl_onehot is not None:
                lbls_onehot.append(lbl_onehot)
            else:
                lbls_onehot.append(np.zeros(1))
            
            img_paths.append(img_path)
            lbl_paths.append(lbl_path)


        # Meta info
        meta = dict(folder_path=folder_path, img_paths=img_paths, lbl_paths=lbl_paths)

        lbls_tensor = torch.Tensor(len(lbls), newh, neww, 3)
        for i in range(len(lbls)):
            lbls_tensor[i] = torch.from_numpy(lbls[i])

        return imgs, lbls_tensor, np.stack(lbls_onehot), meta

    def __len__(self):
        return len(self.jpgfiles)
