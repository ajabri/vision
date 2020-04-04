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

import time




######################################################################
def try_np_load(p):
    try:
        return np.load(p)
    except:
        return None

def make_lbl_set(lbls):
    print(lbls.shape)
    t00 = time.time()

    lbl_set = [np.zeros(3).astype(np.uint8)]
    count_lbls = [0]    
    
    flat_lbls_0 = lbls[0].copy().reshape(-1, lbls.shape[-1]).astype(np.uint8)
    lbl_set = np.unique(flat_lbls_0, axis=0)
    count_lbls = [np.all(flat_lbls_0 == ll, axis=-1).sum() for ll in lbl_set]
    
    print('lbls', time.time() - t00)
    
    # only keep labels that appear ten times?!
    lbl_set_temp = [ll for ii, ll in enumerate(lbl_set) if count_lbls[ii] > 10]
    lbl_set = lbl_set_temp
    print(lbl_set)
    print(count_lbls)

    return lbl_set


######################################################################


class DavisSet(data.Dataset):
    def __init__(self, params, is_train=True):

        self.filelist = params['filelist']
        self.imgSize = params['imgSize']
        self.videoLen = params['videoLen']
        self.mapScale = params['mapScale']

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
        imgs_orig = []
        lbls = []
        lbls_onehot = []
        patches = []
        target_imgs = []

        frame_num = len(os.listdir(folder_path)) + self.videoLen

        # mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        
        lbl_paths = []
        img_paths = []

        t000 = time.time()

        # frame_num = 30
        for i in range(frame_num):
            t00 = time.time()

            if i < self.videoLen:
                img_path = folder_path + "/{:05d}.jpg".format(0)
                lbl_path = label_path + "/{:05d}.png".format(0)
            else:
                img_path = folder_path + "/{:05d}.jpg".format(i - self.videoLen)
                lbl_path = label_path + "/{:05d}.png".format(i - self.videoLen)
            
            img = load_image(img_path)  # CxHxW
            lblimg  = cv2.imread(lbl_path)

            # print('loaded', i, time.time() - t00)

            ht, wd = img.size(1), img.size(2)
            if self.imgSize > 0:
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

                lblimg  = cv2.resize(lblimg, (newh, neww), cv2.INTER_NEAREST)

            img_orig = img.clone()
            img = color_normalize(img, mean, std)

            imgs_orig.append(img_orig)
            imgs.append(img)
            lbls.append(lblimg.copy())
            
            img_paths.append(img_path)
            lbl_paths.append(lbl_path)

        # Meta info
        meta = dict(folder_path=folder_path, img_paths=img_paths, lbl_paths=lbl_paths)


        ########################################################
        # Prepare reshaped label information

        lbls = np.stack(lbls)
        prefix = '/' + '/'.join(lbl_paths[0].split('.')[:-1])

        # Get lblset
        lblset_path = "%s_%s.npy" % (prefix, 'lblset')
        lblset = try_np_load(lblset_path)
        if lblset is None:
            print('making label set', lblset_path)
            lblset = make_lbl_set(lbls)
            np.save(lblset_path, lblset)

        onehots = []
        resizes = []

        rsz_h, rsz_w = img.size(1) // self.mapScale[0], img.size(2) // self.mapScale[1]

        for i,p in enumerate(lbl_paths):
            prefix = '/' + '/'.join(p.split('.')[:-1])
            # print(prefix)
            oh_path = "%s_%s.npy" % (prefix, 'onehot')
            rz_path = "%s_%s.npy" % (prefix, 'size%sx%s' % (rsz_h, rsz_w))

            onehot = try_np_load(oh_path) 
            if onehot is None:
                print('computing onehot lbl for', oh_path)
                onehot = np.stack([np.all(lbls[i] == ll, axis=-1) for ll in lblset], axis=-1)
                np.save(oh_path, onehot)

            resized = try_np_load(rz_path)
            if resized is None:
                print('computing resized lbl for', rz_path)
                resized = cv2.resize(np.float32(onehot), (rsz_h, rsz_w))
                np.save(rz_path, resized)
            
            onehots.append(onehot)
            resizes.append(resized)
            # print('frame', i, time.time() - t00)



        ########################################################
        
        imgs = torch.stack(imgs)
        imgs_orig = torch.stack(imgs_orig)
        lbls_tensor = torch.from_numpy(np.stack(lbls))
        lbls_onehot = np.stack(onehots)
        lbls_resize = np.stack(resizes)

        assert lbls_onehot.shape[0] == len(meta['lbl_paths'])

        print('vid', i, 'took', time.time() - t000)

        return imgs, imgs_orig, lblset, lbls_tensor, lbls_onehot, lbls_resize, meta

    def __len__(self):
        return len(self.jpgfiles)




# class DavisResizedLabels(data.Dataset):
#     def __getitem__(self, index):
#         imgs, imgs_orig, lbls_tensor, lbls_onehot, lbls_resize, meta = super(DavisResizedLabels, self).__get__item(index)




