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
from PIL import Image

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

    # print(lbl_set)
    # if (lbl_set > 20).sum() > 0:
    #     import pdb; pdb.set_trace()
    # count_lbls = [np.all(flat_lbls_0 == ll, axis=-1).sum() for ll in lbl_set]
    
    print('lbls', time.time() - t00)

    return lbl_set


def texturize(onehot):
    flat_onehot = onehot.reshape(-1, onehot.shape[-1])
    lbl_set = np.unique(flat_onehot, axis=0)

    count_lbls = [np.all(flat_onehot == ll, axis=-1).sum() for ll in lbl_set]
    object_id = np.argsort(count_lbls)[::-1][1]

    hidxs = []
    for h in range(onehot.shape[0]):
        # appears = any(np.all(onehot[h] == lbl_set[object_id], axis=-1))
        appears = np.any(onehot[h, :, 1:] == 1)
        if appears:    
            hidxs.append(h)

    nstripes = min(10, len(hidxs))

    out = np.zeros((*onehot.shape[:2], nstripes+1))
    out[:, :, 0] = 1

    for i, h in enumerate(hidxs):
        cidx = int(i // (len(hidxs) / nstripes))
        w = np.any(onehot[h, :, 1:] == 1, axis=-1)
        out[h][w] = 0
        out[h][w, cidx+1] = 1
        # print(i, h, cidx)

#    import pdb; pdb.set_trace()

    return out

def hard_prop(predlbls, axis=-1):
    pred_max = predlbls.max(axis=axis)[..., None]
    predlbls[predlbls <  pred_max] = 0
    predlbls[predlbls >= pred_max] = 1
    predlbls /= predlbls.sum(axis)[..., None]
    return predlbls

# def resize_labels(lbl, w, h):

# lbl_r  = cv2.resize(lbl, (w, h), cv2.INTER_LINEAR)
# maxx = lbl_r.max()
# if maxx <= 1.1:

#     lblimg = (((lblimg2 + 1) / 128).round() * 128).astype(lblimg.dtype)

######################################################################

from matplotlib import cm

class DavisSet(data.Dataset):
    def __init__(self, args, is_train=True):

        self.filelist = args.filelist
        self.imgSize = args.imgSize
        self.videoLen = args.videoLen
        self.mapScale = args.mapScale

        self.texture = args.texture
        self.round = args.round

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
    

    def make_paths(self, folder_path, label_path):
        I, L = os.listdir(folder_path), os.listdir(label_path)
        L = [ll for ll in L if 'npy' not in ll]

        frame_num = len(I) + self.videoLen
        I.sort(key=lambda x:int(x.split('.')[0]))
        L.sort(key=lambda x:int(x.split('.')[0]))

        I_out, L_out = [], []

        for i in range(frame_num):
            i = max(0, i - self.videoLen)
            img_path = "%s/%s" % (folder_path, I[i])
            lbl_path = "%s/%s" % (label_path,  L[i])

            I_out.append(img_path)
            L_out.append(lbl_path)

        return I_out, L_out


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

        #mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        img_paths, lbl_paths = self.make_paths(folder_path, label_path)

        t000 = time.time()

        # frame_num = 30
        for i in range(frame_num):
            t00 = time.time()

            img_path, lbl_path = img_paths[i], lbl_paths[i]
            
            img = load_image(img_path)  # CxHxW

            # if 'davis' in lbl_path:
            #     lblimg = Image.open(lbl_path)
            #     lblimg = np.array(lblimg, dtype=np.int32)[..., None]
            # else:
            lblimg = cv2.imread(lbl_path)

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

                if self.round:
                    lblimg2  = cv2.resize(lblimg, (newh, neww), cv2.INTER_LINEAR)
                    lblimg = (((lblimg2 + 0) / 128).round() * 128).astype(lblimg.dtype)
                else:
                    lblimg = cv2.resize(lblimg, (newh, neww), cv2.INTER_NEAREST)

            img_orig = img.clone()
            img = color_normalize(img, mean, std)

            imgs_orig.append(img_orig)
            imgs.append(img)
            lbls.append(lblimg.copy())
            
        # Meta info
        meta = dict(folder_path=folder_path, img_paths=img_paths, lbl_paths=lbl_paths)

        ########################################################
        # Prepare reshaped label information
        lbls = np.stack(lbls)
        prefix = '/' + '/'.join(lbl_paths[0].split('.')[:-1])

        # Get lblset
        lblset_path = "%s_%s.npy" % (prefix, 'lblset')
        lblset = make_lbl_set(lbls)

        if np.all((lblset[1:] - lblset[:-1]) == 1):
            lblset = lblset[:, 0:1]

        # import pdb; pdb.set_trace()
        # lblset = try_np_load(lblset_path)
        # if lblset is None or True:
        #     print('making label set', lblset_path)
        #     lblset = make_lbl_set(lbls)
        #     np.save(lblset_path, lblset)

        # if len(lblset) != 20 or np.array(lblset).max() >= 20:
        #     import pdb; pdb.set_trace()

        onehots = []
        resizes = []

        rsz_h, rsz_w = math.ceil(img.size(1) / self.mapScale[0]), math.ceil(img.size(2) /self.mapScale[1])

        for i,p in enumerate(lbl_paths):
            prefix = '/' + '/'.join(p.split('.')[:-1])
            # print(prefix)
            oh_path = "%s_%s.npy" % (prefix, 'onehot')
            rz_path = "%s_%s.npy" % (prefix, 'size%sx%s' % (rsz_h, rsz_w))

            onehot = try_np_load(oh_path) 
            if onehot is None or True:
                print('computing onehot lbl for', oh_path)
                onehot = np.stack([np.all(lbls[i] == ll, axis=-1) for ll in lblset], axis=-1)
                np.save(oh_path, onehot)

            resized = try_np_load(rz_path)
            if resized is None or True:
                print('computing resized lbl for', rz_path)
                resized = cv2.resize(np.float32(onehot), (rsz_w, rsz_h), cv2.INTER_LINEAR)
                # import pdb; pdb.set_trace()
                
                # NOTE give boost for color model_5, mainly for J score
                # resized[:,:,0] -= 0.0001
                # resized = hard_prop(resized)

                # resized = np.round(resized)
                # import pdb; pdb.set_trace()
                # resized = onehot[::self.mapScale[0], ::self.mapScale[1]] * 1.0

                np.save(rz_path, resized)

            if self.texture:
                texturized = texturize(resized)
                resizes.append(texturized)
                lblset = np.array([[0, 0, 0]] + [cm.Paired(i)[:3] for i in range(texturized.shape[-1])]) * 255.0
                break
            else:
                resizes.append(resized)
                onehots.append(onehot)

            # import pdb; pdb.set_trace()

            # print('frame', i, time.time() - t00)

        if self.texture:
            resizes = resizes * self.videoLen
            for _ in range(len(lbl_paths)-self.videoLen):
                resizes.append(np.zeros(resizes[0].shape))
            onehots = resizes

        ########################################################
        
        imgs = torch.stack(imgs)
        imgs_orig = torch.stack(imgs_orig)
        lbls_tensor = torch.from_numpy(np.stack(lbls))
        lbls_resize = np.stack(resizes)
        # lbls_onehot = np.stack(onehots)

        assert lbls_resize.shape[0] == len(meta['lbl_paths'])

        print('vid', i, 'took', time.time() - t000)

        # import pdb; pdb.set_trace()

        # return imgs, imgs_orig, lblset, lbls_tensor, lbls_onehot, lbls_resize, meta
        return imgs, imgs_orig, lblset, lbls_tensor, lbls_resize, lbls_resize, meta


    def __len__(self):
        return len(self.jpgfiles)




# class DavisResizedLabels(data.Dataset):
#     def __getitem__(self, index):
#         imgs, imgs_orig, lbls_tensor, lbls_onehot, lbls_resize, meta = super(DavisResizedLabels, self).__get__item(index)




