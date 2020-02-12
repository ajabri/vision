import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data
import random
import cv2
from PIL import Image

import torchvision.transforms as transforms

import scipy.io as sio
import scipy.misc

class VideoList(data.Dataset):
    def __init__(self, filelist, clip_len, is_train=True, frame_gap=1, transform=None):

        self.filelist = filelist
        self.clip_len = clip_len
        self.is_train = is_train
        self.frame_gap = frame_gap

        self.transform = transform
        
        f = open(self.filelist, 'r')
        self.jpgfiles = []
        self.fnums = []

        for line in f:
            rows = line.split()
            jpgfile = rows[0]
            fnum = int(rows[1])

            self.jpgfiles.append(jpgfile)
            self.fnums.append(fnum)

        # import pdb; pdb.set_trace()
        f.close()


    def __getitem__(self, index):
        index = index % len(self.jpgfiles)
        folder_path = self.jpgfiles[index]
        fnum = self.fnums[index]

        frame_gap = self.frame_gap
        startframe = 0
        
        readjust = False
        
        while fnum - self.clip_len * frame_gap < 0:
            frame_gap -= 1
            readjust = True

        if readjust:
            print('framegap adjusted to ', frame_gap, 'for', folder_path)
        
        diffnum = fnum - self.clip_len * frame_gap
        startframe = random.randint(0, diffnum)

        files = os.listdir(folder_path)
        imgs = []

        # reading video
        for i in range(self.clip_len):
            idx = int(startframe + i * frame_gap)

#             import pdb; pdb.set_trace()
            img_path = "%s/%s" % (folder_path, files[idx])
            img = cv2.imread(img_path) #.astype(np.float32)
            # print(img.shape)
            imgs.append(img)

        imgs = np.stack(imgs)
        # import pdb; pdb.set_trace()
        # if self.frame_transform is not None:
        #     imgs = [np.asarray(self.frame_transform(Image.fromarray(img))) for img in imgs]

        # imgs = torch.from_numpy(np.stack(imgs))

        if self.transform is not None:
            imgs = self.transform(imgs)


        return imgs, torch.tensor(0), torch.tensor(0)

    def __len__(self):
        return len(self.jpgfiles) * 1000