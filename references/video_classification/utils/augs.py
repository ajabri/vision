import PIL
import torchvision
import skimage

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

import numpy as np
from PIL import Image

class PerTimestepTransform(object):
    def __init__(self, transforms, pil_convert=True):
        self.transforms = transforms
        self.pil_convert = pil_convert

    def __call__(self, vid):
        # import pdb; pdb.set_trace()

        if isinstance(vid, Image.Image):
            return np.stack([self.transforms(vid)])
        
        if isinstance(vid, torch.Tensor):
            vid = vid.numpy()

        # for idx in range(vid.shape[0]):
        #     vid[idx] = np.asarray(self.transforms(Image.fromarray(vid[idx]))) if self.pil_convert else self.transforms(vid[dx])
        # # return vid

        if self.pil_convert:
            x = np.stack([np.asarray(self.transforms(Image.fromarray(v))) for v in vid])
            return x
        else:
            return np.stack([self.transforms(v) for v in vid])
    

def n_patches(x, n, transform, shape=(64, 64, 3), scale=[0.2, 0.8]):

    if shape[-1] == 0:
        shape = np.random.uniform(64, 128)
        shape = (shape, shape, 3)

    crop = transforms.Compose([
        lambda x: PIL.Image.fromarray(x) if not 'PIL' in str(type(x)) else x,
        transforms.RandomResizedCrop(shape[0], scale=scale)
    ])    

    if torch.is_tensor(x):
        x = x.numpy().transpose(1,2, 0)
    
    P = []
    for _ in range(n):
        xx = transform(crop(x))
        P.append(xx)


    return torch.cat(P, dim=0)


def patch_grid(x, transform, shape=(64, 64, 3), stride=[1.0, 1.0]):
    stride = np.random.random() * (stride[1] - stride[0]) + stride[0]
    stride = [int(shape[0]*stride), int(shape[1]*stride), shape[2]]

    spatial_jitter = transforms.Compose([
        lambda x: PIL.Image.fromarray(x),
        transforms.RandomResizedCrop(shape[0], scale=(0.7, 0.9))
    ])

    if torch.is_tensor(x):
        x = x.numpy().transpose(1, 2, 0)
    elif 'PIL' in str(type(x)):
        x = np.array(x)#.transpose(2, 0, 1)
    
    winds = skimage.util.view_as_windows(x, shape, step=stride)
    winds = winds.reshape(-1, *winds.shape[-3:])

    # import pdb; pdb.set_trace()
    # winds = torch.from_numpy(winds).contiguous().view(-1, *winds.shape[-3:])
    # patches = sklearn.feature_extraction.image.extract_patches_2d(x, shape[1:], 0.0002)

    P = [transform(spatial_jitter(w)) for w in winds]

    # import pdb; pdb.set_trace()

    return torch.cat(P, dim=0)


def get_frame_aug(args):
    train_transform = []

    if 'cj' in args.frame_aug:
        _cj = 0.1
        train_transform += [
            #transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(_cj, _cj, _cj, 0),
        ]

    if 'flip' in args.frame_aug:
        train_transform += [transforms.RandomHorizontalFlip()]

    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    train_transform = transforms.Compose(train_transform)

    print('Frame augs:', train_transform, args.frame_aug)

    # HACK if you up value the args field, it is a pointer!!!
    patch_size = np.array(args.patch_size)

    def aug(x):
        if 'grid' in args.frame_aug:
            return patch_grid(x, transform=train_transform,
                shape=patch_size, stride=args.pstride)
        elif 'randpatch' in args.frame_aug:
            return n_patches(x, args.npatch, transform=train_transform,
                shape=patch_size, scale=args.npatch_scale)
        else:
            return train_transform(x)


    return aug


def get_frame_transform(args):
    imsz = args.img_size
    norm_size = torchvision.transforms.Resize((imsz, imsz))

    tt = []
    fts = args.frame_transforms#.split(',')

    if 'crop' in fts:
        tt.append(torchvision.transforms.RandomResizedCrop(
            imsz, scale=(0.8, 0.95), ratio=(0.7, 1.3), interpolation=2),)
    else:
        tt.append(norm_size)

    if 'cj' in fts:
        _cj = 0.1
        tt += [
            #transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(_cj, _cj, _cj, 0),
        ]

    if 'flip' in fts:
        tt.append(torchvision.transforms.RandomHorizontalFlip())
    
    if args.npatch > 1 and args.frame_aug != '':
        tt += [get_frame_aug(args)]
    else:
        tt += [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
         
    print('Frame transforms:', tt, args.frame_transforms)


    frame_transform_train = PerTimestepTransform(
            torchvision.transforms.Compose(tt)
        )
    plain1 = torchvision.transforms.Compose([
        norm_size, 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    plain = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        plain1
    ])

    def with_orig(x):
        patchify = (not args.visualize)  or True
        
        if 'numpy' in str(type(x[0])):
            x = frame_transform_train(x) if patchify \
                    else PerTimestepTransform(plain1)(x), plain(x[0])
        else:
            x = frame_transform_train(x) if patchify \
                    else PerTimestepTransform(plain1)(x), plain(x[0].permute(2, 0, 1))

        return x

    return with_orig

