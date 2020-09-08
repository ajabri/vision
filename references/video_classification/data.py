import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import kornia
import kornia.augmentation as K

import transforms as T
import numpy as np

IMG_MEAN = torch.Tensor([0.4914, 0.4822, 0.4465])
IMG_STD  = torch.Tensor([0.2023, 0.1994, 0.2010])


'''

Video-level:
    Crop, color aug, flip

Frame-level:
    Crop, color aug, flip

Patch-level:
    Crop, color aug, flip
'''

class MapTransform(object):
    def __init__(self, transforms, pil_convert=True):
        self.transforms = transforms
        self.pil_convert = pil_convert

    def __call__(self, vid):
        # if isinstance(vid, Image.Image):
        #     return np.stack([self.transforms(vid)])
        
        # if isinstance(vid, torch.Tensor):
        #     vid = vid.numpy()

        return torch.stack([self.transforms(v) for v in vid])
    

def n_patches(x, n, transform, shape=(64, 64, 3), scale=[0.2, 0.8]):
    if shape[-1] == 0:
        shape = np.random.uniform(64, 128)
        shape = (shape, shape, 3)

    crop = K.RandomResizedCrop(size=(shape[0]), scale=scale, ratio=(0.7, 1.3))
    if torch.is_tensor(x):
        x = x.numpy().transpose(1,2, 0)
    
    P = []
    for _ in range(n):
        xx = transform(crop(x))
        P.append(xx)

    # import pdb;

    return torch.cat(P, dim=0)

def patch_grid(x, transform, shape=(64, 64, 3), stride=[1.0, 1.0]):
    stride = np.random.random() * (stride[1] - stride[0]) + stride[0]
    stride = [int(shape[0]*stride), int(shape[1]*stride), shape[2]]

    spatial_jitter = K.RandomResizedCrop(size=shape[:2], scale=(0.7, 0.9), ratio=(0.7, 1.3))

    import time
    t0 = time.time()
    x1 = x.unfold(2, 64, 32).unfold(3, 64, 32)
    t1 = time.time()
    x = kornia.contrib.extract_tensor_patches(x, 
        window_size=shape[:2], stride=stride[:2])
    t2 = time.time()
    print(t2-t1, t1-t0)
    # import pdb; pdb.set_trace()
    # x = x.view(3, _sz, _sz, x.shape[-1])

    T, N, C = x.shape[:3]
    x = transform(spatial_jitter(x.flatten(0,1))).view(T, N*C, *x.shape[3:])

    return x

    # self.k_frame = nn.Sequential(
    #     # kornia.color.Normalize(mean=-mmm/sss, std=1/sss),
    #     # K.ColorJitter(0.1, 0.1, 0.1, 0),
    #     # K.RandomResizedCrop(size=(256, 256), scale=(0.8, 0.9), ratio=(0.7, 1.3)),
    #     # kornia.color.Normalize(mean=mmm, std=sss)
    # )
    
    # self.k_frame_same = nn.Sequential(
    #     kornia.geometry.transform.Resize(256 + 20),
    #     K.RandomHorizontalFlip(same_on_batch=True),
    #     K.RandomCrop((256, 256), same_on_batch=True),
    # )

    
def get_frame_aug(args):
    train_transform = []

    if 'cj' in args.frame_aug:
        _cj = 0.1
        train_transform += [
            #K.RandomGrayscale(p=0.2),
            K.ColorJitter(_cj, _cj, _cj, 0),
        ]

    if 'flip' in args.frame_aug:
        train_transform += [
            K.RandomHorizontalFlip(same_on_batch=True),
        ]

    train_transform.append(kornia.color.Normalize(mean=IMG_MEAN, std=IMG_STD))
    
    train_transform = nn.Sequential(*train_transform)
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


def get_frame_transform(args, cuda=True):
    imsz = args.img_size
    norm_size = kornia.geometry.transform.Resize((imsz, imsz))
    norm_imgs = kornia.color.Normalize(mean=IMG_MEAN, std=IMG_STD)

    tt = []
    fts = args.frame_transforms#.split(',')

    if 'gray' in fts:
        tt.append(K.RandomGrayscale(p=1))

    if 'crop' in fts:
        tt.append(K.RandomResizedCrop(imsz, scale=(0.8, 0.95), ratio=(0.7, 1.3)))
    else:
        tt.append(norm_size)

    if 'cj2' in fts:
        _cj = 0.2
        tt += [
            K.RandomGrayscale(p=0.2),
            K.ColorJitter(_cj, _cj, _cj, _cj),
        ]
    elif 'cj' in fts:
        _cj = 0.1
        tt += [
            # K.RandomGrayscale(p=0.2),
            K.ColorJitter(_cj, _cj, _cj, 0),
        ]

    if 'flip' in fts:
        tt += [K.RandomHorizontalFlip()]
    
    if args.npatch > 1 and args.frame_aug != '':
        tt += [get_frame_aug(args)]
    else:
        tt += [norm_imgs]

    print('Frame transforms:', tt, args.frame_transforms)

    # frame_transform_train = MapTransform(transforms.Compose(tt))
    frame_transform_train = transforms.Compose(tt)
    plain = nn.Sequential(norm_size, norm_imgs)


    def with_orig(x):
        if cuda:
            x = x.cuda()
        if x.max() > 1 and x.min() >= 0:
            x = x.float()
            x -= x.min()
            x /= x.max()
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)

        patchify = (not args.visualize)  or True
        
        x = (frame_transform_train(x) if patchify else plain(x)).cpu(), \
                plain(x[0:1]).cpu()

        return x

    return with_orig
