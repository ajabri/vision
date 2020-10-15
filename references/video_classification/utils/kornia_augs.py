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
    

def patch_grid(x, transform, shape=(64, 64, 3), stride=[1.0, 1.0]):
    stride = np.random.random() * (stride[1] - stride[0]) + stride[0]
    stride = [int(shape[0]*stride), int(shape[1]*stride), shape[2]]

    def transform(x):
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

    return transform

    
def get_frame_aug(frame_aug, patch_size):
    tt = []

    if 'cj' in frame_aug:
        _cj = 0.1
        tt += [
            #K.RandomGrayscale(p=0.2),
            K.ColorJitter(_cj, _cj, _cj, 0),
        ]

    if 'flip' in frame_aug:
        tt += [
            K.RandomHorizontalFlip(same_on_batch=True),
        ]

    tt += [kornia.color.Normalize(mean=IMG_MEAN, std=IMG_STD)]
    transform = nn.Sequential(*tt)
    print('Frame augs:', transform, frame_aug)

    if 'grid' in frame_aug:
        aug = patch_grid(x, transform=transform,
            shape=patch_size, stride=patch_size // 2)
    else:
        aug = transform
    
    return aug



def get_frame_transform(frame_transform_str, img_size, cuda=True):
    tt = []

    if 'gray' in frame_transform_str:
        tt += [K.RandomGrayscale(p=1)]

    if 'crop' in frame_transform_str:
        tt += [K.RandomResizedCrop(img_size, scale=(0.8, 0.95), ratio=(0.7, 1.3))]
    else:
        tt += [kornia.geometry.transform.Resize((img_size, img_size))]

    if 'cj2' in frame_transform_str:
        _cj = 0.2
        tt += [K.RandomGrayscale(p=0.2), K.ColorJitter(_cj, _cj, _cj, _cj)]

    elif 'cj' in frame_transform_str:
        _cj = 0.1
        tt += [K.ColorJitter(_cj, _cj, _cj, 0),]

    if 'flip' in frame_transform_str:
        tt += [K.RandomHorizontalFlip()]

    return tt


def get_train_transform(args, cuda=True):
    imsz = args.img_size
    norm_size = kornia.geometry.transform.Resize((imsz, imsz))
    norm_imgs = kornia.color.Normalize(mean=IMG_MEAN, std=IMG_STD)

    frame_transform = get_frame_transform(args.frame_transform, imsz, cuda)
    frame_aug = get_frame_aug(args.frame_aug, args.patch_size)

    transform = transforms.Compose(frame_transform + frame_aug)
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
        
        x = (transform(x) if patchify else plain(x)).cpu(), \
                plain(x[0:1]).cpu()

        return x

    return with_orig

# class Patchify(nn.Module):
#     def __init__(self, args):
#         p_sz, stride = 64, 32
#         self.k_patch =  nn.Sequential(
#             K.RandomResizedCrop(size=(p_sz, p_sz), scale=(0.7, 0.9), ratio=(0.7, 1.3))
#         )
#         # import pdb; pdb.set_trace()
#         self.k_frame = nn.Sequential(
#             # K.ColorJitter(0.1, 0.1, 0.1, 0),
#             # K.
#             # K.Normalize()
#             K.RandomResizedCrop(size=(256, 256), scale=(0.8, 0.9), ratio=(0.7, 1.3))
#         )
#         # self.k_frame_same = nn.Sequential(
#         #     K.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), same_on_batch=True)
#         # )
#         # self.k_frame_same = None
#         self.k_frame_same = nn.Sequential(
#             kornia.geometry.transform.Resize(256 + 20),
#             kornia.augmentation.RandomCrop((256, 256), same_on_batch=True),
#         )

#         self.unfold = torch.nn.Unfold((p_sz,p_sz), dilation=1, padding=0, stride=(stride, stride))


#     def patchify(self, x):
#         B, T, C, H, W = x.shape
#         _N, C = C//3, 3

#         _sz = self.unfold.kernel_size[0]
#         x = x.flatten(0, 1)


#         x = self.k_frame_same(x)
#         x = self.k_frame(x)
#         x = self.unfold(x)

#         # import pdb; pdb.set_trace()

#         x, _N = x.view(B, T, C, _sz, _sz, x.shape[-1]), x.shape[-1]
#         x = x.permute(0, -1, 1, 2, 3, 4)   # B x _N x T x C x H x W
#         x = x.flatten(0, 2)

#         x = self.k_patch(x)
#         x = x.view(B, _N, T, C, _sz, _sz).transpose(2, 3)

#         return x


