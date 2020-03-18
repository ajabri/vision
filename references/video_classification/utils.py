from __future__ import print_function
from collections import defaultdict, deque
import datetime
import time
import torch
import torch.distributed as dist

import errno
import os

import sys

def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb) # more "modern"

sys.excepthook = info


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)


import math
import numbers
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)



# Video's features
import numpy as np
from sklearn.decomposition import PCA
import cv2
import imageio as io

import visdom
import time
import PIL
import torchvision
import transforms as T
import skimage

import utils

def pca_feats(ff, solver='auto', img_normalize=True):
    ## expect ff to be   N x C x H x W
    
    N, C, H, W = ff.shape
    pca = PCA(
        n_components=3,
        svd_solver=solver,
        whiten=False
    )
#     print(ff.shape)
    ff = ff.transpose(1, 2).transpose(2, 3)
#     print(ff.shape)
    ff = ff.reshape(N*H*W, C).numpy()
#     print(ff.shape)
    
    pca_ff = torch.Tensor(pca.fit_transform(ff))
    pca_ff = pca_ff.view(N, H, W, 3)
#     print(pca_ff.shape)
    pca_ff = pca_ff.transpose(3, 2).transpose(2, 1)

    if img_normalize:
        pca_ff = (pca_ff - pca_ff.min()) / (pca_ff.max() - pca_ff.min())

    return pca_ff


def make_gif(video, outname='/tmp/test.gif', sz=256):
    if hasattr(video, 'shape'):
        video = video.cpu()
        if video.shape[0] == 3:
            video = video.transpose(0, 1)

        video = video.numpy().transpose(0, 2, 3, 1)
        # print(video.min(), video.max())
        
        video = (video*255).astype(np.uint8)
#         video = video.chunk(video.shape[0])
        
    video = [cv2.resize(vv, (sz, sz)) for vv in video]

    if outname is None:
        return np.stack(video)

    io.mimsave(outname, video, duration = 0.2)


from matplotlib import cm
import time
import cv2

class PatchGraph(object):
    
    color = cm.get_cmap('jet')
    pad = 0
    
    def blend(self, i):

        y, x = i // self.W, i % self.W
        cx, cy = [int((self.w + self.pad) * (x  + 0.5)), int((self.h + self.pad) * (y  + 0.5))]

        # if self.orig is None:
        img1 = self.grid[...,:-self.pad, :-self.pad] if self.pad > 0 else self.grid
        # else:
        img2 = self.orig

        img1 = (0.5 * self.maps[i] + 0.5 * img1).copy() * 255
        img2 = (0.5 * self.maps[i] + 0.5 * img2).copy() * 255

        img1[:, cy-5:cy+5, cx-5:cx+5] = 255
        img2[:, cy-5:cy+5, cx-5:cx+5] = 255
        # img = cv2.circle(img.transpose(1,2,0), ctr, 10, (255, 255, 255), -1).get().transpose(2, 0, 1)

        return img1, img2

    def update(self):
        self.viz.image(self.curr[0], win=self.win_id, env=self.viz.env+'_pg')
        # self.viz.image(self.curr[1], win=self.win_id2, env=self.viz.env+'_pg')

    def __init__(self, viz, I, A, win='patchgraph', orig=None):
        self._win = win
        self.viz = viz
        self._birth = time.time()

        _, C, h, w = I.shape
        N = A.shape[-1]
        H = W = int(N ** 0.5)
        self.N, self.H, self.W, self.h, self.w = N, H, W, h, w

        I = I.cpu()
        orig = orig.cpu()
        A = A.view(H * W, H, W).cpu() #.transpose(-1, -2)

        # psize = min(2000 // H, I.shape[-1])
        # if psize < I.shape[-1]:
        #     I = [cv2.resize(ii, (psize, psize)) for ii in I]

        ####################################################################
        # Construct image data
        if N == 1:
            self.grid = cv2.resize(orig[0].numpy().transpose(1,2,0), (800, 800), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        else:
            self.grid = torchvision.utils.make_grid(I, nrow=H, padding=self.pad, pad_value=0).cpu().numpy()

        self.grid -= self.grid.min()
        self.grid /= self.grid.max()
        
        if orig is not None:
            self.orig = cv2.resize(orig[0].numpy().transpose(1,2,0), self.grid.shape[-2:], interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
            self.orig -= self.orig.min()
            self.orig /= self.orig.max()
        else:
            self.orig = None
        # import pdb; pdb.set_trace()

        ####################################################################
        # Construct map data

        # pstride = utils.get_stride(orig.shape[-1], h, H)   # stride length used to gen patches, h is patch size, H is n patch per side
        # map_sz_ratio = (pstride * H ) / orig.shape[-1]     # compute percentage of image spanned by affinity map overlay
        # map_sz = int(map_sz_ratio * self.orig.shape[-1])
        # lpad = int(((h-pstride)//2 / orig.shape[-1]) * self.orig.shape[-1])
        # rpad = self.orig.shape[-1] - map_sz - lpad

        map_sz = self.orig.shape[-1]
        lpad, rpad = 0, 0

        zeros = np.zeros(self.orig.shape).transpose(1,2,0)
        maps = []
        for a in A[:, :, :, None].cpu().detach().numpy():
            _a = cv2.resize(a, (map_sz, map_sz), interpolation=cv2.INTER_NEAREST)
            _a = self.color(_a)[...,:3]
            a = zeros.copy()
            if lpad > 0 and rpad > 0:
                a[lpad:-rpad, lpad:-rpad, :] = _a
            else:
                a = _a
            maps.append(a)

        self.maps = np.array(maps).transpose(0, -1, 1, 2)

        ####################################################################
        # Set first image

        self.curr_id = N//2
        self.curr = self.blend(self.curr_id)
        # viz.text('', opts=dict(width=10000, height=2), env=viz.env+'_pg')
        
        self.win_id = self._win 
        self.win_id2 = self._win+'2'

        self.update()
        ####################################################################

        def str2inttuple(s):
            try:
                ss = s.split(',')
                assert(len(ss) == 2)
                return int(ss[0]), int(ss[1])
            except:
                return False

        def callback(event):
            # nonlocal win_id #, win_id_text
            # print(event['event_type'])

            #TODO make the enter key recompute the A under a
            if event['event_type'] == 'KeyPress':
                # print(event['key'], 'KEYYYYY')

                if 'Arrow' in event['key']:
                    self.curr_id += {'ArrowLeft':-1, 'ArrowRight': 1, 'ArrowUp': -self.W, 'ArrowDown': self.W}[event['key']]
                    # print('hello!!', self.curr_id)
                    self.curr_id = min(max(self.curr_id, 0), N)
                    self.curr = self.blend(self.curr_id)
                    self.update()

                # curr_txt = event['pane_data']['content']

                # print(event['key'], 'KEYYYYY')
                # if event['key'] == 'Enter':
                #     itup = str2inttuple(curr_txt)
                #     if itup:
                #         self.curr = self.blend(itup[0]*H + itup[1])
                #         viz.image(self.curr, win=self.win_id, env=viz.env+'_pg')
                #         curr_txt='Set %s' % curr_txt
                #     else:
                #         curr_txt='Invalid position tuple'

                # elif event['key'] == 'Backspace':
                #     curr_txt = curr_txt[:-1]
                # elif event['key'] == 'Delete':
                #     curr_txt = ''
                # elif len(event['key']) == 1:
                #     curr_txt += event['key']
                

                # viz.text(curr_txt, win=self.win_id_text, env=viz.env+'_pg')

            if event['event_type'] == 'Click':
                # print(event.keys())
                # import pdb; pdb.set_trace()
                # viz.text(event)
                coords = "x: {}, y: {};".format(
                    event['image_coord']['x'], event['image_coord']['y']
                )
                viz.text(coords, win=self.win_id_text, env=viz.env+'_pg')

                self.curr = self.blend(np.random.randint(N))

                viz.image(self.curr, win=self.win_id, env=viz.env+'_pg')


        viz.register_event_handler(callback, self.win_id)
        # viz.register_event_handler(callback, self.win_id_text)
        # import pdb; pdb.set_trace()


class Visualize(object):
    def __init__(self, args, suffix='metrics', log_interval=2*60):
        self.vis = visdom.Visdom(
            port=args.port,
            server='http://%s' % args.server,
            env="%s_%s" % (args.name, suffix),
        )
        self.data = dict()

        self.log_interval = log_interval
        self._last_flush = time.time()

    def log(self, key, value):
        if not key in self.data:
            self.data[key] = [[],[]]

        if isinstance(value, tuple):
            self.data[key][0].append(value[0])
            self.data[key][1].append(value[1])
        else:
            self.data[key][1].append(value)
            self.data[key][0].append(len(self.data[key][1]) * 1.0)
            # import pdb; pdb.set_trace()

        if (time.time() - self._last_flush) > (self.log_interval):
            for k in self.data:
                self.vis.line(
                    X=np.array(self.data[k][0]),
                    Y=np.array(self.data[k][1]),
                    win=k,
                    opts=dict( title=k )
                )
            self._last_flush = time.time()

    def nn_patches(self, P, A_k, prefix='', N=10, K=20):
        nn_patches(self.vis, P, A_k, prefix, N, K)


def get_stride(im_sz, p_sz, res):
    stride = (im_sz - p_sz)//(res-1)
    return stride

def nn_patches(vis, P, A_k, prefix='', N=10, K=20):
    # produces nearest neighbor visualization of N patches given an affinity matrix with K channels

    P = P.cpu().detach().numpy()
    P -= P.min()
    P /= P.max()

    A_k = A_k.cpu().detach().numpy() #.transpose(-1,-2).numpy()
    # assert np.allclose(A_k.sum(-1), 1)

    A = np.sort(A_k, axis=-1)
    I = np.argsort(-A_k, axis=-1)
    
    vis.text('', opts=dict(width=10000, height=1), win='%s_patch_header' %(prefix))


    for n,i in enumerate(np.random.permutation(P.shape[0])[:N]):
        p = P[i]
        vis.text('', opts=dict(width=10000, height=1), win='%s_patch_header_%s' % (prefix, n))
        # vis.image(p,  win='%s_patch_query_%s' % (prefix, n))

        for k in range(I.shape[0]):
            vis.images(P[I[k, i, :K]], nrow=min(I.shape[-1], 20), win='%s_patch_values_%s_%s' % (prefix, n, k))
            vis.bar(A[k, i][::-1][:K], opts=dict(height=150, width=500), win='%s_patch_affinity_%s_%s' % (prefix, n, k))


def unnormalize(x):
    t = transforms.Normalize(
        -np.array([0.4914, 0.4822, 0.4465])/np.array([0.2023, 0.1994, 0.2010]),
        1/np.array([0.2023, 0.1994, 0.2010]).tolist()
    )
    return t(x)

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
        train_transform += [
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        ]

    if 'flip' in args.frame_aug:
        train_transform += [transforms.RandomHorizontalFlip()]

    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    train_transform = transforms.Compose(train_transform)

    print('Frame augs:', train_transform, args.frame_aug)
    def aug(x):
        if 'grid' in args.frame_aug:
            return patch_grid(x, transform=train_transform,
                shape=args.patch_size, stride=args.pstride)
        elif 'randpatch' in args.frame_aug:
            return n_patches(x, args.npatch, transform=train_transform,
                shape=args.patch_size, scale=args.npatch_scale)
        else:
            return train_transform(x)


    return aug


def get_frame_transform(args):
    imsz = args.img_size
    norm_size = torchvision.transforms.Resize((imsz, imsz))

    tt = []
    fts = args.frame_transforms#.split(',')

    # torchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0),
    # torchvision.transforms.RandomGrayscale(p=0.1)
    # torchvision.transforms.RandomHorizontalFlip(p=0.5)
    # torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)
    # torchvision.transforms.Resize(256, 256),
    
    # import pdb; pdb.set_trace()
    if 'blur' in fts:
        tt += [torchvision.transforms.ToTensor(), T.GaussianBlurTransform(), torchvision.transforms.ToPILImage()]
        
    # if 'gray' in fts:
    #     tt.append(torchvision.transforms.RandomGrayscale(p=1))

    if 'crop' in fts:
        tt.append(torchvision.transforms.RandomResizedCrop(
            imsz, scale=(0.8, 0.95), ratio=(0.7, 1.3), interpolation=2),)
    else:
        tt.append(norm_size)

    if 'cj' in fts:
        tt += [
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
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


    frame_transform_train = T.PerTimestepTransform(
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
        # import pdb; pdb.set_trace()
        return frame_transform_train(x) if not args.visualize else T.PerTimestepTransform(plain1)(x), plain(x[0].transpose(2, 0, 1))

    return with_orig

# def vis_flow(u, v):
#     flows = []
#     u, v = u.data.cpu().numpy().astype(np.float32), v.data.cpu().numpy().astype(np.float32)
#     hsv = np.zeros((u.shape[1], u.shape[2], 3))
    
#     for i in range(u.shape[0]):
#         mag, ang = cv2.cartToPolar(u[i], v[i])
#         hsv[...,1] = 255
#         hsv[...,0] = ang*180/np.pi/2
#         hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#         bgr = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2BGR)
#         flows.append(bgr)

#     return flows

# def compute_flow(corr):
#     nnf = corr.argmax(dim=1)
#     # nnf = nnf.transpose(-1, -2)

#     u = nnf % nnf.shape[-1]
#     v = nnf / nnf.shape[-2] # nnf is an IntTensor so rounds automatically

#     rr = torch.arange(u.shape[-1])[None].long().cuda()

#     for i in range(u.shape[-1]):
#         u[:, i] -= rr

#     for i in range(v.shape[-1]):
#         v[:, :, i] -= rr

#     flows = vis_flow(u, v)

#     return flows, u, v


def compute_flow(corr):
    # assume corr is shape N x H * W x W x H
    nnf = corr.argmax(dim=1)
    nnf = nnf.transpose(-1, -2)

    u = nnf % nnf.shape[-1]
    v = nnf / nnf.shape[-2] # nnf is an IntTensor so rounds automatically

    rr = torch.arange(u.shape[-1])[None].long().cuda()

    for i in range(u.shape[-1]):
        u[:, i] -= rr

    for i in range(v.shape[-1]):
        v[:, :, i] -= rr

    return u, v
        
import cv2
import numpy as np

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)#[::-1]
    return bgr




### Network Utils

import cmc_resnet

def load_selfsup_model():
    r50 = cmc_resnet.InsResNet50()
    r50.load_state_dict(torch.load('/home/jabreezus/clones/CMC/models/MoCo_softmax_16384_epoch200.pth')['model'])
    return r50


class From3D(nn.Module):
    def __init__(self, resnet):
        super(From3D, self).__init__()
        self.model = resnet
    
    def forward(self, x):
        N, C, T, h, w = x.shape
        
        xx = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, h, w)
        mm = self.model(xx)
        # import pdb; pdb.set_trace()

        return mm.view(N, T, *mm.shape[-3:]).permute(0, 2, 1, 3, 4)

def make_encoder(model_type='scratch'):
    import resnet3d
    import resnet2d
    import antialiased as aa
    import antialiased.resnet as aa_resnet
    if model_type == 'scratch':
        import torchvision.models.video.resnet as _resnet3d
        # resnet = resnet3d.r2d_10(pretrained=False)
        resnet = resnet2d.resnet18(pretrained=False,)
        # resnet = aa_resnet.resnet18(pretrained=False)
        # norm_layer=lambda x: nn.GroupNorm(1, x))
        # resnet = resnet2d.resnet34(pretrained=False,)        

    elif model_type == 'bagnet':
        import bagnet
        resnet = bagnet.bagnet17(pretrained=True)
        
    elif model_type == 'imagenet':
        resnet2d._REFLECT_PAD = False
        resnet = resnet2d.resnet18(pretrained=True)

    elif model_type == 'moco':
        resnet = load_selfsup_model().encoder

        # self.resnet = _resnet3d.r3d_18(pretrained=True)
        # self.resnet = resnet3d.r2d_18(pretrained=True)


    resnet.fc, resnet.avgpool, = None, None
    resnet.layer4 = None

    if 'Conv2d' in str(resnet):
        resnet = From3D(resnet)

    return resnet

def make_stack_encoder(in_dim, out_dim=None):
    import resnet3d
    import torchvision.models.video.resnet as _resnet3d
    
    dim = in_dim
    out_dim = out_dim if out_dim is not None else dim

    return _initialize_weights(nn.Sequential(
        resnet3d.Conv3DChooseTemporal(dim, dim, stride=2),
        nn.BatchNorm3d(dim),
        nn.ReLU(inplace=True),

        nn.Conv3d(dim, dim, stride=(1, 2, 2), kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        nn.BatchNorm3d(dim),
        nn.ReLU(inplace=True),

        nn.Conv3d(dim, out_dim, stride=(1, 2, 2), kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        nn.BatchNorm3d(out_dim),
        nn.ReLU(inplace=True),
    ))

def make_aff_encoder(in_dim, out_dim=None):
    out_dim = out_dim if out_dim is not None else in_dim
    
    return _initialize_weights(nn.Sequential(*[
        nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1),
        nn.BatchNorm2d(in_dim),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1),
        nn.BatchNorm2d(in_dim),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(inplace=True),
    ]))

def _initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    return model