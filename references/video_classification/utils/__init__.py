from collections import defaultdict, deque
import datetime
import time
import torch

import errno
import os
import sys

from . import arguments
from . import visualize
from . import augs

#########################################################
# DEBUG
#########################################################

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


#########################################################
# Meters
#########################################################

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
        import torch.distributed as dist
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


#################################################################################
### Network Utils
#################################################################################

import math
import numbers
from torch import nn
from torch.nn import functional as F
from torchvision import transforms


def partial_load(pretrained_dict, model, skip_keys=[]):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not any([sk in k for sk in skip_keys])}
    skipped_keys = [k for k in pretrained_dict if k not in filtered_dict]
    print('Skipped keys: ',  skipped_keys)

    # 2. overwrite entries in the existing state dict
    model_dict.update(filtered_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def load_cmc_model():
    from .models import cmc_resnet
    r50 = cmc_resnet.InsResNet50()
    r50.load_state_dict(torch.load('/home/jabreezus/clones/CMC/models/MoCo_softmax_16384_epoch200.pth')['model'])
    return r50


def load_vince_model(path):
    checkpoint = torch.load(path, map_location={'cuda:0': 'cpu'})
    checkpoint = {k.replace('feature_extractor.module.model.', ''): checkpoint[k] for k in checkpoint if 'feature_extractor' in k}
    return checkpoint

class From3D(nn.Module):
    '''
    Use a 2D convnet as a 3D conbnet
    '''
    def __init__(self, resnet):
        super(From3D, self).__init__()
        self.model = resnet
    
    def forward(self, x):
        N, C, T, h, w = x.shape
        
        xx = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, h, w)
        mm = self.model(xx)

        return mm.view(N, T, *mm.shape[-3:]).permute(0, 2, 1, 3, 4)


def resnet_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x if self.maxpool is None else self.maxpool(x) 

        x = self.layer1(x)
        x = self.layer2(x)

        x = x if self.layer3 is None else self.layer3(x) 
        x = x if self.layer4 is None else self.layer4(x) 
    
        return x        

def adapt_resnet(net, remove_layers=[]):
    from types import MethodType
        
    
    
    filter_layers = lambda x: [l for l in x if getattr(net, l) is not None]
    for layer in filter_layers(['layer3', 'layer4']):
        for m in getattr(net, layer).modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv3d):
                m.stride = tuple(1 for ss in m.stride)

    remove_layers += ['fc', 'avgpool']
    for layer in filter_layers(remove_layers):
        setattr(net, layer, None)

    # net.forward = MethodType(resnet_forward, net)

    return net


import torchvision.models.resnet as official_resnet
from models import resnet2d, antialiased
def make_encoder(args):

    model_type = args.model_type

    if model_type == 'scratch':
        net = resnet2d.resnet18(pretrained=False)#, norm_layer=norm_layer)

    elif model_type == 'scratchvq':
        net = resnet2d.resnet18vq(pretrained=False)#, norm_layer=norm_layer)

    elif model_type == 'official':
        net = official_resnet.resnet18(pretrained=False)

    elif 'vince_weights' in model_type:
        checkpoint = load_vince_model(model_type)
        resnet2d._REFLECT_PAD = False
        net = resnet2d.resnet18(pretrained=False)
        net.load_state_dict(checkpoint)

    elif model_type == 'aaresnet':
        net = aa_resnet.resnet18(pretrained=False)
        
    elif model_type == 'bagnet':
        import bagnet
        net = bagnet.bagnet33(pretrained=True)
        
    elif model_type == 'imagenet':
        resnet2d._REFLECT_PAD = False
        net = resnet2d.resnet18(pretrained=True)

    else: 
        assert False, 'invalid args.model_type'

    net = adapt_resnet(net,
            remove_layers=[] if args.use_res4 else ['layer4'])

    if 'Conv2d' in str(net):  # HACK, if this is a 2D convnet
        net = From3D(net)

    return net



class MaskedAttention(nn.Module):
    '''
    A module that implements masked attention based on spatial locality 
    '''
    def __init__(self, radius, flat=True):
        super(MaskedAttention, self).__init__()
        self.radius = radius
        self.flat = flat
        self.masks = {}
        self.index = {}


    def mask(self, H, W):
        if not ('%s-%s' %(H,W) in self.masks):
            self.make(H, W)
        return self.masks['%s-%s' %(H,W)]

    def index(self, H, W):
        if not ('%s-%s' %(H,W) in self.index):
            self.make_index(H, W)
        return self.index['%s-%s' %(H,W)]

    def make(self, H, W):
        if self.flat:
            H = int(H**0.5)
            W = int(W**0.5)
        
        gx, gy = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        D = ( (gx[None, None, :, :] - gx[:, :, None, None])**2 + (gy[None, None, :, :] - gy[:, :, None, None])**2 ).float() ** 0.5
        D = (D < self.radius)[None].float()

        if self.flat:
            D = self.flatten(D)

        self.masks['%s-%s' %(H,W)] = D
        # self.index['%s-%s' %(H,W)] = self.masks['10-10']

        return D

    def flatten(self, D):
        return torch.flatten(torch.flatten(D, 1, 2), -2, -1)

    def make_index(self, H, W, pad=False):
        mask = self.mask(H, W).view(1, -1).byte()
        idx = torch.arange(0, mask.numel())[mask[0]][None]

        self.index['%s-%s' %(H,W)] = idx

        return idx
        
    def forward(self, x):
        H, W = x.shape[-2:]
        sid = '%s-%s' % (H,W)
        if sid not in self.masks:
            self.masks[sid] = self.make(H, W).to(x.device)
        mask = self.masks[sid]

        return x * mask[0]


def log_sinkhorn_knopp(log_alpha, tol=0.01, n_iters=20, verbose=False):
    m,n  = log_alpha.size()[-2:]
    log_alpha = log_alpha.view(-1, m, n)

    for i in range(n_iters):
        # torch.logsumexp(input, dim, keepdim, out=None)
        #Returns the log of summed exponentials of each row of the input tensor in the given dimension dim
        #log_alpha -= (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, n, 1)
        #log_alpha -= (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, n)
        #avoid in-place
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, m, 1)
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, n)

    return torch.exp(log_alpha)

def sinkhorn_knopp(A, tol=0.01, max_iter=1000, verbose=False):
    _iter = 0
    
    if A.ndim > 2:
        A = A / A.sum(-1).sum(-1)[:, None, None]
    else:
        A = A / A.sum(-1).sum(-1)[None, None]

    A1 = A2 = A 

    while (A2.sum(-2).std() > tol and _iter < max_iter) or _iter == 0:
        A1 = F.normalize(A2, p=1, dim=-2)
        A2 = F.normalize(A1, p=1, dim=-1)

        _iter += 1
        if verbose:
            print(A2.max(), A2.min())
            print('row/col sums', A2.sum(-1).std().item(), A2.sum(-2).std().item())

    if verbose:
        print('------------row/col sums aft', A2.sum(-1).std().item(), A2.sum(-2).std().item())

    return A2 