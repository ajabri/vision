
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import cv2
import imageio
import utils

import numpy as np

import pickle
import scipy.misc

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data

import timecycle as tc
import davis_test as davis

from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from torch.autograd import Variable

import torch.nn.functional as F

params = {}


def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--batchSize', default=1, type=int,
                    help='batchSize')
parser.add_argument('--temperature', default=1.0, type=float,
                    help='temperature')

parser.add_argument('--topk_vis', default=20, type=int,
                    help='topk_vis')
parser.add_argument('--radius', default=10, type=int,
                    help='topk_vis')
parser.add_argument('--all-nn', default=False, action='store_true',
                    help='use all as nn')
parser.add_argument('--videoLen', default=4, type=int,
                    help='predict how many frames away')

parser.add_argument('--cropSize', default=320, type=int,
                    help='predict how many frames away')
parser.add_argument('--outSize', default=640, type=int,
                    help='size of output mask image')

parser.add_argument('--filelist', default='/scratch/ajabri/data/davis/val2017.txt', type=str)
parser.add_argument('--save-path', default='./results', type=str)

parser.add_argument('--visdom', default=False, action='store_true')
parser.add_argument('--server', default='localhost', type=str)
parser.add_argument('--model-type', default='scratch', type=str)
parser.add_argument('--head-depth', default=0, type=int,
                    help='')
args = parser.parse_args()
params = {k: v for k, v in args._get_kwargs()}


print('batchSize: ' + str(params['batchSize']) )
print('videoLen: ' + str(params['videoLen']) )
print('cropSize: ' + str(params['cropSize']) )
params['imgSize'] = params['cropSize']


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
# args.gpu_id = os.getenv('CUDA_VISIBLE_DEVICES')
print(args.gpu_id)

import visdom
vis = visdom.Visdom(server=args.server, port=8095, env='main'); vis.close()


# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

    
class Wrap(nn.Module):
    def __init__(self, model):
        super(Wrap, self).__init__()
        self.model = model
        
    def forward(self, *args, func='forward', **kwargs):
        return getattr(self.model, func)(*args, **kwargs)
    
def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def main():
    global best_loss
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    model = tc.TimeCycle(
        args
    ).cuda()
    model = Wrap(model)
    
    params['mapScale'] = model(torch.zeros(1, 10, 3, 320, 320).cuda(), None, True, func='forward')[1].shape[-2:]
    params['mapScale'] = 320 // np.array(params['mapScale'])

    val_loader = torch.utils.data.DataLoader(
        davis.DavisSet(params, is_train=False),
        batch_size=int(params['batchSize']), shuffle=False,
        num_workers=args.workers, pin_memory=True)


    cudnn.benchmark = False
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Load checkpoint.
    if os.path.isfile(args.resume):
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume)
        # model.model.load_state_dict(checkpoint['model'])
        utils.partial_load(checkpoint['model'], model.model)

        del checkpoint
    
    model.eval()
    model = torch.nn.DataParallel(model).cuda()    #     model = model.cuda()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    print('\Testing')
    with torch.no_grad():
        test_loss = test(val_loader, model, 1, use_cuda)
            

def dump_predictions(predlbls, lbl_set, img_now, prefix):
    sz = img_now.shape[:-1]

    predlbls_cp = predlbls.copy()
    predlbls_cp = cv2.resize(predlbls_cp, sz)
    predlbls_val = np.zeros((*sz, 3))

    ids = np.argmax(predlbls_cp[:, :, 1 : len(lbl_set)], 2)

    predlbls_val = np.array(lbl_set)[np.argmax(predlbls_cp, axis=-1)]
    predlbls_val = predlbls_val.astype(np.uint8)

    # if img_now.shape[0] != args.outSize:
    #     img_now = cv2.resize(img_now, (args.outSize, args.outSize), interpolation=cv2.INTER_LINEAR)

    predlbls_val2 = cv2.resize(predlbls_val, (img_now.shape[1], img_now.shape[0]), interpolation=cv2.INTER_NEAREST)

    # activation_heatmap = cv2.applyColorMap(predlbls, cv2.COLORMAP_JET)
    img_with_heatmap =  np.float32(img_now) * 0.5 + np.float32(predlbls_val2) * 0.5

    imname  = prefix + '_label.jpg'
    imname2  = prefix + '_mask.png'

    imageio.imwrite(imname, np.uint8(img_with_heatmap))
    imageio.imwrite(imname2, np.uint8(predlbls_val))

    if args.visdom:
        vis.image(np.uint8(img_with_heatmap).transpose(2, 0, 1))
        vis.image(np.uint8(predlbls_val).transpose(2, 0, 1))


def softmax_base(A):

    if not args.all_nn:
        N, T, H, W, H, W = A.shape
        A = A.view(N, T, H*W, H, W)
        A = torch.nn.functional.softmax(A, dim=-3)
    else:
        N, T, H, W, H, W = A.shape
        A = A.view(N, T*H*W, H, W)
        A = torch.nn.functional.softmax(A, dim=-3)
    return A

def extract_values(lbls, ids, weights):
    T, H, W, L = lbls.shape
    
    if args.all_nn:
        lbls = lbls.view(T*H*W, L)
        
        predlbls = batched_index_select(
            lbls, 0, ids.view(-1))
        predlbls = (weights.unsqueeze(-1) * \
            predlbls.view(weights.shape[0], H, W, L)).sum(0)
    else:
        lbls = lbls.view(T, H*W, L)
        predlbls = batched_index_select(
            lbls, 1, ids.view(T, -1))
        predlbls = (weights.unsqueeze(-1)/T * \
            predlbls.view(T, weights.shape[0], H, W, L)).sum(0).sum(0)

    return predlbls

def test(val_loader, model, epoch, use_cuda):

    save_path = args.save_path + '/'
    save_file = '%s/list.txt' % save_path
    fileout = open(save_file, 'w')

    end = time.time()
    
    job_args = []

    n_context = params['videoLen']
    topk_vis = args.topk_vis

    # Radius mask
    D = None
    t_vid = 0

    for batch_idx, (imgs_total, imgs_orig, lbl_set, lbls_tensor, lbls_onehot, lbls_resize, meta) in enumerate(val_loader):
        t_vid = time.time()
        print('******* Vid %s *******' % batch_idx)

        # measure data loading time
        imgs_total = imgs_total.cuda()
        bs, total_frame_num, channel_num, height_len, weight_len = imgs_total.shape

        assert(bs == 1)

        folder_paths = meta['folder_path']
        print('total_frame_num: ' + str(total_frame_num))

        ##################################################################
        # Print the images
        ##################################################################
        
        imgs_set = imgs_total.data
        imgs_set = imgs_set.cpu().numpy()
        imgs_set = imgs_set[0]

        imgs_toprint = [ii for ii in imgs_orig[0]]

        # ref image
        t00 = time.time()

        # for t in range(imgs_orig.shape[0]):
        #     img_now = imgs_orig[t]
        #     img_now = np.transpose(img_now, (1, 2, 0))
        #     img_now = cv2.resize(img_now, (img_now.shape[0] * 2, img_now.shape[1] * 2) )
        #     imgs_toprint.append(img_now)

        #     imname  = save_path + str(batch_idx) + '_' + str(t) + '_frame.jpg'
        #     imageio.imwrite(imname, img_now.astype(np.uint8))
    
        # print('printed images', time.time()-t00)

        ##################################################################
        # Compute image features
        ##################################################################        
        
        t00 = time.time()
        feats = []
        bsize = 50
        for b in range(0, imgs_total.shape[1], bsize):
            node, feat = model.module(imgs_total[:, b:b+bsize], None, True, func='forward')
            feats.append(feat.cpu())
        feats = torch.cat(feats, dim=2)
        
        # nodes, feats = model.module(imgs_total, None, True, func='forward')
        feats = feats.squeeze(1)
        feats = torch.nn.functional.normalize(feats, dim=1)

        print('computed features', time.time()-t00)

        ##################################################################
        # Prep labels
        ##################################################################        

        for t in range(n_context):
            nowlbl = lbls_tensor[0][t]
            imname  = save_path + str(batch_idx) + '_' + str(t) + '_label.jpg'
            imageio.imwrite(imname, nowlbl.numpy().astype(np.uint8))
        # print('wrote frames and labels')
        
        ##################################################################
        # Compute correlation features
        ##################################################################
        
        # torch.cuda.empty_cache()

        im_num = total_frame_num - n_context
        t03 = time.time()
        indices = torch.cat([torch.zeros(im_num, 1).long(),
            (torch.arange(n_context)[None].repeat(im_num, 1) +  torch.arange(im_num)[:, None])[:, 1:]],
                dim=-1)

        keys, query = feats[:, :, indices], feats[:, :, n_context:]

        restrict = utils.RestrictAttention(args.radius, flat=False)
        H, W = query.shape[-2:]

        
        D = restrict.mask(H, W)[None].cuda()
        D[D==0] = -1e10
        D[D==1] = 0

        rad = int(args.radius)
        D2 = restrict.mask(H+2*rad, W+2*rad)[..., rad:-rad, rad:-rad]
        D2[D2==0] = -1

        I2 = torch.arange(0, D2.shape[1]*D2.shape[2]*n_context).view(4, -1)[:, :, None, None]
        I2 = (D2.flatten(1, 2) * I2)
        I2 = I2[I2>0].view(1, -1, I2.shape[-2], I2.shape[-1]).long()

        Ws, Is = [], []
        bsize = 1
        for b in range(0, keys.shape[2], bsize):
            # import pdb; pdb.set_trace()
            A = torch.einsum('ijklmn,ijkop->iklmnop',
                keys[:, :, b:b+bsize].cuda(), query[:, :, b:b+bsize].cuda()) / args.temperature
            A[0, :, 1:] += D

            # Extract valid regions from the global affinity matrix
            _A = F.pad(A.permute(0,1,2,-2, -1, -4, -3), [int(args.radius)]*4, 'constant', -1e20).permute(0,1,2,-2, -1, -4, -3)
            _A = torch.gather(A.flatten(-5,-3), 2, I2[None].cuda())

            # import pdb; pdb.set_trace()

            _A = F.softmax(_A, dim=2)

            # import pdb; pdb.set_trace()

            # TODO MASK OUT A before softmax. And keep a double index
            # A = softmax_base(A[0])[None]

            # TODO potentially re-softmax???
            q_dim = 2 if args.all_nn else 3
            weights, ids = torch.topk(_A, topk_vis, dim=q_dim)

            import pdb; pdb.set_trace()

            # weights = torch.nn.functional.softmax(weights, dim=1)
            weights = torch.nn.functional.normalize(weights, dim=q_dim, p=1)

            Ws.append(weights.cpu())
            Is.append(ids.cpu())

        Ws, Is = torch.cat(Ws, 1), torch.cat(Is, 1)
        # A = torch.cat(As, dim=1) 

        t04 = time.time()
        print(t04-t03, 'affinity forward, max mem', torch.cuda.max_memory_allocated() / (1024**2))


        if isinstance(lbl_set, list):
            lbl_set = torch.cat(lbl_set)[None]
        lbls_resize[0, n_context*2 - 1:] *= 0

        Ws, Is = Ws[0], Is[0]
        lbl_set, lbls_resize = lbl_set[0], lbls_resize[0]


        ##################################################################
        # Label propagation
        ##################################################################

        for it in range(indices.shape[0]):
            if it % 10 == 0:
                print(it)

            lbls_base = lbls_resize[indices[it]].cuda()
            predlbls = extract_values(lbls_base, Is[it].cuda(), Ws[it].cuda())
            img_now = imgs_toprint[it + n_context].permute(1,2,0).numpy() * 255
                        
            if it > 0:
                lbls_resize[it + n_context] = predlbls
            else:
                predlbls = lbls_resize[0]

            
            # Save Predictions
            dump_predictions(
                predlbls.cpu().numpy(),
                lbl_set, img_now, save_path + str(batch_idx) + '_' + str(it))

        torch.cuda.empty_cache()
        print('******* Vid %s TOOK %s *******' % (batch_idx, time.time() - t_vid))


if __name__ == '__main__':
    main()
