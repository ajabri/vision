
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

from spatial_correlation_sampler import spatial_correlation_sample

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
parser.add_argument('--radius', default=3, type=float,
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
parser.add_argument('--head-depth', default=0, type=int, help='')

parser.add_argument('--no-maxpool', default=False, action='store_true', help='')
parser.add_argument('--use-res4', default=False, action='store_true', help='')

parser.add_argument('--long-mem', default=[0], type=int, nargs='+', help='')

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
            

def softmax_base(A):
    # import pdb; pdb.set_trace()
    if not args.all_nn:
        N, T, H1, W1, H, W = A.shape
        A = A.view(N, T, H1*W1, H, W)
        A = torch.nn.functional.softmax(A, dim=-3)
    else:
        N, T, H1, W1, H, W = A.shape
        A = A.view(N, T*H1*W1, H, W)
        A = torch.nn.functional.softmax(A, dim=-3)
    return A

def hard_prop(predlbls):
    # pred_max = predlbls.max(axis=0)[0]
    # predlbls[predlbls <  pred_max] = 0
    # predlbls[predlbls >= pred_max] = 1
    # predlbls /= predlbls.sum(0)[None]
    return predlbls
    # import pdb; pdb.set_trace()

def test(val_loader, model, epoch, use_cuda):

    save_path = args.save_path + '/'
    save_file = '%s/list.txt' % save_path
    fileout = open(save_file, 'w')

    end = time.time()
    
    job_args = []

    n_context = params['videoLen']
    topk_vis = args.topk_vis
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
        bsize = 5
        for b in range(0, imgs_total.shape[1], bsize):
            torch.cuda.empty_cache()
            node, feat = model.module(imgs_total[:, b:b+bsize], None, True, func='forward')
            feats.append(feat.cpu())
        feats = torch.cat(feats, dim=2)

        feats = feats.detach().squeeze(1)
        feats = torch.nn.functional.normalize(feats, dim=1)

        print('computed features', time.time()-t00)

        ##################################################################
        # Prep labels
        ##################################################################        

        for t in range(n_context):
            imname  = save_path + str(batch_idx) + '_' + str(t) + '_label.jpg'
            imageio.imwrite(imname, lbls_tensor[0][t].numpy().astype(np.uint8))
        # print('wrote frames and labels')
        
        ##################################################################
        # Compute correlation features
        ##################################################################
        
        imgs_stack = []
        im_num = total_frame_num - n_context
        t03 = time.time()

        def make_bank():
            ll = []
            
            for t in args.long_mem:
                idx = torch.zeros(im_num, 1).long()
                if t > 0:
                    assert t < im_num
                    idx += t + (args.videoLen+1)
                    idx[:args.videoLen+t+1] = 0

                ll.append(idx)
                # idx[:]


            ss = [(torch.arange(n_context)[None].repeat(im_num, 1) + 
                    torch.arange(im_num)[:, None])[:, 1:]
            ]

            return ll + ss
                
        indices = torch.cat(make_bank(), dim=-1)
        feats = feats.cpu()

        if isinstance(lbl_set, list):
            lbl_set = torch.cat(lbl_set)[None]
        lbls_resize[0, n_context*2 - 1:] *= 0
        
        # H x W x L -> L x H x W
        lbls_resize = lbls_resize.transpose(-1,-3).transpose(-1,-2)

        As, Ws, Is = [], [], []
        Al, Wl, Il = [], [], []

        keys, query = feats[:, :, indices], feats[:, :, n_context:]

        _, C, N, T, H, W = keys.shape
        # for ease with spatial_correlation_sampler
        keys = keys.permute(0, 2, 3, 1, 4, 5)        # reshape to 1 x N x T X C X H X W
        query = query.permute(0, 2, 1, 3, 4).unsqueeze(2).expand_as(keys)

        ###########################
        # import pdb; pdb.set_trace()

        ## For long term mem (no radius)
        restrict = utils.RestrictAttention(args.radius, flat=False)
        H, W = query.shape[-2:]
        D = restrict.mask(H, W)[None].cuda()
        D[D==0] = -1e10
        D[D==1] = 0

    
        ###########################

        q_dim = 2 if args.all_nn else 3
        bsize = 2

        ###########################

        
        def restricted(k, q):
            _, N, T, C, H, W = k.shape

            A = spatial_correlation_sample(
                    k.view(k.shape[1]*T, C, H, W),
                    q.view(q.shape[1]*T, C, H, W),
                    patch_size=int(2*args.radius+1))

            A = A.view(1, _q.shape[1], T, *A.shape[-4:]) /args.temperature
            # A[A==0] = -1e20  # ignored idxs in softmax

            _, N, T, H1, W1, H, W = A.shape
            A = A.view(N, T*H1*W1, H, W)
            weights, ids = torch.topk(A, topk_vis, dim=-3)
            weights = torch.nn.functional.softmax(weights, dim=-3)

            return A, weights, ids

        def full(k, q):
            A2 = torch.einsum('ikljmn,ikjop->iklmnop', k, q) / args.temperature
            _, N, T, H1, W1, H, W = A2.shape
            A2 = A2.view(N, T*H1*W1, H, W)
            weights2, ids2 = torch.topk(A2, topk_vis, dim=-3)
            weights2 = torch.nn.functional.softmax(weights2, dim=-3)

            return A2, weights2, ids2

        ###########################

        for b in range(0, keys.shape[1], bsize):
            _k, _q = keys[:, b:b+bsize], query[:, b:b+bsize, len(args.long_mem):].cuda()

            k_l = _k[:, :, :len(args.long_mem)].cuda()
            k_s = _k[:, :, len(args.long_mem):].cuda()


            A1, w1, i1 = restricted(k_s, _q)
            A2, w2, i2 = full(k_l, _q[:, :, 0])

            A12 = torch.cat([
                torch.gather(A1, 1, i1),
                torch.gather(A2, 1, i2),
            ], dim=1)
            weights, ids = torch.topk(A12, topk_vis, dim=-3)



            import pdb; pdb.set_trace()
            
            Ws += [w  for w  in w1.cpu()]
            Is += [ii for ii in i1.cpu()]

            Wl += [w  for w  in w2.cpu()]
            Il += [ii for ii in i2.cpu()]


            # As += [a for a in A.cpu()[0]]

        # As, Ws, Is = (torch.cat(_, dim=1) for _ in (As, Ws, Is))


        t04 = time.time()
        print(t04-t03, 'computed affinities', torch.cuda.max_memory_allocated()/(1024**2))

        # As, Ws, Is = As[0], Ws[0], Is[0]
        lbl_set, lbls_resize = lbl_set[0], lbls_resize[0]
        
        ##################################################################
        # Label propagation
        ##################################################################

        L, H, W = lbls_resize.shape[1:]
        lbls_idx = torch.arange(T*H*W).view(T, H, W)
        lbls_idx = F.pad(lbls_idx, [int(args.radius)]*4, 'constant', -1)
        lbls_idx = F.unfold(lbls_idx[None].float(), kernel_size=int(args.radius*2+1)).view(1, -1, H, W).long().cuda()

        nstep = len(imgs_toprint) - n_context

        for it in range(nstep):
            if it % 10 == 0:
                print(it, torch.cuda.max_memory_allocated()/(1024**2))

            weights, idxs = Ws[it].cuda(), Is[it].cuda()
            lbls_base = lbls_resize[indices[it]].cuda()
            t06 = time.time()

            lamda = (len(args.long_mem) / (args.videoLen + len(args.long_mem)))

            # Short term (indexing based)
            flat_lbls = lbls_base.transpose(0, 1).flatten(1)
            global_idxs = torch.gather(lbls_idx, 1, idxs[None])
            nn_lbls = flat_lbls[:, global_idxs.view(topk_vis, -1).t()].transpose(-1,-2)
            predlbls = (nn_lbls.view(L, topk_vis, H, W) * weights[None]).sum(1)

            # Long term
            predlbls = lamda * (flat_lbls[:, Il[it]] * Wl[it].cuda()[None]).sum(1)  + (1-lamda) * predlbls

            # predlbls /= 2

            # hard prop
            # predlbls = hard_prop(predlbls)

            img_now = imgs_toprint[it + n_context].permute(1,2,0).numpy() * 255
            
            if it > 0:
                lbls_resize[it + n_context] = predlbls
            else:
                predlbls = lbls_resize[0]


            # Save Predictions
            dump_predictions(
                predlbls.cpu().permute(1, 2, 0).numpy(),
                lbl_set, img_now, save_path + str(batch_idx) + '_' + str(it))

        torch.cuda.empty_cache()

        print('******* Vid %s TOOK %s *******' % (batch_idx, time.time() - t_vid))


if __name__ == '__main__':
    main()
