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
import skimage
import skimage.io


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data

import timecycle as tc
import davis_test as davis
import jhmdb_test as jhmdb

from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from torch.autograd import Variable


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
parser.add_argument('--head-depth', default=0, type=int,
                    help='')

parser.add_argument('--no-maxpool', default=False, action='store_true', help='')
parser.add_argument('--use-res4', default=False, action='store_true', help='')
parser.add_argument('--no-l2', default=False, action='store_true', help='')

parser.add_argument('--long-mem', default=[0], type=int, nargs='+', help='')
parser.add_argument('--texture', default=False, action='store_true', help='')
parser.add_argument('--round', default=False, action='store_true', help='')

parser.add_argument('--time-dilation', default=1, type=int, help='time dilation of context')
parser.add_argument('--mapRatio', default=1, type=int, help='map aspect ratio')

parser.add_argument('--norm_mask', default=False, action='store_true', help='')

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
vis = None
if args.visdom:
    vis = visdom.Visdom(server=args.server, port=8095, env='main_davis_viz1'); vis.close()
    import wandb
    wandb.init(project='palindromes')
    vis.close()

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
        args,
        vis=vis
    ).cuda()
    # model = Wrap(model)
    
    params['mapScale'] = model(torch.zeros(1, 10, 3, 320, 320).cuda(), just_feats=True)[1].shape[-2:]
    params['mapScale'] = 320 // np.array(params['mapScale'])

    val_loader = torch.utils.data.DataLoader(
        davis.DavisSet(params, is_train=False) if not 'jhmdb' in args.filelist  else \
            jhmdb.JhmdbSet(params, is_train=False),
        batch_size=int(params['batchSize']), shuffle=False,
        num_workers=args.workers, pin_memory=True)


    cudnn.benchmark = False
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Load checkpoint.
    if os.path.isfile(args.resume):
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume)
        utils.partial_load(checkpoint['model'], model, skip_keys=['head'])

        del checkpoint
    
    model.eval()
    # model = torch.nn.DataParallel(model).cuda()    #     model = model.cuda()
    model = model.cuda()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    print('\Testing')
    # with torch.no_grad():
    test_loss = test(val_loader, model, 1, use_cuda)
            

def dump_predictions(predlbls, lbl_set, img_now, prefix):
    sz = img_now.shape[:-1]

    pad = params['mapScale']
    predlbls_cp = predlbls.copy()

    predlbls_cp = cv2.resize(predlbls_cp, sz[::-1])[:]
    
    # predlbls_cp2 = np.zeros((sz[-2]+pad[0], sz[-1]+pad[1], predlbls_cp.shape[-1]))
    # predlbls_cp2[:, :, 0] = 1
    # predlbls_cp2[:-pad[0], :-pad[1], :] = predlbls_cp
    # predlbls_cp = predlbls_cp2
    # predlbls_cp = cv2.resize(predlbls_cp, (sz[-1] + pad[0], sz[-2] + pad[1]))[:]
    # predlbls_cp =predlbls_cp[pad[0]//2:-pad[0]//2, pad[1]//2:-pad[1]//2]
    # predlbls_cp =predlbls_cp[pad[0]:, pad[1]:]
    
    predlbls_val = np.zeros((*sz, 3))

    ids = np.argmax(predlbls_cp[:, :, 1 : len(lbl_set)], 2)
    
    predlbls_val = np.argmax(predlbls_cp, axis=-1)
    predlbls_val = np.array(lbl_set, dtype=np.int32)[predlbls_val]        
    # predlbls_val = predlbls_val.astype(np.uint8)

    # if img_now.shape[0] != args.outSize:
    #     img_now = cv2.resize(img_now, (args.outSize, args.outSize), interpolation=cv2.INTER_LINEAR)

    # import pdb; pdb.set_trace()
    predlbls_val2 = cv2.resize(predlbls_val, (img_now.shape[1], img_now.shape[0]), interpolation=cv2.INTER_NEAREST)

    # activation_heatmap = cv2.applyColorMap(predlbls, cv2.COLORMAP_JET)
    img_with_heatmap =  np.float32(img_now) * 0.5 + np.float32(predlbls_val2) * 0.5

    
    # imname  = prefix + '_blend.jpg'
    # imageio.imwrite(imname, np.uint8(img_with_heatmap))

    if prefix[-4] != '.':
        imname2 = prefix + '_mask.png'
        # skimage.io.imsave(imname2, np.uint8(predlbls_val))
    else:
        imname2 = prefix.replace('jpg','png')
        
        # predlbls_val = np.uint8(predlbls_val)

        # if predlbls_val.max() > 20:#: or :
        #     import pdb; pdb.set_trace()
    
        # skimage.io.imsave(imname2.replace('jpg','png'), predlbls_val)

    imageio.imwrite(imname2, np.uint8(predlbls_val))

    return img_with_heatmap, predlbls_val


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

def hard_prop(predlbls):
    pred_max = predlbls.max(axis=0)[0]
    predlbls[predlbls <  pred_max] = 0
    predlbls[predlbls >= pred_max] = 1
    predlbls /= predlbls.sum(0)[None]
    return predlbls

def process_pose(predlbls, lbl_set, topk=3):
    # generate the coordinates:
    predlbls = predlbls[..., 1:]
    flatlbls = predlbls.flatten(0,1)
    topk = min(flatlbls.shape[0], topk)
    
    vals, ids = torch.topk(flatlbls, k=topk, dim=0)
    vals /= vals.sum(0)[None]
    xx, yy = ids % predlbls.shape[1], ids // predlbls.shape[1]

    current_coord = torch.stack([(xx * vals).sum(0), (yy * vals).sum(0)], dim=0)
    current_coord[:, flatlbls.sum(0) == 0] = -1

    predlbls_val_sharp = np.zeros((*predlbls.shape[:2], 3))

    for t in range(len(lbl_set) - 1):
        x = int(current_coord[0, t])
        y = int(current_coord[1, t])

        if x >=0 and y >= 0:
            predlbls_val_sharp[y, x, :] = lbl_set[t + 1]

    return current_coord.cpu(), predlbls_val_sharp

def test(val_loader, model, epoch, use_cuda):

    save_path = args.save_path + '/'

    end = time.time()
    
    n_context = params['videoLen']

    # Radius mask
    D = None
    t_vid = 0

    _model_state = model.state_dict().copy()

    for batch_idx, (imgs_total, imgs_orig, lbl_set, lbls_tensor, lbls_onehot, lbls_resize, meta) in enumerate(val_loader):
        t_vid = time.time()
        print('******* Vid %s *******' % batch_idx)

        # measure data loading time
        imgs_total = imgs_total.cuda()
        bs, total_frame_num, channel_num, height_len, weight_len = imgs_total.shape

        imgs_toprint = [ii for ii in imgs_orig[0]]

        assert(bs == 1)

        folder_paths = meta['folder_path']
        print('total_frame_num: ' + str(total_frame_num))

        ##################################################################
        # Compute image features
        ##################################################################        
        model.load_state_dict(_model_state)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0)

        def fit(model, video, targets, steps=1):
            for _ in range(steps):
                output, xent_loss, kldv_loss, diagnostics = model(video, orig=video, targets=targets)
                loss = (xent_loss.mean() + kldv_loss.mean())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # make labels: uniform prob to indices with same mask id
        targets = lbls_onehot.max(-1)[1]
        targets = (targets[0, 0:1, ..., None, None] == targets[0, 0:1])
        targets = targets*1.0/ targets.sum(-1).sum(-1)[..., None, None]*1.0
        
        b, bsize = 0, 5
        # for b in range(0, imgs_total.shape[1], bsize):
        for iters in range(10):
            video  = imgs_total[:, b:b+bsize]
            fit(model, video, targets)
            import pdb; pdb.set_trace()

        with torch.no_grad():

            t00 = time.time()
            feats = []
            bsize = 5
            for b in range(0, imgs_total.shape[1], bsize):
                node, feat = model(imgs_total[:, b:b+bsize], orig=None, just_feats=True)
                feats.append(feat.cpu())
            feats = torch.cat(feats, dim=2)
            
            # nodes, feats = model.module(imgs_total, None, True, func='forward')
            feats = feats.squeeze(1)
            if not args.no_l2:
                feats = torch.nn.functional.normalize(feats, dim=1)

            print('computed features', time.time()-t00)

            ##################################################################
            # Compute correlation features
            ##################################################################
            
            torch.cuda.empty_cache()

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

                ss = [
                    (torch.arange(n_context)[None].repeat(im_num, 1) + 
                        torch.arange(im_num)[:, None])[:, 1:]
                ]
                # import pdb; pdb.set_trace()
                
                return ll + ss
                    
            indices = torch.cat(make_bank(), dim=-1)
            keys, query = feats[:, :, indices], feats[:, :, n_context:]

            restrict = utils.RestrictAttention(args.radius, flat=False)
            D = restrict.mask(*query.shape[-2:])[None]
            D = D.flatten(-4, -3).flatten(-2)
            D[D==0] = -1e10; D[D==1] = 0

            Ws, Is = [], []
            keys, query = keys.flatten(-2), query.flatten(-2)

            bsize, pbsize = 2, 100 #keys.shape[2] // 2

            for b in range(0, keys.shape[2], bsize):
                _k, _q = keys[:, :, b:b+bsize].cuda(), query[:, :, b:b+bsize].cuda()
                w_s, i_s = [], []

                for pb in range(0, _k.shape[-1], pbsize):
                    # A = torch.einsum('ijklmn,ijkop->iklmnop', _k, _q) / args.temperature
                    A = torch.einsum('ijklm,ijkn->iklmn',
                        _k, _q[..., pb:pb+pbsize]) 
                    
                    A[0, :, len(args.long_mem):] += D[..., pb:pb+pbsize].cuda()

                    _, N, T, h1w1, hw = A.shape
                    A = A.view(N, T*h1w1, hw)
                    A /= args.temperature

                    weights, ids = torch.topk(A, args.topk_vis, dim=-2)
                    weights = torch.nn.functional.softmax(weights, dim=-2)
                    
                    w_s.append(weights.cpu())
                    i_s.append(ids.cpu())

                # import pdb; pdb.set_trace()

                weights = torch.cat(w_s, dim=-1)
                ids = torch.cat(i_s, dim=-1)
                Ws += [w for w in weights]
                Is += [ii for ii in ids]

            # Ws, Is = torch.cat(Ws, 1), torch.cat(Is, 1)         # A = torch.cat(As, dim=1) 

            t04 = time.time()
            print(t04-t03, 'affinity forward, max mem', torch.cuda.max_memory_allocated() / (1024**2))

            if isinstance(lbl_set, list):
                lbl_set = torch.cat(lbl_set)[None]
            lbls_resize[0, n_context*2 - 1:] *= 0
            lbl_set, lbls_resize = lbl_set[0], lbls_resize[0]

                
            ##################################################################
            # Label propagation
            ##################################################################

            maps = []
            keypts = []
            images = []
            for it in range(indices.shape[0]):
                if it % 10 == 0:
                    print(it)

                lbls_base = lbls_resize[indices[it]].cuda()
                flat_lbls = lbls_base.flatten(0, 2).transpose(0, 1)

                predlbls = (flat_lbls[:, Is[it]] * Ws[it].cuda()[None]).sum(1)
                predlbls = predlbls.view(-1, *feats.shape[-2:])

                # print(predlbls.mean(-1).mean(-1))
                #predlbls = hard_prop(predlbls)

                predlbls = predlbls.permute(1,2,0)

                img_now = imgs_toprint[it + n_context].permute(1,2,0).numpy() * 255
                            
                if it > 0:
                    lbls_resize[it + n_context] = predlbls
                else:
                    predlbls = lbls_resize[0]

                if args.norm_mask:
                    # import pdb; pdb.set_trace()
                    predlbls[:, :, :] -= predlbls.min(-1)[0][:, :, None]
                    predlbls[:, :, :] /= predlbls.max(-1)[0][:, :, None]

                _maps = []

                if 'jhmdb' in args.filelist.lower():
                    coords, predlbls_sharp = process_pose(predlbls, lbl_set)
                    keypts.append(coords)
                    pose_map = utils.vis_pose(np.array(img_now).copy(), coords.numpy() * params['mapScale'][..., None])
                    _maps += [pose_map]


                # Save Predictions            
                if 'VIP' in args.filelist:
                    outpath = os.path.join(save_path, 'videos'+meta['img_paths'][it+n_context][0].split('videos')[-1])
                    os.makedirs(os.path.dirname(outpath), exist_ok=True)
                else:
                    outpath = os.path.join(save_path, str(batch_idx) + '_' + str(it))

                heatmap, lblmap = dump_predictions(
                    predlbls.cpu().numpy(),
                    lbl_set, img_now, outpath)

                _maps += [heatmap, lblmap]
                maps.append(_maps)
                images.append(img_now)

                if args.visdom:
                    [vis.image(np.uint8(_m).transpose(2, 0, 1)) for _m in _maps]

            if len(keypts) > 0:
                # import pdb; pdb.set_trace()
                coordpath = os.path.join(save_path, str(batch_idx) + '.dat')
                np.stack(keypts, axis=-1).dump(coordpath)

            if args.visdom:
                # wandb.log({'vid%s' % batch_idx: [wandb.Image(mm[0]) for mm in maps]})  
                wandb.log({'vid%s' % batch_idx: wandb.Video(
                    np.array([m[0] for m in maps]).transpose(0, -1, 1, 2), fps=12, format="gif")})  
                
    #            import pdb; pdb.set_trace()
                wandb.log({'plain vid%s' % batch_idx: wandb.Video(
                    np.array(images).transpose(0, -1, 1, 2), fps=12, format="gif")})  
                
            torch.cuda.empty_cache()
            print('******* Vid %s TOOK %s *******' % (batch_idx, time.time() - t_vid))


if __name__ == '__main__':
    main()
