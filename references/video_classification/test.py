
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
parser.add_argument('--no-maxpool', default=False, action='store_true',
    help='')
parser.add_argument('--use-res4', default=False, action='store_true',
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
    def do_label_prop(A, lbl_set, lbls_resize, imgs_toprint, save_path, batch_idx):        
        ##################################################################
        # Label propagation
        ##################################################################

        nstep = len(imgs_toprint) - n_context
        indices = torch.cat([
            torch.zeros(nstep, 1).long(),
            (torch.arange(n_context)[None].repeat(nstep, 1) + torch.arange(nstep)[:, None])[:, 1:]
            ], dim=-1)

        for it in range(nstep):
            if it % 10 == 0:
                print(it)

            t05 = time.time()
            A_t = A[it].cuda()

            # TODO potentially re-softmax???
            q_dim = 0 if args.all_nn else 1
            weights, ids = torch.topk(A_t, topk_vis, dim=q_dim)

            # import pdb; pdb.set_trace()
            # weights = torch.nn.functional.softmax(weights, dim=1)
            weights = torch.nn.functional.normalize(weights, dim=q_dim, p=1)

            t06 = time.time()
            lbls_base = lbls_resize[indices[it]]
            T, H, W, L = lbls_base.shape

            if q_dim == 0:
                lbls_base = lbls_base.view(T*H*W, L).cuda()
                predlbls = batched_index_select(
                    lbls_base, 0, ids.view(-1))
                predlbls = (weights.unsqueeze(-1) * predlbls.view(topk_vis, H, W, L)).sum(0)
            else:
                lbls_base = lbls_base.view(T, H*W, L).cuda()
                predlbls = batched_index_select(
                    lbls_base, 1, ids.view(T, -1))
                predlbls = (weights.unsqueeze(-1)/T * predlbls.view(T, topk_vis, H, W, L)).sum(0).sum(0)

            img_now = imgs_toprint[it + n_context].permute(1,2,0).numpy() * 255
            
            # print(time.time()-t06, 'lbl proc', t06-t05, 'argsorts')

            # normalize across pixels?? labels distribution...
            # import pdb; pdb.set_trace()
            # predlbls -= predlbls.min(0)[0].min(0)[0][None, None]
            # predlbls /= predlbls.max(0)[0].max(0)[0][None, None]
            # import pdb; pdb.set_trace()

            
            if it > 0:
                lbls_resize[it + n_context] = predlbls
            else:
                predlbls = lbls_resize[0]

            predlbls_cp = predlbls.cpu().numpy().copy()
            predlbls_cp = cv2.resize(predlbls_cp, (params['imgSize'], params['imgSize']))
            predlbls_val = np.zeros((params['imgSize'], params['imgSize'], 3))

            ids = np.argmax(predlbls_cp[:, :, 1 : len(lbl_set)], 2)

            predlbls_val = np.array(lbl_set)[np.argmax(predlbls_cp, axis=-1)]
            predlbls_val = predlbls_val.astype(np.uint8)

            # if img_now.shape[0] != args.outSize:
            #     img_now = cv2.resize(img_now, (args.outSize, args.outSize), interpolation=cv2.INTER_LINEAR)

            predlbls_val2 = cv2.resize(predlbls_val, (img_now.shape[0], img_now.shape[1]), interpolation=cv2.INTER_NEAREST)

            # activation_heatmap = cv2.applyColorMap(predlbls, cv2.COLORMAP_JET)
            img_with_heatmap =  np.float32(img_now) * 0.5 + np.float32(predlbls_val2) * 0.5

            imname  = save_path + str(batch_idx) + '_' + str(it) + '_label.jpg'
            imname2  = save_path + str(batch_idx) + '_' + str(it) + '_mask.png'

            imageio.imwrite(imname, np.uint8(img_with_heatmap))
            imageio.imwrite(imname2, np.uint8(predlbls_val))

            if args.visdom:
                vis.image(np.uint8(img_with_heatmap).transpose(2, 0, 1))
                vis.image(np.uint8(predlbls_val).transpose(2, 0, 1))


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
        nodes = []
        bsize = 50
        # for b in range(0, imgs_total.shape[1], bsize):
        #     node, feat = model.module(imgs_total[b:b+bsize], None, True, func='forward')
        #     feats.append(feat); nodes.append(node)
        # feats = torch.cat(feats, dim=1)
        
        nodes, feats = model.module(imgs_total, None, True, func='forward')
        # feats = 
        feats = feats.detach().squeeze(1)
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
        
        imgs_stack = []
        im_num = total_frame_num - n_context
        t03 = time.time()

        indices = torch.cat([
            torch.zeros(im_num, 1).long(),
            (torch.arange(n_context)[None].repeat(im_num, 1) + 
                torch.arange(im_num)[:, None])[:, 1:]],
                dim=-1)

        feats = feats.cpu()
        keys, query = feats[:, :, indices], feats[:, :, n_context:]

        H, W = query.shape[-2:]
        gx, gy = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        D = ( (gx[None, None, :, :] - gx[:, :, None, None])**2 + (gy[None, None, :, :] - gy[:, :, None, None])**2 ).float() ** 0.5
        D = (D < args.radius)[None, None].float().cuda()
        D[D==0] = 1e-10
        # import pdb; pdb.set_trace()

        As = []
        bsize = 3
        for b in range(0, keys.shape[2], bsize):
            A = torch.einsum('ijklmn,ijkop->iklmnop', keys[:, :, b:b+bsize].cuda(), query[:, :, b:b+bsize].cuda()) / args.temperature
            A[0, :, 1:] *= D

            A = softmax_base(A[0])[None]

            As.append(A.cpu())

        A = torch.cat(As, dim=1) 
        t04 = time.time()
        print(t04-t03, 'model forward')


        if isinstance(lbl_set, list):
            lbl_set = torch.cat(lbl_set)[None]
        
        # import pdb; pdb.set_trace()
        lbls_resize[0, n_context*2 - 1:] *= 0
        do_label_prop(A[0], lbl_set[0], lbls_resize[0], imgs_toprint, save_path, batch_idx)

        print('******* Vid %s TOOK %s *******' % (batch_idx, time.time() - t_vid))


    converted_path = "%s_converted/" % args.save_path[:-1]
    data_path = os.path.dirname(args.filelist)

    cmd = """python davis/convert_davis.py --in_folder %s --out_folder %s --dataset %s/ && \
    python %s-2017/python/tools/eval.py -i %s -o %s/results.yaml --year 2017 --phase val \
    | tee %s/output.txt & """ % (args.save_path, converted_path, data_path, data_path, converted_path, converted_path, converted_path)

    # import pdb; pdb.set_trace()

# python davis/convert_davis.py --in_folder results/ --out_folder results_converted/ --dataset /scratch/ajabri/data/davis/ &&     python /scratch/ajabri/data/davis-2017/python/tools/eval.py -i results_converted/ -o results_converted//results.yaml --year 2017 --phase val     | tee results_converted//output.txt &


'python davis/convert_davis.py --in_folder /scratch/ajabri/logs/timecycle/results_123456/ --out_folder /scratch/ajabri/logs/timecycle/results_123456_converted/ --dataset /home/ajabri/data/davis/ &&     python /home/ajabri/data/davis-2017/python/tools/eval.py -i /scratch/ajabri/logs/timecycle/results_123456_converted/ -o /scratch/ajabri/logs/timecycle/results_123456_converted//results.yaml --year 2017 --phase val     | tee /scratch/ajabri/logs/timecycle/results_123456_converted//output.txt & '
if __name__ == '__main__':
    main()
