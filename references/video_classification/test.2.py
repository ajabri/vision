
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import cv2
import imageio

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
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='use pre-trained model')
#Device options
parser.add_argument('--gpu-id', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--batchSize', default=1, type=int,
                    help='batchSize')
parser.add_argument('--T', default=1.0, type=float,
                    help='temperature')

parser.add_argument('--topk_vis', default=20, type=int,
                    help='topk_vis')

parser.add_argument('--videoLen', default=4, type=int,
                    help='predict how many frames away')

parser.add_argument('--cropSize', default=320, type=int,
                    help='predict how many frames away')

parser.add_argument('--filelist', default='/scratch/ajabri/data/davis/val2017.txt', type=str)
parser.add_argument('--save-path', default='', type=str)

args = parser.parse_args()
params = {k: v for k, v in args._get_kwargs()}


print('batchSize: ' + str(params['batchSize']) )
print('videoLen: ' + str(params['videoLen']) )
print('cropSize: ' + str(params['cropSize']) )
params['imgSize'] = params['cropSize']


# Use CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
args.gpu_id = os.getenv('CUDA_VISIBLE_DEVICES')
print(args.gpu_id)

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


def partial_load(pretrained_dict, model):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

def process_labels(lbls, height_dim, width_dim, lbls_onehot=None):
    # processing labels
    t00 = time.time()
    
    
    lbls = lbls[0].data.cpu().numpy()
    print(lbls.shape)

    lbls_new = []

    lbl_set = []
    lbl_set.append(np.zeros(3).astype(np.uint8))
    count_lbls = []
    count_lbls.append(0)
    
    lbls_new = [ll.copy() for ll in lbls]
    
    flat_lbls_0 = lbls[0].copy().reshape(-1, lbls.shape[-1]).astype(np.uint8)
    lbl_set = np.unique(flat_lbls_0, axis=0)
    count_lbls = [np.all(flat_lbls_0 == ll, axis=-1).sum() for ll in lbl_set]
    
    print('lbls', time.time() - t00)
    
    # only keep labels that appear ten times?!
    lbl_set_temp = [ll for ii, ll in enumerate(lbl_set) if count_lbls[ii] > 10]
    lbl_set = lbl_set_temp
    print(lbl_set)
    print(count_lbls)
    
    t01 = time.time()
    if lbls_onehot is None:
        lbls_onehot = np.stack([np.stack([np.all(_lbl == ll, axis=-1) for ll in lbl_set], axis=-1) for _lbl in lbls])
    else:
        assert lbls_onehot.shape[-1] == len(lbl_set)
    t02 = time.time()
    
    lbls_resize2 = np.zeros((lbls.shape[0], height_dim, width_dim, len(lbl_set)))
    
    for i in range(lbls.shape[0]):
        lbls_resize2[i] = cv2.resize(np.float32(lbls_onehot[i]), (height_dim, width_dim))
        
    t03 = time.time()
    print(t03 - t02, 'resize', t02 - t01, 'relabel', t01-t00, 'label')
        
    return lbl_set, lbls_resize2, lbls_onehot, lbls_new


def dump_lbls_onehot(lbls_onehot, lbls_resize, meta):
    '''
    to avoid recomputing one-hot version of labels, which is v expensive
    '''
    assert lbls_onehot.shape[0] == len(meta['lbl_paths'])
    
    for i,l in enumerate(meta['lbl_paths']):
        name = '/' + '/'.join(l[0].split('.')[:-1])
        np.save("%s_%s.npy" % (name, 'onehot'), lbls_onehot[i])
        np.save("%s_%s.npy" % (name, 'size%s' % lbls_resize[i].shape[0]), lbls_resize[i])

    import pdb; pdb.set_trace()

    
class Wrap(nn.Module):
    def __init__(self, model):
        super(Wrap, self).__init__()
        self.model = model
        
    def forward(self, *args, func='forward', **kwargs):
        return getattr(self.model, func)(*args, **kwargs)
    
def main():
    global best_loss
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    model = tc.TimeCycle()
    model = Wrap(model)
    
    model = torch.nn.DataParallel(model).cuda()    #     model = model.cuda()

    params['mapSize'] = model.module(torch.zeros(1, 10, 3, args.cropSize, args.cropSize).cuda(), None, True, func='forward')[1].shape[-2:]

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
        partial_load(checkpoint['state_dict'], model)
        del checkpoint
    
    model.eval()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    print('\Testing')
    test_loss = test(val_loader, model, 1, use_cuda)

def do_label_prop(corrfeat2_set, lbl_set, lbls_resize2, imgs_toprint,
                  save_path, batch_idx,
                  finput_num, topk_vis, n_context):
    ##################################################################
    # Label propagation
    ##################################################################
    height_dim, width_dim = corrfeat2_set[0].shape[-2:]
    
    for iter in range(len(imgs_toprint) - n_context):
        if iter % 10 == 0:
            print(iter)

#             imgs = imgs_total[:, iter + 1: iter + n_context, :, :, :]
#             imgs2 = imgs_total[:, 0, :, :, :].unsqueeze(1)
#             imgs = torch.cat((imgs2, imgs), dim=1)

        corrfeat2   = corrfeat2_set[iter]
        corrfeat2   = torch.from_numpy(corrfeat2)

        out_frame_num = int(finput_num)
        height_dim = corrfeat2.size(2)
        width_dim = corrfeat2.size(3)

        corrfeat2 = corrfeat2.view(corrfeat2.size(0), height_dim, width_dim, height_dim, width_dim)
        corrfeat2 = corrfeat2.data.cpu().numpy()

        vis_ids_h = np.zeros((corrfeat2.shape[0], height_dim, width_dim, topk_vis)).astype(np.int)
        vis_ids_w = np.zeros((corrfeat2.shape[0], height_dim, width_dim, topk_vis)).astype(np.int)

        t05 = time.time()

        atten1d  = corrfeat2.reshape(corrfeat2.shape[0], height_dim * width_dim, height_dim, width_dim)
        ids = np.argpartition(atten1d, -topk_vis, axis=1)[:, -topk_vis:]
        # ids = np.argsort(atten1d, axis=1)[:, -topk_vis:]

        hid = ids // width_dim
        wid = ids % width_dim

        vis_ids_h = wid.transpose(0, 2, 3, 1)
        vis_ids_w = hid.transpose(0, 2, 3, 1)

        t06 = time.time()

        img_now = imgs_toprint[iter + n_context]

        predlbls = np.zeros((height_dim, width_dim, len(lbl_set)))
        # predlbls2 = np.zeros((height_dim * width_dim, len(lbl_set)))

        for t in range(finput_num):

            tt1 = time.time()

            h, w, k = np.meshgrid(np.arange(height_dim), np.arange(width_dim), np.arange(topk_vis), indexing='ij')
            h, w = h.flatten(), w.flatten()

            hh, ww = vis_ids_h[t].flatten(), vis_ids_w[t].flatten()

            if t == 0:
                lbl = lbls_resize2[0, hh, ww, :]
            else:
                lbl = lbls_resize2[t + iter, hh, ww, :]

            np.add.at(predlbls, (h, w), lbl * corrfeat2[t, ww, hh, h, w][:, None])

        t07 = time.time()
        # print(t07-t06, 'lbl proc', t06-t05, 'argsorts')

        predlbls = predlbls / finput_num

        for t in range(len(lbl_set)):
            nowt = t
            predlbls[:, :, nowt] = predlbls[:, :, nowt] - predlbls[:, :, nowt].min()
            predlbls[:, :, nowt] = predlbls[:, :, nowt] / predlbls[:, :, nowt].max()


        lbls_resize2[iter + n_context] = predlbls

        predlbls_cp = predlbls.copy()
        predlbls_cp = cv2.resize(predlbls_cp, (params['imgSize'], params['imgSize']))
        predlbls_val = np.zeros((params['imgSize'], params['imgSize'], 3))

        ids = np.argmax(predlbls_cp[:, :, 1 : len(lbl_set)], 2)

        predlbls_val = np.array(lbl_set)[np.argmax(predlbls_cp, axis=-1)]
        predlbls_val = predlbls_val.astype(np.uint8)
        predlbls_val2 = cv2.resize(predlbls_val, (img_now.shape[0], img_now.shape[1]), interpolation=cv2.INTER_NEAREST)

        # activation_heatmap = cv2.applyColorMap(predlbls, cv2.COLORMAP_JET)
        img_with_heatmap =  np.float32(img_now) * 0.5 + np.float32(predlbls_val2) * 0.5

        imname  = save_path + str(batch_idx) + '_' + str(iter + n_context) + '_label.jpg'
        imname2  = save_path + str(batch_idx) + '_' + str(iter + n_context) + '_mask.png'

        imageio.imwrite(imname, np.uint8(img_with_heatmap))
        imageio.imwrite(imname2, np.uint8(predlbls_val))
            

def test(val_loader, model, epoch, use_cuda):

    save_path = args.save_path + '/'
    save_file = '%s/list.txt' % save_path
    fileout = open(save_file, 'w')

    end = time.time()
    
    job_args = []
    print('Beginning ')

    for batch_idx, (imgs_total, imgs_orig, lbls, lbls_onehot, meta) in enumerate(val_loader):
        
        print('******* Vid %s *******', batch_idx)

        if batch_idx > 2: 
            break
            
        n_context = params['videoLen']
        finput_num = n_context

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
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        imgs_toprint = []

        # ref image
        t00 = time.time()
        # for t in range(imgs_set.shape[0]):
        #     img_now = imgs_set[t]

        #     for c in range(3):
        #         img_now[c] = img_now[c] * std[c]
        #         img_now[c] = img_now[c] + mean[c]

        #     img_now = img_now * 255
        #     img_now = np.transpose(img_now, (1, 2, 0))
        #     img_now = cv2.resize(img_now, (img_now.shape[0] * 2, img_now.shape[1] * 2) )
                        
        #     imgs_toprint.append(img_now)

#             imname  = save_path + str(batch_idx) + '_' + str(t) + '_frame.jpg'
#             imageio.imwrite(imname, img_now.astype(np.uint8))
#             imageio.imwrite(img_now, imname)
    
        print('printed images', time.time()-t00)

        ##################################################################
        # Compute image features
        ##################################################################        
        
        t00 = time.time()
        feats = []
        nodes, feats = model.module(imgs_total, None, True, func='forward')
        feats = feats.cpu().detach()

        print('computed features', time.time()-t00)

        ##################################################################
        # Prep labels
        ##################################################################        

                
        height_dim = feats.shape[-2]
        width_dim  = feats.shape[-1]

        if lbls_onehot.sum() > 0:
            lbl_set, lbls_resize2, lbls_onehot, lbls_new = process_labels(
                lbls, height_dim, width_dim, lbls_onehot=lbls_onehot[0])
        else:
            lbl_set, lbls_resize2, lbls_onehot, lbls_new = process_labels(
                lbls, height_dim, width_dim, lbls_onehot=None)
            dump_lbls_onehot(lbls_onehot, meta)
        
        import pdb; pdb.set_trace()
            
        for t in range(n_context):
            nowlbl = lbls_new[t]
            imname  = save_path + str(batch_idx) + '_' + str(t) + '_label.jpg'
            imageio.imwrite(imname, nowlbl.astype(np.uint8))
#             imageio.imwrite(nowlbl, imname)
        print('wrote frames and labels')
        
        
        ##################################################################
        # Compute correlation features
        ##################################################################
        
        now_batch_size = 4
        imgs_stack = []

        im_num = total_frame_num - n_context
        corrfeat2_set = []
                        
        feats_tensor = torch.Tensor(now_batch_size, feats[0].shape[0], finput_num, *feats.shape[-2:]).cuda()
        feats_targ_tensor = torch.Tensor(now_batch_size, feats[0].shape[0], 1, *feats.shape[-2:]).cuda()
        
        t03 = time.time()
        
        for iter in range(0, im_num, now_batch_size):

            print(iter)

            startid = iter
            endid   = iter + now_batch_size

            if endid > im_num:
                endid = im_num

            now_batch_size2 = endid - startid

            for i in range(now_batch_size2):

                _feats = feats[:, :, iter + i + 1: iter + i + n_context, :, :]
                _feats2 = feats[:, :, 0:1, :, :]

                feats_tensor[i] = torch.cat((_feats2, _feats), dim=2)
                feats_targ_tensor[i, :, 0] = feats[0, :, iter + i + n_context]

            corrfeat2_now = model(feats_tensor, feats_targ_tensor, func='forward_affinity')
            corrfeat2_now = corrfeat2_now.view(now_batch_size, n_context, -1, corrfeat2_now.size(-2), corrfeat2_now.size(-1))

            for i in range(now_batch_size2):
                corrfeat2_set.append(corrfeat2_now[i].data.cpu().numpy())

        t04 = time.time()
        print(t04-t03, 'model forward')
        
        import copy
        job_args.append([
            copy.deepcopy(corrfeat2_set),  copy.deepcopy(lbl_set), copy.deepcopy(lbls_resize2),
            imgs_toprint, save_path, batch_idx,
                finput_num, args.topk_vis, total_frame_num, n_context])
    
    
    import multiprocessing as mp

    pool = mp.Pool(processes=5)
    for jargs in job_args:
        pool.apply_async(do_label_prop, args=jargs)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
