
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
params['filelist'] = '/home/ajabri/data/davis/val2017.txt'
# params['batchSize'] = 24
params['imgSize'] = 320
params['cropSize'] = 320
params['videoLen'] = 8
params['offset'] = 0
params['sideEdge'] = 80
params['predFrames'] = 1


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

parser.add_argument('--videoLen', default=8, type=int,
                    help='predict how many frames away')

parser.add_argument('--cropSize', default=320, type=int,
                    help='predict how many frames away')


parser.add_argument('--save-path', default='', type=str)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


params['batchSize'] = state['batchSize']
print('batchSize: ' + str(params['batchSize']) )

params['videoLen'] = state['videoLen']
print('videoLen: ' + str(params['videoLen']) )

params['cropSize'] = state['cropSize']
print('cropSize: ' + str(params['cropSize']) )
params['imgSize'] = state['cropSize']


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

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

def process_labels(lbls, height_dim, width_dim):
    # processing labels
    t00 = time.time()
    
    
    lbls = lbls[0].data.cpu().numpy()
    print(lbls.shape)

    lbls_new = []

    lbl_set = []
    lbl_set.append(np.zeros(3).astype(np.uint8))
    count_lbls = []
    count_lbls.append(0)

    for i in range(lbls.shape[0]):
        nowlbl = lbls[i].copy()
        if i == 0:
            for j in range(nowlbl.shape[0]):
                for k in range(nowlbl.shape[1]):

                    pixellbl = nowlbl[j, k, :].astype(np.uint8)

                    flag = 0
                    for t in range(len(lbl_set)):
                        if lbl_set[t][0] == pixellbl[0] and lbl_set[t][1] == pixellbl[1] and lbl_set[t][2] == pixellbl[2]:
                            flag = 1
                            count_lbls[t] = count_lbls[t] + 1
                            break

                    if flag == 0:
                        lbl_set.append(pixellbl)
                        count_lbls.append(0)

        lbls_new.append(nowlbl)
        
    print('lbls', time.time() - t00)

    import pdb; pdb.set_trace()

    lbl_set_temp = []
    for i in range(len(lbl_set)):
        if count_lbls[i] > 10:
            lbl_set_temp.append(lbl_set[i])

    lbl_set = lbl_set_temp
    print(lbl_set)
    print(count_lbls)
    
    import pdb; pdb.set_trace()
    t01 = time.time()

    lbls_resize = np.zeros((lbls.shape[0], lbls.shape[1], lbls.shape[2], len(lbl_set)))
    lbls_resize2 = np.zeros((lbls.shape[0], height_dim, width_dim, len(lbl_set)))


    for i in range(lbls.shape[0]):
        nowlbl = lbls[i].copy()
        for j in range(nowlbl.shape[0]):
            for k in range(nowlbl.shape[1]):

                pixellbl = nowlbl[j, k, :].astype(np.uint8)
                for t in range(len(lbl_set)):
                    if lbl_set[t][0] == pixellbl[0] and lbl_set[t][1] == pixellbl[1] and lbl_set[t][2] == pixellbl[2]:
                        lbls_resize[i, j, k, t] = 1

    for i in range(lbls.shape[0]):
        lbls_resize2[i] = cv2.resize(lbls_resize[i], (height_dim, width_dim))
        
    t02 = time.time()
    print(t02 - t01, 'relabel', t01-t00, 'label')
        
    return lbl_set, lbls_resize2, lbls_new

class Wrap(nn.Module):
    def __init__(self, model, func):
        super(Wrap, self).__init__()
        self.model = model
        self.func = func
        
    def forward(self, x):
        return getattr(self.model, self.func)(x)
    
def main():
    global best_loss
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch


    val_loader = torch.utils.data.DataLoader(
        davis.DavisSet(params, is_train=False),
        batch_size=int(params['batchSize']), shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model = tc.TimeCycle()
    model = Wrap(model, 'forward_affinity')
    
    model = torch.nn.DataParallel(model).cuda()

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


def test(val_loader, model, epoch, use_cuda):

    save_path = args.save_path + '/'
    save_file = '%s/list.txt' % save_path
    fileout = open(save_file, 'w')

    end = time.time()

    for batch_idx, (imgs_total, lbls, meta) in enumerate(val_loader):

        n_context = params['videoLen']
        finput_num = n_context

        # measure data loading time
        imgs_total = torch.autograd.Variable(imgs_total.cuda())


        bs = imgs_total.size(0)
        total_frame_num = imgs_total.size(1)
        channel_num = imgs_total.size(2)
        height_len  = imgs_total.size(3)
        width_len   = imgs_total.size(4)

        assert(bs == 1)

        folder_paths = meta['folder_path']
        print('total_frame_num: ' + str(total_frame_num))

        height_dim = int(params['cropSize'] / 8)
        width_dim  = int(params['cropSize'] / 8)

        
        lbl_set, lbls_resize2, lbls_new = process_labels(lbls, height_dim, width_dim)
        
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
        for t in range(imgs_set.shape[0]):
            img_now = imgs_set[t]

            for c in range(3):
                img_now[c] = img_now[c] * std[c]
                img_now[c] = img_now[c] + mean[c]

            img_now = img_now * 255
            img_now = np.transpose(img_now, (1, 2, 0))
            img_now = cv2.resize(img_now, (img_now.shape[0] * 2, img_now.shape[1] * 2) )

            imgs_toprint.append(img_now)

            imname  = save_path + str(batch_idx) + '_' + str(t) + '_frame.jpg'
            scipy.misc.imsave(imname, img_now)

        for t in range(n_context):
            nowlbl = lbls_new[t]
            imname  = save_path + str(batch_idx) + '_' + str(t) + '_label.jpg'
            scipy.misc.imsave(imname, nowlbl)
        
        ##################################################################
        # Compute image features
        ##################################################################        
        
        
        
        
        ##################################################################
        # Compute correlation features
        ##################################################################
        
        now_batch_size = 4
        imgs_stack = []

        im_num = total_frame_num - n_context
        corrfeat2_set = []

        imgs_tensor = torch.Tensor(now_batch_size, finput_num, 3, params['cropSize'], params['cropSize']).cuda()
        target_tensor = torch.Tensor(now_batch_size, 1, 3, params['cropSize'], params['cropSize']).cuda()

        t03 = time.time()
        
        for iter in range(0, im_num, now_batch_size):

            print(iter)

            startid = iter
            endid   = iter + now_batch_size

            if endid > im_num:
                endid = im_num

            now_batch_size2 = endid - startid

            for i in range(now_batch_size2):

                imgs = imgs_total[:, iter + i + 1: iter + i + n_context, :, :, :]
                imgs2 = imgs_total[:, 0, :, :, :].unsqueeze(1)
                imgs = torch.cat((imgs2, imgs), dim=1)

                imgs_tensor[i] = imgs
                target_tensor[i, 0] = imgs_total[0, iter + i + n_context]

            corrfeat2_now = model(imgs_tensor, target_tensor)
            corrfeat2_now = corrfeat2_now.view(now_batch_size, n_context, corrfeat2_now.size(1), corrfeat2_now.size(2), corrfeat2_now.size(3))

            for i in range(now_batch_size2):
                corrfeat2_set.append(corrfeat2_now[i].data.cpu().numpy())

        t04 = time.time()
        print(t04-t03, 'model forward', t03-t02, 'image prep')
        
        
        ##################################################################
        # Label propagation
        ##################################################################
        topk_vis = args.topk_vis
        
        for iter in range(total_frame_num - n_context):

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

            # predlbls = predlbls / finput_num

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

            scipy.misc.imsave(imname, np.uint8(img_with_heatmap))
            scipy.misc.imsave(imname2, np.uint8(predlbls_val))

    fileout.close()


if __name__ == '__main__':
    main()