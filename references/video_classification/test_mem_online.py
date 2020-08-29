from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import cv2
import imageio

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from models import crawl

from data import davis_test as davis
from data import jhmdb_test as jhmdb

import utils
import test_utils

args = utils.arguments.test_args()

args.imgSize = args.cropSize

print('Context Length:', args.videoLen, 'Image Size:', args.imgSize)
print('Arguments', args)

vis = None
if args.visdom:
    import visdom
    import wandb
    vis = visdom.Visdom(server=args.visdom_server, port=8095, env='main_davis_viz1'); vis.close()
    wandb.init(project='palindromes', group='test_online')
    vis.close()

def main():
    
    # HACK set default training args for online finetune-ing
    args.kldv_coef = 1
    args.long_coef = 1

    args.frame_transforms = 'crop'
    args.frame_aug = 'grid'
    args.npatch = 49
    args.img_size = 256
    args.pstride = [0.5,0.5]
    args.patch_size = [64, 64, 3]
    args.visualize=False

    model = crawl.CRaWl(args, vis=vis).to(args.device)
    
    args.mapScale = model(torch.zeros(1, 10, 3, 320, 320).to(args.device), just_feats=True)[1].shape[-2:]
    args.mapScale = 320 // np.array(args.mapScale)

    dataset = davis.DavisSet(args, is_train=False) if not 'jhmdb' in args.filelist  else \
            jhmdb.JhmdbSet(args, is_train=False)

    val_loader = torch.utils.data.DataLoader(dataset,
        batch_size=int(args.batchSize), shuffle=False, num_workers=args.workers, pin_memory=True)

    cudnn.benchmark = False
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Load checkpoint.
    if os.path.isfile(args.resume):
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume)

        utils.partial_load(checkpoint['model'], model, skip_keys=['head'])

        del checkpoint
    
    model.eval()
    model = model.to(args.device)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    if args.finetune > 0:
        test_loss = test(val_loader, model, args)
    else:
        with torch.no_grad():
            test_loss = test(val_loader, model, args)
            

def test(val_loader, model, args):

    save_path = args.save_path + '/'

    end = time.time()
    
    n_context = args.videoLen

    # Radius mask
    D = None
    t_vid = 0

    import copy
    _model_state = copy.deepcopy(model.state_dict())
    # _model_state = model.state_dict().copy()

    res4 = model.encoder.model.layer4
    ssfc = model.selfsim_fc


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
        print("MODEL StATE AVG", list(_model_state.items())[0][1].mean())

        model.encoder.model.layer4 = res4
        model.selfsim_fc   = ssfc

        model.load_state_dict(_model_state)

        model.xent_coef, model.kldv_coef = 1, 0
        model.dropout_rate = 0.1
        train_len = 3

        def fit(model, video, targets, steps=1):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)#, momentum=0.9, weight_decay=0)
            # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)#, momentum=0.9, weight_decay=0)

            for _ in range(steps):
                fps = np.random.randint(1, 3)
                idx = np.random.randint(video.shape[1]//fps - train_len)
                x = video[:, ::fps][:, idx:idx+train_len].cuda()

                # output, xent_loss, kldv_loss, diagnostics = model(video, orig=video, targets=targets)
                # print('step', _, kldv_loss.mean().item(), diagnostics)

                output, xent_loss, kldv_loss, diagnostics = model(x, orig=x[0], unfold=True)
                if (_ % 20) == 0:
                    print('step', _, xent_loss.mean().item(), diagnostics)

                loss = (xent_loss.mean() + kldv_loss.mean())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            optimizer = None
            torch.cuda.empty_cache()

        # make labels: uniform prob to indices with same mask id
        # targets = lbls_onehot.max(-1)[1]
        # targets = (targets[0, 0:1, ..., None, None] == targets[0, 0:1])
        # targets = targets*1.0/ targets.sum(-1).sum(-1)[..., None, None]*1.0
        # targets = targets.flatten(1,2).flatten(-2,-1).cuda()

        b, bsize = 0, 5
        # fit(model, video, targets, steps=nsteps)
        fit(model, imgs_total, None, steps=args.finetune)
        # import pdb; pdb.set_trace()
        torch.cuda.empty_cache()

        model.encoder.model.layer4 = None
        model.selfsim_fc = tc.Identity()
        model.dropout_rate = 0.0

        with torch.no_grad():

            ##################################################################
            # Compute image features
            ##################################################################
            t00 = time.time()

            feats = []
            bsize = 5
            for b in range(0, imgs_total.shape[1], bsize):
                node, feat = model(imgs_total[:, b:b+bsize].cuda(), orig=None, just_feats=True)
                feats.append(feat.cpu())

            feats = torch.cat(feats, dim=2)
            
            # nodes, feats = model.module(imgs_total, None, True, func='forward')
            feats = feats.squeeze(1)
            if not args.no_l2:
                feats = F.normalize(feats, dim=1)

            print('computed features', time.time()-t00)

            ##################################################################
            # Compute affinities
            ##################################################################
            
            torch.cuda.empty_cache()
            t03 = time.time()
            
            # Prepare source (keys) and target (query) frame features
            key_indices = test_utils.context_index_bank(n_context, args.long_mem, total_frame_num - n_context)
            key_indices = torch.cat(key_indices, dim=-1)
            keys, query = feats[:, :, key_indices], feats[:, :, n_context:]

            # Make spatial radius mask
            restrict = utils.RestrictAttention(args.radius, flat=False)
            D = restrict.mask(*query.shape[-2:])[None]
            D = D.flatten(-4, -3).flatten(-2)
            D[D==0] = -1e10; D[D==1] = 0

            # Flatten source frame features to make context feature set
            keys, query = keys.flatten(-2), query.flatten(-2)

            Ws, Is = test_utils.mem_efficient_batched_affinity(query, keys, mask, 
                        args.temperature, args.topk_vis, args.long_mem, args.device)
            # Ws, Is = test_utils.batched_affinity(query, keys, mask, 
            #             args.temperature, args.topk_vis, args.long_mem, args.device)

            print(time.time()-t03, 'affinity forward, max mem', torch.cuda.max_memory_allocated() / (1024**2))

            if isinstance(lbl_set, list):
                lbl_set = torch.cat(lbl_set)[None]
            lbls_resize[0, n_context*2 - 1:] *= 0

            lbl_set, lbls_resize = lbl_set.squeeze(0), lbls_resize.squeeze(0)

            ##################################################################
            # Label propagation
            ##################################################################

            maps = []
            keypts = []
            images = []
            for it in range(key_indices.shape[0]):
                if it % 10 == 0:
                    print(it)

                lbls_base = lbls_resize[key_indices[it]].cuda()
                flat_lbls = lbls_base.flatten(0, 2).transpose(0, 1)

                predlbls = (flat_lbls[:, Is[it]] * Ws[it].cuda()[None]).sum(1)
                predlbls = predlbls.view(-1, *feats.shape[-2:])

                # print(predlbls.mean(-1).mean(-1))
                #predlbls = test_utils.hard_prop(predlbls)

                predlbls = predlbls.permute(1,2,0)

                img_now = imgs_toprint[it + n_context].permute(1,2,0).numpy() * 255
                
                if it > 0:
                    lbls_resize[it + n_context] = predlbls
                else:
                    predlbls = lbls_resize[0]
                    lbls_resize[it + n_context] = predlbls

                if args.norm_mask:
                    # import pdb; pdb.set_trace()
                    predlbls[:, :, :] -= predlbls.min(-1)[0][:, :, None]
                    predlbls[:, :, :] /= predlbls.max(-1)[0][:, :, None]

                _maps = []

                if 'jhmdb' in args.filelist.lower():
                    coords, predlbls_sharp = test_utils.process_pose(predlbls, lbl_set)
                    keypts.append(coords)
                    pose_map = utils.vis_pose(np.array(img_now).copy(), coords.numpy() * args.mapScale[..., None])
                    _maps += [pose_map]


                # Save Predictions            
                if 'VIP' in args.filelist:
                    outpath = os.path.join(save_path, 'videos'+meta['img_paths'][it+n_context][0].split('videos')[-1])
                    os.makedirs(os.path.dirname(outpath), exist_ok=True)
                else:
                    outpath = os.path.join(save_path, str(batch_idx) + '_' + str(it))

                heatmap, lblmap, heatmap_prob = test_utils.dump_predictions(
                    predlbls.cpu().numpy(),
                    lbl_set, img_now, outpath)

                _maps += [heatmap, lblmap, heatmap_prob]
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
                for m in maps:
                    wandb.log({'blend vid%s' % batch_idx: wandb.Image(
                        m[0])})

                # wandb.log({'blend vid%s' % batch_idx: wandb.Video(
                # np.array([m[0] for m in maps]).transpose(0, -1, 1, 2), fps=12, format="gif")})  
                
                wandb.log({'prob vid%s' % batch_idx: wandb.Video(
                    np.array([m[-1] for m in maps]).transpose(0, -1, 1, 2), fps=12, format="gif")})  
                
                wandb.log({'plain vid%s' % batch_idx: wandb.Video(
                    np.array(images).transpose(0, -1, 1, 2), fps=12, format="gif")})  
                
            torch.cuda.empty_cache()
            print('******* Vid %s TOOK %s *******' % (batch_idx, time.time() - t_vid))


if __name__ == '__main__':
    main()
