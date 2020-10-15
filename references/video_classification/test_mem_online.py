from __future__ import print_function

import os
import time
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import model
from model import CRaWl
# from models.crawl import CRaWl
# from crawl import CRaWl

from data import davis_test as davis
from data import jhmdb_test as jhmdb
from data.video import SingleVideoDataset

import utils
import test_utils


def main(args, vis):
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

    model = CRaWl(args, vis=vis).to(args.device)
    args.mapScale = test_utils.infer_downscale(model)

    dataset = (davis.DavisSet if not 'jhmdb' in args.filelist  else jhmdb.JhmdbSet)(args)
    val_loader = torch.utils.data.DataLoader(dataset,
        batch_size=int(args.batchSize), shuffle=False, num_workers=args.workers, pin_memory=True)

    # cudnn.benchmark = False
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
            

def test(loader, model, args):

    end = time.time()
    
    n_context = args.videoLen

    # Radius mask
    D = None
    t_vid = 0

    import copy
    _model_state = copy.deepcopy(model.state_dict())

    res4 = model.encoder.model.layer4
    ssfc = model.selfsim_fc

    for vid_idx, (imgs, imgs_orig, lbls, lbls_orig, lbl_map, meta) in enumerate(loader):
        t_vid = time.time()

        # measure data loading time
        imgs = imgs.to(args.device)
        B, N = imgs.shape[:2]
        assert(B == 1)

        print('******* Vid %s (%s frames) *******' % (vid_idx, N))

        ##################################################################
        # Self-supervised Adaptation
        ##################################################################        
        print("Model State Avg (sanity)", list(_model_state.items())[0][1].mean())

        # model.encoder.model.layer4 = res4
        # model.selfsim_fc = ssfc

        model.load_state_dict(_model_state)

        model.xent_coef, model.kldv_coef = 1, 0
        model.args.skip_coef = 1.0
        model.dropout_rate = 0.0

        def fit(model, video, targets, steps=1, bsz=4):
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
            dataset = SingleVideoDataset(video[0].cpu(), 8, fps_range=[1,2], n_clips=steps*bsz)
            train_loader = torch.utils.data.DataLoader(dataset,
                 batch_size=bsz, shuffle=False, num_workers=args.workers, pin_memory=True)
            temp = model.temperature

            for step, x in enumerate(train_loader):
                x = x.to(args.device)

                # output, xent_loss, kldv_loss, diagnostics = model(video, orig=video, targets=targets)
                # print('step', _, kldv_loss.mean().item(), diagnostics)
                model.temperature = 0.01 if step < 50 else temp

                if np.random.random() > 0:            
                    output, xent_loss, kldv_loss, diagnostics = model(x, orig=x[:, 0], unfold=True)
                else:
                    # x = random_crop(x[0])[None]
                    _h, _w = x.shape[-2:]
                    offset = np.random.randint(0, _w-_h-1)
                    x = x[..., offset:offset+_h]
                    output, xent_loss, kldv_loss, diagnostics = model(x, orig=x[:, 0], unfold=False)

                if (step % 10) == 0:
                    print('step', step, xent_loss.mean().item(), diagnostics)

                loss = (xent_loss.mean() + kldv_loss.mean())
                optimizer.zero_grad()
                loss.backward()
                # print(torch.nn.utils.clip_grad_norm_(model.parameters(), 5), 'grad norm')

                optimizer.step()
            
            optimizer = None
            torch.cuda.empty_cache()

        # make labels: uniform prob to indices with same mask id
        # targets = lbls_onehot.max(-1)[1]
        # targets = (targets[0, 0:1, ..., None, None] == targets[0, 0:1])
        # targets = targets*1.0/ targets.sum(-1).sum(-1)[..., None, None]*1.0
        # targets = targets.flatten(1,2).flatten(-2,-1).cuda()
        # fit(model, video, targets, steps=nsteps)
        if args.finetune > 0:
            fit(model, imgs, None, steps=args.finetune)
            torch.cuda.empty_cache()

        # model.encoder.model.layer4 = None
        # model.selfsim_fc = crawl.Identity()
        model.dropout_rate = 0.0

        with torch.no_grad():

        ##################################################################
        # Compute image features (batched for memory efficiency)
        ##################################################################
            t00 = time.time()

            bsize = 5   # minibatch size for computing features
            feats = []
            for b in range(0, imgs.shape[1], bsize):
                node, feat = model(imgs[:, b:b+bsize].to(args.device), orig=None, just_feats=True)
                feats.append(feat.cpu())

            feats = torch.cat(feats, dim=2).squeeze(1)            
            if not args.no_l2:
                feats = torch.nn.functional.normalize(feats, dim=1)

            print('computed features', time.time()-t00)

        ##################################################################
        # Compute affinities
        ##################################################################
            
            torch.cuda.empty_cache()
            t03 = time.time()
            
            # Prepare source (keys) and target (query) frame features
            key_indices = test_utils.context_index_bank(n_context, args.long_mem, N - n_context)
            key_indices = torch.cat(key_indices, dim=-1)
            keys, query = feats[:, :, key_indices], feats[:, :, n_context:]

            # Make spatial radius mask
            restrict = utils.MaskedAttention(args.radius, flat=False)
            D = restrict.mask(*feats.shape[-2:])[None]
            D = D.flatten(-4, -3).flatten(-2)
            D[D==0] = -1e10; D[D==1] = 0

            # Flatten source frame features to make context feature set
            keys, query = keys.flatten(-2), query.flatten(-2)

            print('computing affinity')
            Ws, Is = test_utils.mem_efficient_batched_affinity(query, keys, D, 
                        args.temperature, args.topk_vis, args.long_mem, args.device)
            # Ws, Is = test_utils.batched_affinity(query, keys, D, 
            #             args.temperature, args.topk_vis, args.long_mem, args.device)

            if torch.cuda.is_available():
                print(time.time()-t03, 'affinity forward, max mem', torch.cuda.max_memory_allocated() / (1024**2))


        ##################################################################
        # Propagate Labels and Save Predictions
        ###################################################################
            lbls[0, n_context*2 - 1:] *= 0 
            lbl_map, lbls = lbl_map.squeeze(0), lbls.squeeze(0)

            maps, keypts = [], []
            for t in range(key_indices.shape[0]):
                if t % 10 == 0:
                    print(t)

                ctx_lbls = lbls[key_indices[t]].to(args.device)
                ctx_lbls = ctx_lbls.flatten(0, 2).transpose(0, 1)

                pred = (ctx_lbls[:, Is[t]] * Ws[t].to(args.device)[None]).sum(1)
                pred = pred.view(-1, *feats.shape[-2:])

                #pred = test_utils.hard_prop(pred)

                pred = pred.permute(1,2,0)

                cur_img = imgs_orig[0, t + n_context].permute(1,2,0).numpy() * 255
                
                if t > 0:
                    lbls[t + n_context] = pred
                else:
                    pred = lbls[0]
                    lbls[t + n_context] = pred

                if args.norm_mask:
                    # import pdb; pdb.set_trace()
                    pred[:, :, :] -= pred.min(-1)[0][:, :, None]
                    pred[:, :, :] /= pred.max(-1)[0][:, :, None]

                _maps = []

                if 'jhmdb' in args.filelist.lower():
                    coords, pred_sharp = test_utils.process_pose(pred, lbl_map)
                    keypts.append(coords)
                    pose_map = utils.vis_pose(np.array(cur_img).copy(), coords.numpy() * args.mapScale[..., None])
                    _maps += [pose_map]

                # Save Predictions            
                if 'VIP' in args.filelist:
                    outpath = os.path.join(args.save_path, 'videos'+meta['img_paths'][t+n_context][0].split('videos')[-1])
                    os.makedirs(os.path.dirname(outpath), exist_ok=True)
                else:
                    outpath = os.path.join(args.save_path, str(vid_idx) + '_' + str(t))

                heatmap, lblmap, heatmap_prob = test_utils.dump_predictions(
                    pred.cpu().numpy(),
                    lbl_map, cur_img, outpath)

                _maps += [heatmap, lblmap, heatmap_prob]
                maps.append(_maps)

                if args.visdom:
                    [vis.image(np.uint8(_m).transpose(2, 0, 1)) for _m in _maps]

            if len(keypts) > 0:
                coordpath = os.path.join(args.save_path, str(vid_idx) + '.dat')
                np.stack(keypts, axis=-1).dump(coordpath)
            
            if args.visdom:
                wandb.log({'blend vid%s' % vid_idx: wandb.Video(
                    np.array([m[0] for m in maps]).transpose(0, -1, 1, 2), fps=12, format="gif")})  
                # wandb.log({'prob vid%s' % vid_idx: wandb.Video(
                #     np.array([m[-1] for m in maps]).transpose(0, -1, 1, 2), fps=12, format="gif")})  
                wandb.log({'plain vid%s' % vid_idx: wandb.Video(
                    imgs_orig[0, n_context:].numpy(), fps=4, format="gif")})  
                
            torch.cuda.empty_cache()
            print('******* Vid %s TOOK %s *******' % (vid_idx, time.time() - t_vid))


if __name__ == '__main__':
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

    main(args, vis)
