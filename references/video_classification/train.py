from __future__ import print_function
import datetime
import os
import time
import sys

import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torchvision
import torchvision.datasets.video_utils
from kinetics import Kinetics400
from video_folder import VideoList

import utils
import data

from sampler import DistributedSampler, UniformClipSampler, RandomClipSampler
from scheduler import WarmupMultiStepLR
import transforms as T
import numpy as np

try:
    from apex import amp
except ImportError:
    amp = None

def nn_visualize(model, data_loader, device, vis=None):

    header = 'Visualizing'
    f1 = []
    X1 = []
    for i, (video, orig) in enumerate(data_loader):
        start_time = time.time()
        print('#### %s ####' % i)

        video = video.to(device)
        f, m = model(video, orig=orig, just_feats=True)

        if True:
            vid = video.view(*video.shape[:2], video.shape[2]//3, 3, *video.shape[-2:])
            vid = vid.flatten(0, -4)
            ff = f.permute(0, 2,3,1).flatten(0, -2)
            X1.append(vid.detach().cpu())            
            f1.append(ff.detach().cpu())
            # import pdb; pdb.set_trace()

        else:
            feats = torch.nn.functional.normalize(m.sum(-1).sum(-1).transpose(-1, -2).contiguous().view(-1, m.shape[1]), dim=-1, p=2).detach().cpu()
    
            f1.append(feats)
            X1.append(video.view(-1, *video.shape[2:]).detach().cpu())
            
        if len(f1) > 250:
            break
        
    utils.nn_pca(torch.cat(f1), torch.cat(X1), name='timecycle-nn')
    import pdb; pdb.set_trace()

def visualize(model, data_loader, device, vis=None):
    import torch.nn.functional as F
    import time

    from sklearn.feature_extraction import image
    from sklearn.cluster import spectral_clustering
    from sklearn.cluster import KMeans

    M = [] # list of tuples of diffusion videos
    model_module = model if not hasattr(model, 'module') else model.module

    import wandb
    import cv2
    import matplotlib.pyplot as plt

    vis.vis.close()

    # wandb.init(project='palindromes', group='ncuts')

    with torch.no_grad():

        for i, (video, orig) in enumerate(data_loader):
            start_time = time.time()

            video = video.to(device)
            # video -= video.min(); video /= video.max()
            # [vis.vis.images(v, env='viz_kinetics') for v in video]

            feats, maps = model(video, orig=orig, just_feats=True)
            H = W = maps.shape[-1]
            N = feats.shape[-1]
            T = feats.shape[-2]

            video = video[0].detach().cpu().numpy()
            video -= video.min(); video /= video.max()

            feats = feats[0].flatten(-2)

            def do(S, prefix='', mode='ncuts'):
                for nc in [8, 16, 24]:
                    if mode == 'ncuts':
                        labels = spectral_clustering(S, n_clusters=nc, eigen_solver='arpack')
                    elif mode == 'kmeans':
                        labels = KMeans(n_clusters=nc, random_state=0).fit_predict(S)
                    else:
                        assert False, 'invalid cluster mode'

                    labels = labels.reshape(T, H, W)            
                    
                    labels = torch.nn.functional.interpolate(
                        torch.from_numpy(labels)[:, None].float(),
                        mode='bilinear',
                        size=video.shape[-2:]).round().int()[:, 0]
                    labels = plt.cm.Paired(labels)[..., :3].transpose(0, -1, 1, 2)

                    vis.vis.images(video*0.5 + labels * 0.5,
                        opts=dict(title="%s_%s" %(prefix, nc)))


            # 1
            S = torch.matmul(feats.t(), feats).detach().cpu().numpy()
            feats = feats.detach().cpu().numpy()
            
            S1 = np.exp(S/ 0.1)
            S1 /= S1.max()

            S2 = np.exp(S/ 0.2)
            S2 /= S2.max()

            S3 = torch.nn.functional.softmax(torch.from_numpy(S) / 0.1, dim=-1).numpy()
            S4 = torch.nn.functional.softmax(torch.from_numpy(S) / 0.05, dim=-1).numpy()
            S5 = torch.nn.functional.softmax(torch.from_numpy(S) / 0.15, dim=-1).numpy()

            # S3 = S.copy()
            # for t in range(T):
            #     S3[:, t*N:(t+1)*N] *= 0
            
            # do(np.clip(S, 0, 1), 'plain')
            # do(S1, 'temp 0.1')
            # do(S2, 'temp 0.2')
            do(S3, 'softmax, temp 0.1')
            do(S4, 'softmax, temp 0.05')
            # do(S5, 'softmax, temp 0.15')
            # do(feats.T, 'kmeans', 'kmeans')
            # do(S3 @ S3.transpose())            
            # do(S3.transpose() @ S3)            



            vis.vis.text('', opts=dict(width=10000, height=10))
            # import pdb; pdb.set_trace()

            # softmax_temp = model_module.temperature
            # for j, ff in enumerate(feats):

            #     ff = feats[j].transpose(0,1)    # C x T -> T x C
            #     xx = video[j:j+1, :, :]
            #     oo = orig[j:j+1]
                
            #     A_traj = model_module.compute_affinity(ff[:-1] , ff[1:], do_dropout=False)

            #     # A_t = [F.softmax(A_traj[0]/softmax_temp, dim=-1)]

            #     N = A_traj[0].shape[-1]
            #     H = int(N**0.5)
            #     A_t, source = [torch.zeros(A_traj[0].shape).cuda()], N // 2
            #     A_t[0][source, source] = 1

            #     for A_tp1 in A_traj[:]:
            #         A_t.append(F.softmax(A_tp1/softmax_temp, dim=-1) @ A_t[-1])

            #     A_t = torch.stack(A_t)

            #     import pdb; pdb.set_trace()

            # if (i+1) % 4 == 0:
            #     # import pdb; pdb.set_trace()
            #     input('#### %s #### %s' % (i, 'Next visualizations?'))

def tsne(model, data_loader, device, vis=None):
    f1 = []
    X1 = []
    for i, (video, orig) in enumerate(data_loader):
        start_time = time.time()
        print('#### %s ####' % i)

        video = video.to(device)
        f, m = model(video, orig=orig, just_feats=True)

        if True:
            vid = video.view(*video.shape[:2], video.shape[2]//3, 3, *video.shape[-2:])
            vid = vid.flatten(0, -4)
            ff = f.permute(0, 2, 3, 1).flatten(0, -2)
            X1.append(vid.detach().cpu())            
            f1.append(ff.detach().cpu())
            # import pdb; pdb.set_trace()
        else:
            feats = torch.nn.functional.normalize(m.sum(-1).sum(-1).transpose(-1, -2).contiguous().view(-1, m.shape[1]), dim=-1, p=2).detach().cpu()
    
            f1.append(feats)
            X1.append(video.view(-1, *video.shape[2:]).detach().cpu())
            
        if len(f1) > 50:
            break
    
    f1, X1 = torch.cat(f1), torch.cat(X1)
    X1 -= X1.min()
    X1 /= X1.max()

    import pickle
    import matplotlib
    from matplotlib.pyplot import imshow
    from PIL import Image
    from sklearn.manifold import TSNE

    for _,idx in enumerate(range(0, f1.shape[0], 49)):
        ff, xx = f1[idx:idx+49], X1[idx:idx+49]
        tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(ff)

        tx, ty = tsne[:,0], tsne[:,1]
        tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))


        width = 500
        height = 500
        max_dim = 100

        full_image = Image.new('RGBA', (width, height))
        for img, x, y in zip(xx, tx, ty):
            tile = Image.fromarray((img.numpy().transpose(1,2,0) * 255).astype(np.uint8))
            rs = max(1, tile.width/max_dim, tile.height/max_dim)
            tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
            full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

        out_img = np.array(full_image).transpose(2, 0, 1)[:3]
        vis.vis.image(out_img)
        full_image.save('scene_tsne%s.png' % str(idx))

        if _ > 20:
            break
        
    import pdb; pdb.set_trace()

    
def visualize2(model, data_loader, device, vis=None):
    import torch.nn.functional as F
    import time

    M = [] # list of tuples of diffusion videos
    model_module = model if not hasattr(model, 'module') else model.module

    for i, (video, orig) in enumerate(data_loader):
        start_time = time.time()

        video = video.to(device)
        # video -= video.min(); video /= video.max()
        # [vis.vis.images(v, env='viz_kinetics') for v in video]

        feats, _ = model(video, orig=orig, just_feats=True)
        tviz = 0
        
        def diffusion_maps(ff, xx, softmax_temp=None, name=''):
            if softmax_temp is None:
                softmax_temp = model_module.temperature

            with torch.no_grad():
                ff = ff.transpose(0,1)
                A_traj = model_module.compute_affinity(ff[:-1] , ff[1:], do_dropout=False)

                # A_t = [model.stoch_mat(A_traj[0])]
                # for A_tp1 in A_traj[1:]:
                #     A_t.append(model.stoch_mat(A_tp1) @ A_t[-1])

                A_t = [F.softmax(A_traj[0]/softmax_temp, dim=-1)]
                for A_tp1 in A_traj[1:]:
                    A_t.append(F.softmax(A_tp1/softmax_temp, dim=-1) @ A_t[-1])

                A_t = torch.stack(A_t)

                # import pdb; pdb.set_trace()
                pg = utils.PatchGraph(xx, A_t, orig=orig,
                    viz=vis.vis, win='patchgraph_%s' % str(name))

                # return [pg.blend(i)[0] for i in range(pg.N)]

        for j, ff in enumerate(feats):
            maps = [
                diffusion_maps(
                    ff, video[j:j+1, :, :],
                    softmax_temp=stemp, name='%s-%s-%s-%s' %(i,j,stemp, str(time.time())))
                for stemp in [0.05]
            ]
            M.append(maps)


        if (i+1) % 4 == 0:
            # import pdb; pdb.set_trace()
            input('#### %s #### %s' % (i, 'Next visualizations?'))
        
        
        
def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq,
    apex=False, max_steps=1e10, vis=None, checkpoint_fn=None):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)

    for step, (video, orig) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if step > max_steps:
            break

        start_time = time.time()

        video = video.to(device)
        output, xent_loss, kldv_loss, diagnostics = model(video, orig=orig)
        xent_loss, kldv_loss = xent_loss.mean(), kldv_loss.mean()

        loss = 0
        if xent_loss > 0:
            loss += xent_loss
        if kldv_loss > 0:
            loss += kldv_loss

        # print(xent_loss, kldv_loss, loss)

        # if vis is not None and np.random.random() < 0.05:
        #     vis.wandb_init()
        #     vis.log(dict(xent_loss=xent_loss.mean().item()))
        #     vis.log(dict(kldv_loss=kldv_loss.mean().item()))
        #     for k,v in diagnostics.items():
        #         vis.log({k:v.mean().item()})

        if checkpoint_fn is not None and np.random.random() < 0.005:
            checkpoint_fn()
 
        # output, xent_loss, kldv_loss = model(video)
        # loss = (kldv_loss.mean() + xent_loss.mean()) #+ kldv_loss
        # print(loss)
        # loss = criterion(output, target)

        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # print(torch.nn.utils.clip_grad_norm_(model.parameters(), 10), 'grad norm')
        optimizer.step()

        # acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = video.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()

    checkpoint_fn()




def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "kinetics", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def collate_fn(batch):
    # remove audio from the batch
    batch = [d[0] for d in batch]
    return default_collate(batch)


def main(args):
    if args.apex:
        if sys.version_info < (3, 0):
            raise RuntimeError("Apex currently only supports Python 3. Aborting.")
        if amp is None:
            raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                               "to enable mixed-precision training.")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    vis = utils.Visualize(args)

    # utils.init_distributed_mode(args)
    args.distributed = False
    
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    ##### VIDEO SPECIFIC ##########
    # Data loading code
    print("Loading data")
    traindir = os.path.join(args.data_path, 'train_256' if not args.fast_test else 'val_256_bob')
    valdir = os.path.join(args.data_path, 'val_256_bob')

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    ######################

    frame_transform_train = utils.get_frame_transform(args)
    transform_train = torchvision.transforms.Compose([
        frame_transform_train,
        # T.ToFloatTensorInZeroOne(),
        # T.Resize((256, 256)),
    ])

    def prep_dataset():
        def make_dataset(is_train):
            _transform = transform_train if is_train else transform_test

            if 'kinetics' in args.data_path.lower():
                return Kinetics400(
                    traindir if is_train else valdir,
                    frames_per_clip=args.clip_len,
                    step_between_clips=1,
                    transform=transform_train,
                    extensions=('mp4'),
                    frame_rate=args.frame_skip
                )
            elif os.path.isdir(args.data_path): # HACK assume image dataset if data path is a directory
                return torchvision.datasets.ImageFolder(
                    root=args.data_path,
                    transform=_transform)
            else:
                return VideoList(
                    filelist=args.data_path,
                    clip_len=args.clip_len,
                    is_train=is_train,
                    frame_gap=args.frame_skip,
                    transform=_transform,
                    random_clip=True,
                    # frame_transform=_frame_transform
                )

        if args.cache_dataset and os.path.exists(cache_path):
            print("Loading dataset_train from {}".format(cache_path))
            dataset, _ = torch.load(cache_path)
            dataset.transform = transform_train
        else:
            if args.distributed:
                print("It is recommended to pre-compute the dataset cache "
                    "on a single-gpu first, as it will be faster")
            dataset = make_dataset(is_train=True)


            if args.cache_dataset and 'kinetics' in args.data_path.lower():
                print("Saving dataset_train to {}".format(cache_path))
                utils.mkdir(os.path.dirname(cache_path))
                dataset.transform = None
                utils.save_on_master((dataset, traindir), cache_path)
        
        if hasattr(dataset, 'video_clips'):
            dataset.video_clips.compute_clips(args.clip_len, 1, frame_rate=args.frame_skip)

        return dataset
        
    dataset = prep_dataset()
    print("Took", time.time() - st)


    def make_data_sampler(is_train, dataset):
        torch.manual_seed(0)
        if hasattr(dataset, 'video_clips'):
            _sampler = RandomClipSampler if is_train else UniformClipSampler
            # _sampler = UniformClipSampler
            return _sampler(dataset.video_clips, args.clips_per_video)
        else:
            return torch.utils.data.sampler.RandomSampler(dataset) if is_train else None
    
        
    print("Creating data loaders")
    train_sampler = make_data_sampler(True, dataset)
    # train_sampler = train_sampler(dataset.video_clips, args.clips_per_video)
    # test_sampler = test_sampler(dataset_test.video_clips, args.clips_per_video)

    if args.distributed:
        train_sampler = DistributedSampler(train_sampler)

    # 64px
    data_loader_64 = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, # shuffle=not args.fast_test,
        sampler=train_sampler, num_workers=args.workers//2,
        pin_memory=False, collate_fn=collate_fn)

    # 128px
    # args.patch_size = (128, 128, 3)
    # dataset_2 = prep_dataset()
    # dataset_2.transform = utils.get_frame_transform(args)

    # data_loader_128 = torch.utils.data.DataLoader(
    #     dataset_2, batch_size=args.batch_size,
    #     sampler=train_sampler, num_workers=args.workers//2,
    #     pin_memory=True, collate_fn=collate_fn)

    # # data_loader = utils.AlternatingLoader([data_loader_64, data_loader_128])
    data_loader = data_loader_64
    
    print("Creating model")
    import resnet3d as resnet
    import timecycle as tc
    import patchcycle as pc

    # model = resnet.__dict__[args.model](pretrained=args.pretrained)
    model = tc.TimeCycle(args, vis=vis).to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()

    lr = args.lr * args.world_size

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        assert args.optim == 'adam'
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.apex_opt_level )

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
        warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.data_parallel:
        model = torch.nn.parallel.DataParallel(model)
        model_without_ddp = model.module
    
    ########################### 
    if args.partial_reload:
        checkpoint = torch.load(args.partial_reload, map_location='cpu')
        utils.partial_load(checkpoint['model'], model_without_ddp)        

    if args.reload:
        checkpoint = torch.load(args.reload, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        optimizer.param_groups[0]["lr"] = args.lr
        args.start_epoch = checkpoint['epoch'] + 1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
    ########################### 

    if args.visualize:
        # nn_visualize(model, data_loader, device, vis=vis)
        # visualize(model, data_loader, device, vis=vis)
        tsne(model, data_loader, device, vis=vis)
        return

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device)
        return

    def save_model_checkpoint():
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader,
                        device, epoch, args.print_freq, args.apex, max_steps=args.steps_per_epoch,
                        vis=vis, checkpoint_fn=save_model_checkpoint)

        # eval on davis
        # from run_test import test as davis_test
        # scores = davis_test(os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)), gpu=1)
        # vis.log('Davis J mean', scores['J']['mean'])                    
        # vis.log('Davis F mean', scores['F']['mean'])

        # evaluate(model, criterion, data_loader_test, device=device)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='/data/ajabri/kinetics/',
        help='/home/ajabri/data/places365_standard/train/ | /data/ajabri/kinetics/')

    parser.add_argument('--model', default='r3d_18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--clip-len', default=8, type=int, metavar='N',
                        help='number of frames per clip')
    parser.add_argument('--clips-per-video', default=5, type=int, metavar='N',
                        help='maximum number of clips per video to consider')
    parser.add_argument('-b', '--batch-size', default=24, type=int)
    parser.add_argument('--epochs', default=45, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--steps-per-epoch', default=1e10, type=int, help='max number of batches per epoch')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--lr-milestones', nargs='+', default=[20, 30, 40], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.3, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=0, type=int, help='number of warmup epochs')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='auto', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--reload', default='', help='reload net from checkpoint without optimizer state')
    parser.add_argument('--partial-reload', default='', help='reload net from checkpoint, ignoring keys that are not in current model')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    
    parser.add_argument( "--cache-dataset", dest="cache_dataset", help="Cache the datasets for quicker initialization. It also serializes the transforms", action="store_true", )
    parser.add_argument( "--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true", )
    parser.add_argument( "--test-only", dest="test_only", help="Only test the model", action="store_true", )

    parser.add_argument( "--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo", action="store_true", )

    parser.add_argument( "--data-parallel", dest="data_parallel", help="", action="store_true", )

    parser.add_argument( "--zero-diagonal", dest="zero_diagonal", help="", action="store_true", )

    parser.add_argument( "--fast-test", dest="fast_test", help="", action="store_true", )

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str, help='For apex mixed precision training' 'O0 for FP32 training, O1 for mixed precision training.' 'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet' )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')


    # my args
    parser.add_argument('--xent-coef', default=1, type=float, help='initial learning rate')
    parser.add_argument('--kldv-coef', default=0, type=float, help='initial learning rate')
    parser.add_argument('--dropout', default=0, type=float, help='dropout rate on A')
    parser.add_argument('--name', default='', type=str, help='')

    parser.add_argument('--frame-transforms', default='crop', type=str,
        help='blur ^ crop ^ cj ^ flip')
    parser.add_argument('--frame-aug', default='', type=str,
        help='(randpatch | grid) + cj ^ flip')

    parser.add_argument('--frame-skip', default=8, type=int, help='number of frames to skip when sampling')

    parser.add_argument('--img-size', default=256, type=int, help='number of patches sampled')
    parser.add_argument('--patch-size', default=[64, 64, 3], type=int, nargs="+", help='number of patches sampled')
    parser.add_argument('--nrel', default=10, type=int, help='number of heads')

    parser.add_argument('--port', default=8095, type=int, help='number of heads')
    parser.add_argument('--server', default='localhost', type=str, help='number of heads')
    parser.add_argument('--npatch', default=5, type=int, help='number of patches sampled')

    # patchifying args
    parser.add_argument('--pstride', default=[0.5, 0.5], nargs=2,
        type=float, help='random sample patchify stride from [this*patch_size, patch_size]')
    parser.add_argument('--npatch-scale', default=[0.2, 0.8], nargs=2,
        type=float, help='range from which to same patch sizes for random patch sampling')
    parser.add_argument('--edgefunc', default='softmax', type=str, help='softmax | relu')

    parser.add_argument('--model-type', default='scratch', type=str, help='scratch | imagenet | moco')
    parser.add_argument('--optim', default='adam', type=str, help='adam | sgd')

    parser.add_argument('--temp', default=0.08,
        type=float, help='softmax temperature when computing affinity')

    parser.add_argument('--featdrop', default=0.0,
        type=float, help='dropout on features')

    parser.add_argument('--restrict', default=-1,
        type=int, help='dropout on features')

    parser.add_argument('--head-depth', default=0,
        type=int, help='depth of head mlp')

    parser.add_argument('--visualize', default=False,
        action='store_true', help='visualize trained model')

    parser.add_argument('--long-coef', default=1,
        type=float, help='long cycle loss coef')
    parser.add_argument('--skip-coef', default=0,
        type=float, help='skip cycle loss coef')
    parser.add_argument('--cal-coef', default=0.0,
        type=float, help='contrastive affinity')

    parser.add_argument('--shuffle', default=0.0,
        type=float, help='shuffle patches across instances for different negatives')

    parser.add_argument('--xent-weight', default=False, action='store_true',
        help='use out-going entropy * max similarity as a loss gate')

    parser.add_argument('--no-maxpool', default=False, action='store_true',
        help='')
    parser.add_argument('--use-res4', default=False, action='store_true',
        help='')
    parser.add_argument('--skip-res3', default=False, action='store_true',
        help='')
    parser.add_argument('--skip-res2', default=False, action='store_true',
        help='')

    # sinkhorn-knopp ideas
    parser.add_argument('--sk-align', default=False, action='store_true',
        help='')

    parser.add_argument('--sk-targets', default=False, action='store_true',
        help='')

    parser.add_argument('--random-resize', default=[256, 256], nargs=2, type=int,
        help='')

    args = parser.parse_args()

    if args.fast_test:
        args.batch_size = 1
        args.workers = 0
        args.data_parallel = False

    if args.output_dir == 'auto':
        args.dataset = 'kinetics' if 'kinetics' in args.data_path else 'pennaction'
        keys = {
            'dropout':'drop', 'clip_len': 'len', 'frame_transforms': 'ftrans', 'frame_aug':'faug', 'zero_diagonal':'zdiag', 
            'pstride':'pstride', 'optim':'optim', 'temp':'temp',
            'featdrop':'fdrop', 'lr':'lr', 'skip_coef':'skip', 'head_depth':'mlp'
        }
        name = '-'.join(["%s%s" % (keys[k], getattr(args, k) if not isinstance(getattr(args, k), list) else '-'.join([str(s) for s in getattr(args, k)])) for k in keys])
        args.output_dir = "checkpoints/%s_%s/" % (args.name, name)

        import datetime
        dt = datetime.datetime.today()
        args.name = "%s-%s--%s_%s" % (str(dt.month), str(dt.day), args.name, name)

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
