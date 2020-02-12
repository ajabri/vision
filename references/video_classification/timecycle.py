import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter

import torchvision
import resnet3d
import resnet2d
import itertools

import time
import numpy as np
import cv2
import visdom
import utils

from matplotlib import cm
color = cm.get_cmap('winter')

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

def unfold_time(x, T):
    return x.view(int(x.shape[0] / T), T, *x.shape[1:])

def fold_time(x):
    return x.view(x.shape[0] * x.shape[1], *x.shape[2:])

class UnfoldTime(nn.Module):
    def __init__(self, T):
        super(UnfoldTime, self).__init__()
        self.T = T
    
    def forward(self, x):
        return x.view(int(x.shape[0] / self.T), self.T, *x.shape[1:])

class FoldTime(nn.Module):
    def __init__(self, T):
        super(FoldTime, self).__init__()
        self.T = T
    
    def forward(self, x):
        return x.view(x.shape[0] * x.shape[1], *x.shape[2:])


class TimeCycle(nn.Module):
    def __init__(self, args=None, vis=None):
        super(TimeCycle, self).__init__()
        # self.resnet = resnet3d.r3d_18(pretrained=False)
        self.resnet = resnet3d.r2d_10()
#         self.resnet = resnet3d.r2d_18(pretrained=True)

        self.resnet.fc, self.resnet.avgpool, self.resnet.layer4 = None, None, None

        self.infer_dims()
        # self.resnet_nchan = self.resnet.

        self.selfsim_fc = torch.nn.Linear(self.enc_hid_dim, 128)
        self.selfsim_head = self.make_head([self.enc_hid_dim, 2*self.enc_hid_dim, self.enc_hid_dim])
        self.context_head = self.make_head([self.enc_hid_dim, 2*self.enc_hid_dim, self.enc_hid_dim])
        

        # assuming no fc pre-training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        # self.kldv = torch.nn.KLDivLoss(reduction="batchmean")
        self.kldv = torch.nn.KLDivLoss(reduction="batchmean")
        self.xent = torch.nn.CrossEntropyLoss(reduction="none")

        self.target_temp = 1

        self._xent_targets = {}
        self._kldv_targets = {}
        
        if args is not None:
            self.kldv_coef = args.kldv_coef
            self.xent_coef = args.xent_coef
            self.zero_diagonal = args.zero_diagonal
            self.dropout_rate = args.dropout
        else:
            self.kldv_coef = 0
            self.xent_coef = 0
            self.zero_diagonal = 0
            self.dropout_rate = 0
            
        self.dropout = torch.nn.Dropout(p=self.dropout_rate, inplace=False)
        
        self.viz = visdom.Visdom(port=8095, env='%s_%s' % (args.name if args is not None else 'test', '')) #int(time.time())))
        self.viz.close()

        if vis is not None:
            self._viz = vis

    def infer_dims(self):
        # if '2D' in str(type(self.resnet.conv1)):
        dummy = torch.Tensor(1, 3, 1, 224, 224)
        # else:
        #     dummy = torch.Tensor(1, 3, 224, 224)

        dummy_out = self.resnet(dummy)

        self.enc_hid_dim = dummy_out.shape[1]

        # import pdb; pdb.set_trace()

    def make_head(self, dims):
        head = []

        for d1, d2 in zip(dims, dims[1:]):
            h = nn.Conv3d(d1, d2, kernel_size=1, bias=True)
            nn.init.kaiming_normal_(h.weight, mode='fan_out', nonlinearity='relu')
            head += [h, nn.LeakyReLU(0.1)]

        head = nn.Sequential(*head)
        return head

    def make_smooth_target_2d(self, H, W):
        import time
        t1 = time.time()
        I = torch.eye(H*W).float()
        Is = []
        for _I in I:
            _I = gaussian_filter(_I.view(H, W).numpy(), sigma=self.target_temp)
            _I = F.softmax(torch.from_numpy(_I).log().view(-1))
            Is.append(_I)

        I = torch.stack(Is)
        print('made target ', H, W, time.time()-t1)

        return I

    def zeroout_diag(self, A, zero=0):
        mask = (torch.eye(A.shape[-1]).unsqueeze(0).repeat(A.shape[0], 1, 1).bool() < 1).float().cuda()
        # import pdb; pdb.set_trace()
        return A * mask

    def compute_affinity(self, x1, x2, do_dropout=True, zero_diagonal=None):
        B, C, N = x1.shape
        # assert x1.shape == x2.shape
        A = torch.einsum('bcn,bcm->bnm', x1, x2)
        A = torch.div(A, 1/(C**0.5))
        # A = torch.div(A, 1/C**0.5)

        if (zero_diagonal is not False) and self.zero_diagonal:
            A = self.zeroout_diag(A)
    
        # A12 = A.view(A.size(0), 1, H * H, W, W)
        # A21 = A.view(A.size(0), 1, H, H, W * W) 
        # A12  = F.softmax(A, dim=2)
        # A21  = F.softmax(A.transpose(1, 2), dim=2)

        A1, A2 = A, A.transpose(1, 2)        
        if do_dropout:
            A1, A2 = self.dropout(A1), self.dropout(A2)

        A1, A2 = F.softmax(A1, dim=-1), F.softmax(A2, dim=-1)

        AA = torch.matmul(A2, A1)
        log_AA = torch.log(AA + 1e-20)

        return A, AA, log_AA, A1, A2
    
    def visualize_frame_pair(self, x, ff, mm, t1, t2):
        # B, C, T, N = ff.shape
        f1, f2 = ff[:, :, t1], ff[:, :, t2]

        A, AA, log_AA, A1, A2 = self.compute_affinity(f1, f2, do_dropout=False, zero_diagonal=False)
        log_AA = log_AA.view(-1, log_AA.shape[1])
        _xent_loss = self.xent(log_AA, self.xent_targets(A))
        
        N = A.shape[-1]
        H = W = int(N**0.5)
        _AA = AA.view(-1, H * W, H, W)

        # # FLOW
        # _A = A.view(*A.shape[:2], f1.shape[-1], -1)
        # u, v = utils.compute_flow(_A[0:1])
        # flows = torch.stack([u, v], dim=-1).cpu().numpy()
        # flows = utils.draw_hsv(flows[0])
        # flows = cv2.resize(flows, (256, 256))
        # self.viz.image((flows).transpose(2, 0, 1), win='flow')
        # # flows = [cv2.resize(flow.clip(min=0).astype(np.uint8), (256, 256)) for flow in flows]
        # # self.viz.image((flows[0]).transpose(2, 0, 1))

        # import pdb; pdb.set_trace()

        if len(x.shape) < 6:
            # IMG VIZ
            # X here is B x C x T x H x W
            x1, x2 = x[0, :, t1], x[0, :, t2]
            xx = torch.stack([x1, x2]).detach().cpu()
            xx -= xx.min(); xx /= xx.max()
            self.viz.images(xx, win='imgs')
            # self._viz.patches(xx, A)

            # # PCA VIZ
            pca_ff = utils.pca_feats(torch.stack([f1[0], f2[0]]).detach().cpu())
            pca_ff = utils.make_gif(pca_ff, outname=None)
            self.viz.images(pca_ff.transpose(0, -1, 1, 2), win='pcafeats')
        else:
            # X here is B x N x C x T x H x W
            x1, x2 =  x[0, :, :, t1],  x[0, :, :, t2]
            m1, m2 = mm[0, :, :, t1], mm[0, :, :, t2]

            pca_feats = utils.pca_feats(torch.cat([m1, m2]).detach().cpu())
            pca_feats = utils.make_gif(pca_feats, outname=None, sz=64).transpose(0, -1, 1, 2)
            self.viz.images(pca_feats[:N], nrow=int(N**0.5), win='pca_viz1')
            self.viz.images(pca_feats[N:], nrow=int(N**0.5), win='pca_viz2')
            
            for i, xx in enumerate([x1, x2]):
                self.viz.images((xx-xx.min()) / ((xx-xx.min()).max()), nrow=int(N**0.5), win='pca_viz_imgs_%s' % str(i))

        # LOSS VIS
        xx = _xent_loss[:H*W]
        xx -= xx.min()
        xx /= xx.max()
        img_grid = [cv2.resize(aa, (50,50), interpolation=cv2.INTER_NEAREST)[None] for aa in _AA[0, :, :, :, None].cpu().detach().numpy()]
        img_grid = [img_grid[_].repeat(3, 0) * np.array(color(xx[_].item()))[:3, None, None] for _ in range(H*W)]
        img_grid = [img_grid[_] / img_grid[_].max() for _ in range(H*W)]
        img_grid = torch.from_numpy(np.array(img_grid))
        img_grid = torchvision.utils.make_grid(img_grid, nrow=H, padding=1, pad_value=1)
        
        # img_grid = cv2.resize(img_grid.permute(1, 2, 0).cpu().detach().numpy(), (1000, 1000), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
        self.viz.images(img_grid, win='lossvis')


    def pixels_to_nodes(self, x):
        # Encode into B x C x T x N
        B, N, C, T, h, w = x.shape
        x = x.reshape(B*N, C, T, h, w) 

        mm = self.resnet(x)
        H, W = mm.shape[-2:]

        # produce node vector representations by spatially pooling feature maps
        ff = mm.sum(-1).sum(-1) / (H*W)
        # ff = torch.einsum('ijklm->ijk', ff) / ff.shape[-1]*ff.shape[-2] 

        ff = self.selfsim_fc(ff.transpose(-1, -2)).transpose(-1,-2)
        ff = F.normalize(ff, p=2, dim=1)
    
        # reshape to add back batch and num node dimensions
        ff = ff.view(B, N, ff.shape[1], T).permute(0, 2, 3, 1)
        mm = mm.view(B, N, *mm.shape[1:])

        return ff, mm



    def kldv_targets(self, A):
        '''
            A: affinity matrix
        '''
        B, N = A.shape[:2]
        key = '%s:%sx%s' % (str(A.device), B,N)

        if key not in self._kldv_targets:
            I = self.make_smooth_target_2d(int(N**0.5), int(N**0.5))
            I = I[None].repeat(B, 1, 1).view(-1, A.shape[-1]).to(A.device)
            self._kldv_targets[key] = I

        return self._kldv_targets[key]

    def xent_targets(self, A):
        B, N = A.shape[:2]
        key = '%s:%sx%s' % (str(A.device), B,N)

        if key not in self._xent_targets:
            I = torch.arange(A.shape[-1])[None].repeat(B, 1)
            I = I.view(-1).to(A.device)
            self._xent_targets[key] = I

        return self._xent_targets[key]

    def compute_xent_loss(self, A, log_AA):
        # Cross Entropy
        targets = self.xent_targets(A)
        if self.xent_coef > 0:
            _xent_loss = self.xent(log_AA, targets)

            return _xent_loss.mean().unsqueeze(-1), \
                (torch.argmax(log_AA, dim=-1) == targets).float().mean().unsqueeze(-1)
        else:
            return 0, 0

    def compute_kldv_loss(self, A, log_AA):
        # KL Div with Smoothed 2D Targets
        targets = self.kldv_targets(A)
        if self.kldv_coef > 0:
            kldv_loss = self.kldv(log_AA, targets)
            # print(kldv_loss, log_AA.min(), AA.min(), A.min())
            return kldv_loss
        else:
            return 0

    def forward(self, x, orig=None):
        xents = torch.tensor([0.]).cuda()
        kldvs = torch.tensor([0.]).cuda()
        diags = dict(skip_accur=torch.tensor([0.]).cuda())

        # Assume input is B x T x N*C x H x W        
        B, T, C, H, W = x.shape
        N, C = C//3, 3
        x = x.transpose(1,2).view(B, N, C, T, H, W)

        ff, mm = self.pixels_to_nodes(x)
        B, C, T, N = ff.shape

        A12s = []
        A21s = []
        AAs  = []

        # produce A between all pairs of frames, store A for adjacent frames
        t_pairs = list(itertools.combinations(range(T), 2))
        L = len(t_pairs)

        for (t1, t2) in t_pairs:
            f1, f2 = ff[:, :, t1], ff[:, :, t2]

            A, AA, log_AA, A12, A21 = self.compute_affinity(f1, f2)
            log_AA = log_AA.view(-1, log_AA.shape[-1])

            xent_loss, acc = self.compute_xent_loss(A, log_AA)
            kldv_loss = self.compute_kldv_loss(A, log_AA)

            xents += xent_loss
            kldvs += kldv_loss
            diags['skip_accur'] += acc/L

            if (t2 - t1) == 1:
                A12s.append(A12)
                A21s.append(A21)
                AAs.append(AA)

            # _AA = AA.view(-1, H * W, H, W)
            if np.random.random() < 0.003:
                self.viz.text('%s %s' % (t1, t2), opts=dict(height=1, width=10000), win='div')
                self.visualize_frame_pair(x, ff, mm, t1, t2)


        # longer cycle:
        a12, a21 = A12s[0], A21s[0]
        for i in range(1, len(A12s)):
            a12, a21 = torch.matmul(A12s[i], a12), torch.matmul(a21, A21s[i])
            aa = torch.matmul(a21, a12)
            log_aa = torch.log(aa + 1e-20).view(-1, aa.shape[-1])

            xent_loss, acc = self.compute_xent_loss(aa, log_aa)
            kldv_loss = self.compute_kldv_loss(aa, log_aa)

            xents += xent_loss
            kldvs += kldv_loss
            diags['acc cyc %s' % str(i)] = acc
            diags['xent cyc %s' % str(i)] = xent_loss.mean().detach()


        if np.random.random() < 0.01:
            # all patches
            all_x = x.permute(0, 3, 1, 2, 4, 5)
            all_x = all_x.reshape(-1, *all_x.shape[-3:])
            all_f = ff.permute(0, 2, 3, 1).reshape(-1, ff.shape[1])
            all_f = all_f.reshape(-1, *all_f.shape[-1:])
            all_A = torch.einsum('ij,kj->ik', all_f, all_f)

            utils.nn_patches(self.viz, all_x, all_A[None])


        return ff, self.xent_coef * (xents/L), self.kldv_coef * (kldvs/L), diags


