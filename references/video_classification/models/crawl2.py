import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter

import torchvision
import itertools

import time
import numpy as np
import cv2
import visdom
import utils

import kornia
import kornia.augmentation as K

from matplotlib import cm
color = cm.get_cmap('winter')

EPS = 1e-20

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class CRaWl(nn.Module):
    def garg(self, k, d):
        return getattr(self.args, k, d)
        
    def __init__(self, args, vis=None):
        super(CRaWl, self).__init__()
        
        self.args = args

        self.zero_diagonal = getattr(args, 'zero_diagonal', 0)
        self.edgedrop_rate = getattr(args, 'dropout', 0)
        self.featdrop_rate = getattr(args, 'featdrop', 0)
        self.model_type = getattr(args, 'model_type', 'scratch')
        self.temperature = getattr(args, 'temp', getattr(args, 'temperature',1))

        self.encoder = utils.make_encoder(args).to(self.args.device)
        self.infer_dims()

        self.selfsim_fc = self.make_head(depth=self.garg('head_depth', 0))

        self.kldv = torch.nn.KLDivLoss(reduction="batchmean")
        self.xent = torch.nn.CrossEntropyLoss(reduction="none")

        self.target_temp = 1

        self._xent_targets = {}
        self._kldv_targets = {}
        
        if self.garg('restrict', 0) > 0:
            self.restrict = utils.MaskedAttention(int(args.restrict))
        else:
            self.restrict =  None

        self.dropout = torch.nn.Dropout(p=self.edgedrop_rate, inplace=False)
        self.featdrop = torch.nn.Dropout(p=self.featdrop_rate, inplace=False)

        self.viz = visdom.Visdom(port=8095, env='%s_%s' % (getattr(args, 'name', 'test'), '')) #int(time.time())))
        self.viz.close()

        if not self.viz.check_connection():
            self.viz = None

        if vis is not None:
            self._viz = vis
    

    def infer_dims(self):
        in_sz = 256
        dummy = torch.zeros(1, 3, 1, in_sz, in_sz).to(next(self.encoder.parameters()).device)
        dummy_out = self.encoder(dummy)
        self.enc_hid_dim = dummy_out.shape[1]
        self.map_scale = in_sz // dummy_out.shape[-1]

    def make_head(self, depth=1):
        head = []

        if depth >= 0:
            dims = [self.enc_hid_dim] + [self.enc_hid_dim] * depth + [128]
            for d1, d2 in zip(dims, dims[1:]):
                h = nn.Linear(d1, d2)
                head += [h, nn.ReLU()]
            head = head[:-1]

        return nn.Sequential(*head)

    def zeroout_diag(self, A, zero=0):
        mask = (torch.eye(A.shape[-1]).unsqueeze(0).repeat(A.shape[0], 1, 1).bool() < 1).float().cuda()
        return A * mask

    def dropout_mask(self, A):
        return (torch.rand(A.shape) < self.edgedrop_rate).to(self.args.device)

    def compute_affinity(self, x1, x2, do_dropout=True, zero_diagonal=None):
        if do_dropout and self.featdrop_rate > 0:
            x1, x2 = self.featdrop(x1), self.featdrop(x2)
            x1, x2 = F.normalize(x1, dim=1), F.normalize(x2, dim=1)

        if x1.ndim < 4:  # add in time dimension if not there
            x1, x2 = x1.unsqueeze(-2), x2.unsqueeze(-2)

        A = torch.einsum('bctn,bctm->btnm', x1, x2)

        if self.restrict is not None:
            A = self.restrict(A)

        return A.squeeze(1)
    
    def stoch_mat(self, A, zero_diagonal=True, do_dropout=True, do_sinkhorn=False):
        ''' Affinity -> Stochastic Matrix '''

        if (zero_diagonal is not False) and self.zero_diagonal:
            A = self.zeroout_diag(A)

        if do_dropout and self.edgedrop_rate > 0:
            A[self.dropout_mask(A)] = -1e20

        if do_sinkhorn:
            return utils.sinkhorn_knopp((A/self.temperature).exp(), tol=0.01, max_iter=100, verbose=False)

        return F.softmax(A/self.temperature, dim=-1)

    def pixels_to_nodes(self, x):
        ''' 
            pixel maps -> node embeddings 
            Handles cases where input is a list of patches of images (N>1), or list of whole images (N=1)

            Inputs:
                -- 'x' (B x N x C x T x h x w), batch of images
            Outputs:
                -- 'feats' (B x C x T x N), node embeddings
                -- 'maps'  (B x N x C x T x H x W), node feature maps
        '''
        B, N, C, T, h, w = x.shape
        maps = self.encoder(x.flatten(0, 1))
        H, W = maps.shape[-2:]

        if N == 1:  # flatten single image's feature map to get node feature 'maps'
            maps = maps.permute(0, -2, -1, 1, 2).contiguous()
            maps = maps.view(-1, *maps.shape[3:])[..., None, None]
            N, H, W = maps.shape[0] // B, 1, 1

        # compute node embeddings by spatially pooling node feature maps
        feats = maps.sum(-1).sum(-1) / (H*W)
        feats = self.selfsim_fc(feats.transpose(-1, -2)).transpose(-1,-2)
        feats = F.normalize(feats, p=2, dim=1)
    
        feats = feats.view(B, N, feats.shape[1], T).permute(0, 2, 3, 1)
        maps  =  maps.view(B, N, *maps.shape[1:])

        return feats, maps

    def forward(self, x, orig=None, just_feats=False, visualize=False, targets=None):
        # Assume input is B x T x N*C x H x W
        # where N>1 -> list of patches of images and N=1 -> list of images
        B, T, C, H, W = x.shape
        _N, C = C//3, 3
    
        #################################################################
        # Pixels to Nodes 
        #################################################################

        x = x.transpose(1, 2).view(B, _N, C, T, H, W)
        ff, mm = self.pixels_to_nodes(x)
        if just_feats:
            h, w = int(np.ceil(x.shape[-2] / self.map_scale)), int(np.ceil(x.shape[-1] / self.map_scale))
            return ff, mm if _N > 1 else ff, ff.view(*ff.shape[:-1], h, w)

        B, C, T, N = ff.shape

        #################################################################
        # Compute walks 
        #################################################################
        walks = dict()

        # Never drop edges in first and last transition matrices
        As0 = self.compute_affinity(ff[:, :, :1], ff[:, :, 1:2], do_dropout=False)
        AsT = self.compute_affinity(ff[:, :, -2:-1], ff[:, :, -1:], do_dropout=False)
        As  = self.compute_affinity(ff[:, :, 1:-2], ff[:, :, 2:-1])

        As = torch.cat([As0[:,None], As, AsT[:,None]], dim=1)
        A12s = [self.stoch_mat(As[:, i]) for i in range(T-1)]

        #################################################### Palindromes
        if not self.garg('sk_targets', False):  
            A21s = [self.stoch_mat(As[:, i].transpose(-1, -2)) for i in range(T-1)]
            a12, a21 = A12s[0], A21s[0]
            # a12, a21 = torch.eye(A12s[0].shape[-2])[None].cuda(), torch.eye(A21s[0].shape[-1])[None].cuda()

            for i in range(1, len(A12s)):
                # a12 = a12 @ A12s[i]
                # a21 = A21s[i] @ a21
                # aa = a12 @ a21
                a12 = A12s[i] @ a12
                a21 = a21 @ A21s[i]
                aa = a21 @ a12
                walks[f"cyc {i}"] = [aa, self.xent_targets(aa)]
        
        #################################################### Sinkhorn-knopp Target
        else:   
            a12, at = A12s[0], self.stoch_mat(As[:, 0], do_dropout=False, do_sinkhorn=True)

            for i in range(1, len(A12s)):
                # a12 = a12 @ A12s[i]
                # at = at @ self.stoch_mat(As[i], do_dropout=False, do_sinkhorn=True)
                a12 = A12s[i] @ a12
                at = self.stoch_mat(As[:, i], do_dropout=False, do_sinkhorn=True) @ at
                with torch.no_grad():
                    targets = utils.sinkhorn_knopp(at, tol=0.001, max_iter=10, verbose=False).argmax(-1).flatten()
                walks[f"sk {i}"] = [a12, targets]

        #################################################### Skip paths
        if self.garg('skip_coef', 0) > 0:
            t_pairs = [(t1, t2) for (t1, t2) in list(itertools.combinations(range(T), 2)) if (t2-t1) > 1]
            As  = [self.compute_affinity(ff[:, :, t1], ff[:, :, t2]) for (t1, t2) in t_pairs]
            A1s = [self.stoch_mat(As[i]) for i in range(T-1)]
            A2s = [self.stoch_mat(As[i].transpose(-1, -2)) for i in range(T-1)]
            AA  = [a@b for (a,b) in zip(A1s, A2s)]

            for i, aa in enumerate(AA):
                walks[f"skip {i}"] = [aa, self.xent_targets(aa)]

        #################################################################
        # Compute loss 
        #################################################################
        xents = [torch.tensor([0.]).to(self.args.device)]
        kldvs = [torch.tensor([0.]).to(self.args.device)]
        diags = dict(skip_accur=torch.tensor([0.]).to(self.args.device))

        for name, (A, target) in walks.items():
            loss, acc = self.compute_xent_loss(A, torch.log(A + EPS).flatten(0, -2), targets=target)
            diags.update({f"{H} xent {name}": loss.mean().detach(), f"{H} acc {name}": acc})
            xents += [loss]

        #################################################################
        # Visualizations
        #################################################################

        if (np.random.random() < (0.02) or visualize) and (self.viz is not None): # and False:
            t1, t2 = np.random.randint(0, T, (2))
            with torch.no_grad():
                self.visualize_frame_pair(x, ff, mm, t1, t2)

                if _N > 1: # and False:
                    # all patches
                    all_x = x.permute(0, 3, 1, 2, 4, 5)
                    all_x = all_x.reshape(-1, *all_x.shape[-3:])
                    all_f = ff.permute(0, 2, 3, 1).reshape(-1, ff.shape[1])
                    all_f = all_f.reshape(-1, *all_f.shape[-1:])
                    all_A = torch.einsum('ij,kj->ik', all_f, all_f)

                    utils.visualize.nn_patches(self.viz, all_x, all_A[None])

        return ff, sum(xents)/max(1, len(xents)-1), 0 * sum(kldvs)/max(1, len(kldvs)-1), diags


    def xent_targets(self, A):
        B, N = A.shape[:2]
        key = '%s:%sx%s' % (str(A.device), B,N)

        if key not in self._xent_targets:
            I = torch.arange(A.shape[-1])[None].repeat(B, 1)
            self._xent_targets[key] = I.view(-1).to(A.device)

        return self._xent_targets[key]

    def compute_xent_loss(self, A, log_AA, weight=None, targets=None):
        # Cross Entropy
        if targets is None:
            targets = self.xent_targets(A)

        _xent_loss = self.xent(log_AA, targets)

        if weight is not None:
            _xent_loss = _xent_loss * weight.flatten()

        return _xent_loss.mean().unsqueeze(-1), \
            (torch.argmax(log_AA, dim=-1) == targets).float().mean().unsqueeze(-1)


    def loss_mask(self, A, P):
        '''
            P is row-normalized, process stochastic matrix of affinity A
        '''
        entropy = (-P * torch.log(P)).sum(-1) / np.log(P.shape[-1])
        # maxsim = A.max(-1)[0]
        maxsim = A.topk(k=2, dim=-1)[0][..., -1]
        xent_weight = 1 / (maxsim * entropy)

        xent_weight = torch.clamp(xent_weight, min=0, max=5)
        return xent_weight

    def visualize_frame_pair(self, x, ff, mm, t1, t2):
        normalize = lambda x: (x-x.min()) / (x-x.min()).max()

        # B, C, T, N = ff.shape
        f1, f2 = ff[:, :, t1], ff[:, :, t2]

        A = self.compute_affinity(f1, f2)
        A1  = self.stoch_mat(A, False, False)
        A2 = self.stoch_mat(A.transpose(-1, -2), False, False)
        AA = A1 @ A2
        log_AA = torch.log(AA + EPS)

        log_AA = log_AA.view(-1, log_AA.shape[1])
        _xent_loss = self.xent(log_AA, self.xent_targets(A))

        N = A.shape[-1]
        H = W = int(N**0.5)
        _AA = AA.view(-1, H * W, H, W)

        ##############################################
        ## LOSS MASK VISUALIZATION
        ##############################################
        
        def mask_to_img(m):
            return cv2.resize(m.view(H, W).detach().cpu().numpy(), (280,280))

        xent_weight = self.loss_mask(A, AA).detach()        
        xent_weight_img = mask_to_img(xent_weight[0] / 5.0)

        entropy = (-AA * torch.log(AA)).sum(-1) / np.log(AA.shape[-1])
        entropy_img = mask_to_img(entropy[0])

        maxsim = A.topk(k=2, dim=-1)[0][..., -1][0]
        maxsim_img = mask_to_img(maxsim)

        self.viz.images(np.stack([xent_weight_img, entropy_img, maxsim_img])[:, None], nrow=3, win='loss_mask_viz')

        ##############################################
        ## FLOW
        ##############################################

        # _A = A.view(*A.shape[:2], f1.shape[-1], -1)
        # u, v = utils.compute_flow(_A[0:1])
        # flows = torch.stack([u, v], dim=-1).cpu().numpy()
        # flows = utils.draw_hsv(flows[0])
        # flows = cv2.resize(flows, (256, 256))
        # self.viz.image((flows).transpose(2, 0, 1), win='flow')
        # # flows = [cv2.resize(flow.clip(min=0).astype(np.uint8), (256, 256)) for flow in flows]
        # # self.viz.image((flows[0]).transpose(2, 0, 1))

        # import pdb; pdb.set_trace()
        if (len(x.shape) == 6 and x.shape[1] == 1):
            x = x.squeeze(1)

        if len(x.shape) < 6:
            # IMG VIZ
            # X here is B x C x T x H x W
            x1, x2 = x[0, :, t1].clone(), x[0, :, t2].clone()
            # x1 -= x1.min(); x1 /= x1.max()
            # x2 -= x2.min(); x2 /= x2.max()

            x1, x2 = normalize(x1), normalize(x2)

            xx = torch.stack([x1, x2]).detach().cpu()
            self.viz.images(xx, win='imgs')
            # self._viz.patches(xx, A)

            # Keypoint Correspondences
            kp_corr = utils.draw_matches(f1[0], f2[0], x1, x2)

            self.viz.image(kp_corr, win='kpcorr')

            # # PCA VIZ
            spatialize = lambda xx: xx.view(*xx.shape[:-1], int(xx.shape[-1]**0.5), int(xx.shape[-1]**0.5))
            ff1 , ff2 = spatialize(f1[0]), spatialize(f2[0])
            pca_ff = utils.visualize.pca_feats(torch.stack([ff1,ff2]).detach().cpu())
            pca_ff = utils.visualize.make_gif(pca_ff, outname=None)
            self.viz.images(pca_ff.transpose(0, -1, 1, 2), win='pcafeats')
        else:
            # X here is B x N x C x T x H x W
            x1, x2 =  x[0, :, :, t1],  x[0, :, :, t2]
            m1, m2 = mm[0, :, :, t1], mm[0, :, :, t2]

            pca_feats = utils.visualize.pca_feats(torch.cat([m1, m2]).detach().cpu())
            pca_feats = utils.visualize.make_gif(pca_feats, outname=None, sz=64).transpose(0, -1, 1, 2)
            
            pca1 = torchvision.utils.make_grid(torch.Tensor(pca_feats[:N]), nrow=int(N**0.5), padding=1, pad_value=1)
            pca2 = torchvision.utils.make_grid(torch.Tensor(pca_feats[N:]), nrow=int(N**0.5), padding=1, pad_value=1)
            img1 = torchvision.utils.make_grid(normalize(x1)*255, nrow=int(N**0.5), padding=1, pad_value=1)
            img2 = torchvision.utils.make_grid(normalize(x2)*255, nrow=int(N**0.5), padding=1, pad_value=1)
            self.viz.images(torch.stack([pca1,pca2]), nrow=4, win='pca_viz_combined1')
            self.viz.images(torch.stack([img1.cpu(),img2.cpu()]), nrow=4, win='pca_viz_combined2')
        
        ##############################################
        # LOSS VIS
        ##############################################
        
        xx = normalize(_xent_loss[:H*W])
        img_grid = [cv2.resize(aa, (50,50), interpolation=cv2.INTER_NEAREST)[None] for aa in _AA[0, :, :, :, None].cpu().detach().numpy()]
        img_grid = [img_grid[_].repeat(3, 0) * np.array(color(xx[_].item()))[:3, None, None] for _ in range(H*W)]
        img_grid = [img_grid[_] / img_grid[_].max() for _ in range(H*W)]
        img_grid = torch.from_numpy(np.array(img_grid))
        img_grid = torchvision.utils.make_grid(img_grid, nrow=H, padding=1, pad_value=1)
        
        # img_grid = cv2.resize(img_grid.permute(1, 2, 0).cpu().detach().numpy(), (1000, 1000), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
        self.viz.images(img_grid, win='lossvis')

