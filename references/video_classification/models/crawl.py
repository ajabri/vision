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

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class CRaWl(nn.Module):
    def garg(self, k, d):
        return getattr(self.args, k) if hasattr(self.args, k) else d
        
    def __init__(self, args=None, vis=None):
        super(CRaWl, self).__init__()
        
        self.args = args

        if args is not None:
            self.kldv_coef = getattr(args, 'kldv_coef', 0)
            self.xent_coef = getattr(args, 'xent_coef', 0)
            self.zero_diagonal = getattr(args, 'zero_diagonal', 0)
            self.dropout_rate = getattr(args, 'dropout', 0)
            self.featdrop_rate = getattr(args, 'featdrop', 0)
            self.model_type = getattr(args, 'model_type', 'scratch')
            self.temperature = getattr(args, 'temp', getattr(args, 'temperature',1))
            self.shuffle = getattr(args, 'shuffle', 0)
            self.xent_weight = getattr(args, 'xent_weight', False)
        else:
            self.kldv_coef = 0
            self.xent_coef = 0
            self.zero_diagonal = 0
            self.dropout_rate = 0
            self.featdrop_rate = 0
            self.model_type = 'scratch'
            self.temperature = 1
            self.shuffle = False
            self.xent_weight = False

        print('Model temp:', self.temperature)
        self.encoder = utils.make_encoder(args).to(self.args.device)


        self.infer_dims()

        self.selfsim_fc = self.make_head(depth=self.garg('head_depth', 0))
        self.selfsim_head = self.make_conv3d_head(depth=1)
        self.context_head = self.make_conv3d_head(depth=1)

        # self.kldv = torch.nn.KLDivLoss(reduction="batchmean")
        self.kldv = torch.nn.KLDivLoss(reduction="batchmean")
        self.xent = torch.nn.CrossEntropyLoss(reduction="none")

        self.target_temp = 1
        self._n_batches = 0

        self._xent_targets = {}
        self._kldv_targets = {}
        
        if self.garg('restrict', 0) > 0:
            self.restrict = utils.MaskedAttention(int(args.restrict))
        else:
            self.restrict =  None

        self.dropout = torch.nn.Dropout(p=self.dropout_rate, inplace=False)
        self.featdrop = torch.nn.Dropout(p=self.featdrop_rate, inplace=False)

        self.viz = visdom.Visdom(port=8095, env='%s_%s' % (getattr(args, 'name', 'test'), '')) #int(time.time())))
        self.viz.close()

        if not self.viz.check_connection():
            self.viz = None

        if vis is not None:
            self._viz = vis
    
        p_sz, stride = 64, 32
        self.k_patch =  nn.Sequential(
            K.RandomResizedCrop(size=(p_sz, p_sz), scale=(0.7, 0.9), ratio=(0.7, 1.3))
        )
        # import pdb; pdb.set_trace()
        self.k_frame = nn.Sequential(
            # K.ColorJitter(0.1, 0.1, 0.1, 0),
            # K.
            # K.Normalize()
            K.RandomResizedCrop(size=(256, 256), scale=(0.8, 0.9), ratio=(0.7, 1.3))
        )
        # self.k_frame_same = nn.Sequential(
        #     K.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), same_on_batch=True)
        # )
        # self.k_frame_same = None
        self.k_frame_same = nn.Sequential(
            kornia.geometry.transform.Resize(256 + 20),
            kornia.augmentation.RandomCrop((256, 256), same_on_batch=True),
        )

        self.unfold = torch.nn.Unfold((p_sz,p_sz), dilation=1, padding=0, stride=(stride, stride))

    def infer_dims(self):
        in_sz = 256
        dummy = torch.zeros(1, 3, 1, in_sz, in_sz).to(next(self.encoder.parameters()).device)
        dummy_out = self.encoder(dummy)

        self.enc_hid_dim = dummy_out.shape[1]
        self.map_scale = in_sz // dummy_out.shape[-1]

    def make_head(self, depth=1):
        head = []

        if depth == -1:
            return Identity()

        if depth == 0:
            return nn.Linear(self.enc_hid_dim, 128)

        dims = [self.enc_hid_dim] + [self.enc_hid_dim] * depth + [128]

        for d1, d2 in zip(dims, dims[1:]):
            # h = nn.Conv3d(d1, d2, kernel_size=1, bias=True)
            # nn.init.kaiming_normal_(h.weight, mode='fan_out', nonlinearity='relu')
            h = nn.Linear(d1, d2)
            
            h.weight.data.normal_(0, 0.01)
            h.bias.data.zero_()

            head += [h, nn.ReLU()]

        head = head[:-1]
        head = nn.Sequential(*head)
        return head

        # return 

    def make_conv3d_head(self, depth=1):
        head = []

        dims = [self.enc_hid_dim] + [2*self.enc_hid_dim] * depth + [256]

        for d1, d2 in zip(dims, dims[1:]):
            h = nn.Conv3d(d1, d2, kernel_size=1, bias=True)
            nn.init.kaiming_normal_(h.weight, mode='fan_out', nonlinearity='relu')
            head += [h, nn.ReLU()]

        head = head[:-1]
        head = nn.Sequential(*head)
        return head

        # return 

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
        return A * mask

    def dropout_mask(self, A):
        return (torch.rand(A.shape) < self.dropout_rate).to(self.args.device)

    def compute_affinity(self, x1, x2, do_dropout=True, zero_diagonal=None):
        if do_dropout and self.featdrop_rate > 0:
            x1, x2 = self.featdrop(x1), self.featdrop(x2)
            x1, x2 = F.normalize(x1, dim=1), F.normalize(x2, dim=1)

        A = torch.einsum('bcn,bcm->bnm', x1, x2)

        if self.restrict is not None:
            A = self.restrict(A)

        return A
    
    def stoch_mat(self, A, zero_diagonal=True, do_dropout=True):
        ''' Affinity -> Stochastic Matrix '''

        if (zero_diagonal is not False) and self.zero_diagonal:
            A = self.zeroout_diag(A)

        if do_dropout and self.dropout_rate > 0:
            mask = self.dropout_mask(A)
            A[mask] = -1e20
        A = F.softmax(A/self.temperature, dim=-1)

        return A

    def pixels_to_nodes(self, x):
        '''
            pixel map to list of node embeddings
        '''

        # Encode into B x C x T x N
        B, N, C, T, h, w = x.shape
        x = x.reshape(B*N, C, T, h, w) 

        mm = self.encoder(x)

        # HACK: Attempted spatial dropout when trying to hack dense feature setting
        # _mm = torch.flatten(mm.transpose(1,2), start_dim=0, end_dim=1)
        # mm = nn.functional.dropout2d(_mm, p=self.featdrop_rate).view(B, T, *_mm.shape[1:]).transpose(1, 2)

        H, W = mm.shape[-2:]

        if N == 1:  # encode 1 image into feature map, use those vecs as nodes
            # mm = mm[..., ::2, ::2]
            # import pdb; pdb.set_trace()

            mm = mm.permute(0, -2, -1, 1, 2).contiguous()
            mm = mm.view(-1, *mm.shape[3:])[..., None, None]
            N = mm.shape[0] // B
            H, W = 1, 1

        # import pdb; pdb.set_trace()

        # produce node vector representations by spatially pooling feature maps
        # mm = F.normalize(mm, dim=1)
        ff = mm.sum(-1).sum(-1) / (H*W)
        # ff = torch.einsum('ijklm->ijk', ff) / ff.shape[-1]*ff.shape[-2] 

        ff = self.selfsim_fc(ff.transpose(-1, -2)).transpose(-1,-2)
        ff = F.normalize(ff, p=2, dim=1)
    
        # reshape to add back batch and num node dimensions
        ff = ff.view(B, N, ff.shape[1], T).permute(0, 2, 3, 1)
        mm = mm.view(B, N, *mm.shape[1:])

        return ff, mm

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

        # import pdb; pdb.set_trace()

    def forward(self, x, orig=None, just_feats=False, visualize=False, targets=None, unfold=False):
        xents = [torch.tensor([0.]).to(self.args.device)]
        kldvs = [torch.tensor([0.]).to(self.args.device)]
        diags = dict(skip_accur=torch.tensor([0.]).to(self.args.device))

        # Assume input is B x T x N*C x H x W        
        B, T, C, H, W = x.shape
        _N, C = C//3, 3

        # x = F.dropout(x)
    
        if _N == 1 and (visualize and False or unfold):
            # patchify with unfold
            # _sz, res = 80, 10
            # stride = utils.get_stride(H, _sz, res)
            _sz = self.unfold.kernel_size[0]
            x = x.flatten(0, 1)

            # if self.k_frame_same is None:
            #     import pdb; pdb.set_trace()

            x = self.k_frame_same(x)
            x = self.k_frame(x)
            x = self.unfold(x)

            x, _N = x.view(B, T, C, _sz, _sz, x.shape[-1]), x.shape[-1]
            x = x.permute(0, -1, 1, 2, 3, 4)   # B x _N x T x C x H x W
            x = x.flatten(0, 2)

            x = self.k_patch(x)
            x = x.view(B, _N, T, C, _sz, _sz).transpose(2, 3)
            # import pdb; pdb.set_trace()
            x = x.to(self.args.device)
        else:
            # import pdb; pdb.set_trace()
            x = x.transpose(1,2).view(B, _N, C, T, H, W)

        if self.shuffle > np.random.random():
            shuffle = torch.randperm(B*_N)
            x = x.reshape(B*_N, C, T, H, W)[shuffle]
            x = x.view(B, _N, C, T, H, W)
            
        ff, mm = self.pixels_to_nodes(x)

        # _ff = ff.view(*ff.shape[:-1], h, w)
        if just_feats:
            h, w = int(np.ceil(x.shape[-2] / self.map_scale)), int(np.ceil(x.shape[-1] / self.map_scale))
            if _N > 1:
                return ff, mm
            else:
                return ff, ff.view(*ff.shape[:-1], h, w)

        B, C, T, N = ff.shape

        A12s = []
        A21s = []
        AAs  = []
        As = []

        # produce A between all pairs of frames, store A for adjacent frames
        t_pairs = list(itertools.combinations(range(T), 2))
        if not self.garg('skip_coef', 0) > 0:
            t_pairs = [(t1, t2) for (t1, t2) in t_pairs if (t2-t1) == 1]
        
        L = len(t_pairs)

        #################################################################
        # PatchGraph
        #################################################################
        
        if visualize and self.viz is not None: #np.random.random() < 0.05 and visualize:
            with torch.no_grad():

                if ff.device.index == 0:
                    for i in range(B):
                        pg_win = 'patchgraph_%s'%i
                        # print('exists', self.viz.win_exists(pg_win, env=self.viz.env+'_pg'))
                        if not self.viz.win_exists(pg_win, env=self.viz.env+'_pg') or visualize:
                            tviz = 0
                            self.viz.clear_event_handlers(pg_win)
                            fff = ff[i].transpose(0,1)
                            A_traj = self.compute_affinity(ff[i].transpose(0,1)[:-1] , ff[i].transpose(0,1)[1:], do_dropout=False)

                            A_t = [self.stoch_mat(A_traj[0])]
                            for A_tp1 in A_traj[1:]:
                                A_t.append(self.stoch_mat(A_tp1) @ A_t[-1])

                            # A_t = [F.softmax(A_traj[0], dim=-1)]
                            # for A_tp1 in A_traj[1:]:
                            #     A_t.append(F.softmax(A_tp1, dim=-1) @ A_t[-1])

                            A_t = torch.stack(A_t)
                            pg = utils.PatchGraph(x[i, :, :].transpose(1, 2),
                                A_t, viz=self.viz,win=pg_win, orig=orig) 

        #################################################################
        # Build all graph once!
        #################################################################

        if len(t_pairs) > 0:
            for (t1, t2) in t_pairs:
                f1, f2 = ff[:, :, t1], ff[:, :, t2]

                A = self.compute_affinity(f1, f2)
                A1, A2 = self.stoch_mat(A), self.stoch_mat(A.transpose(-1, -2))
                AA = A1 @ A2

                if self.garg('skip_coef', 0) > 0:
                    
                    log_AA = torch.log(AA + 1e-20).view(-1, AA.shape[-1])
                    
                    xent_loss, acc = self.compute_xent_loss(A, log_AA)
                    kldv_loss = self.compute_kldv_loss(A, log_AA)

                    # xents += xent_loss
                    # kldvs += kldv_loss
                    xents.append(xent_loss)
                    kldvs.append(kldv_loss)
                    diags['skip_accur'] += acc
                
                if t2 - t1 == 1:
                    As.append(A)
                    A12s.append(A1)
                    A21s.append(A2)
                    AAs.append(AA)

                if (np.random.random() < (0.05 / len(t_pairs)) or visualize) and (self.viz is not None): # and False:
                    self.viz.text('%s %s' % (t1, t2), opts=dict(height=1, width=10000), win='div')
                    with torch.no_grad():
                        self.visualize_frame_pair(x, ff, mm, t1, t2)


            # longer cycle:
            if self.garg('long_coef', 0) > 0:
                a12, a21 = A12s[0], A21s[0]
                # a12, a21 = torch.eye(A12s[0].shape[-2])[None].cuda(), torch.eye(A21s[0].shape[-1])[None].cuda()

                for i in range(1, len(A12s)):
                    # a12 = a12 @ A12s[i]
                    # a21 = A21s[i] @ a21
                    # aa = a12 @ a21

                    a12 = A12s[i] @ a12
                    a21 = a21 @ A21s[i]
                    aa = a21 @ a12

                    log_aa = torch.log(aa + 1e-20).view(-1, aa.shape[-1])

                    xent_weight = None
                    if self.garg('xent_weight', False):
                        xent_weight = self.loss_mask(As[i], aa).detach()
                        # print(xent_weight.min(), xent_weight.max())
                        # import pdb; pdb.set_trace()

                    xent_loss, acc = self.compute_xent_loss(aa, log_aa, weight=xent_weight)
                    kldv_loss = self.compute_kldv_loss(aa, log_aa, targets=targets)

                    if targets is not None:
                        xent_loss*=0

                    xents.append(xent_loss)
                    kldvs.append(kldv_loss)
                    
                    diags['acc cyc %s' % str(i)] = acc

                    if targets is not None:
                        diags['kl cyc %s' % str(i)] = kldv_loss.mean().detach()
                    else:
                        diags['xent cyc %s' % str(i)] = xent_loss.mean().detach()

            
        if _N > 1 and (np.random.random() < (0.01 / len(t_pairs)) or visualize) and self.viz is not None: # and False:
            # all patches
            all_x = x.permute(0, 3, 1, 2, 4, 5)
            all_x = all_x.reshape(-1, *all_x.shape[-3:])
            all_f = ff.permute(0, 2, 3, 1).reshape(-1, ff.shape[1])
            all_f = all_f.reshape(-1, *all_f.shape[-1:])
            all_A = torch.einsum('ij,kj->ik', all_f, all_f)

            with torch.no_grad():
                utils.visualize.nn_patches(self.viz, all_x, all_A[None])

        diag_keys = list(diags.keys())
        for k in diag_keys:
            diags["%s %s" % (H, k)] = diags[k]
            del diags[k]

        self._n_batches += 1

        return ff, self.xent_coef * sum(xents)/max(1, len(xents)-1), self.kldv_coef * sum(kldvs)/max(1, len(kldvs)-1), diags


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

    def compute_xent_loss(self, A, log_AA, weight=None, targets=None):
        # Cross Entropy
        if targets is None:
            targets = self.xent_targets(A)

        if self.xent_coef > 0:
            _xent_loss = self.xent(log_AA, targets)

            if weight is not None:
                _xent_loss = _xent_loss * weight.flatten()

            return _xent_loss.mean().unsqueeze(-1), \
                (torch.argmax(log_AA, dim=-1) == targets).float().mean().unsqueeze(-1)
        else:
            return 0, 0

    def compute_kldv_loss(self, A, log_AA, targets=None):
        # KL Div with Smoothed 2D Targets
        if targets is None:
            targets = self.kldv_targets(A)
            
        if self.kldv_coef > 0:
            kldv_loss = self.kldv(log_AA, targets)
            # print(kldv_loss, log_AA.min(), AA.min(), A.min())
            return kldv_loss
        else:
            return 0

    def visualize_frame_pair(self, x, ff, mm, t1, t2):
        normalize = lambda x: (x-x.min()) / (x-x.min()).max()

        # B, C, T, N = ff.shape
        f1, f2 = ff[:, :, t1], ff[:, :, t2]

        A = self.compute_affinity(f1, f2)
        A1  = self.stoch_mat(A, False, False)
        A2 = self.stoch_mat(A.transpose(-1, -2), False, False)
        AA = A1 @ A2
        log_AA = torch.log(AA + 1e-20)

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

