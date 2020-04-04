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

class RestrictAttention(nn.Module):
    def __init__(self, radius, flat=True):
        super(RestrictAttention, self).__init__()
        self.radius = radius
        self.flat = flat
        self.masks = {}
        self.masks['10-10'] = self.make(10, 10)

    def make(self, H, W):
        if self.flat:
            H = int(H**0.5)
            W = int(W**0.5)
        
        gx, gy = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        D = ( (gx[None, None, :, :] - gx[:, :, None, None])**2 + (gy[None, None, :, :] - gy[:, :, None, None])**2 ).float() ** 0.5
        D = (D < self.radius)[None].float()

        if self.flat:
            D = torch.flatten(torch.flatten(D, 1, 2), -2, -1)

        return D

    def forward(self, x):
        H, W = x.shape[-2:]
        sid = '%s-%s' % (H,W)
        if sid not in self.masks:
            self.masks[sid] = self.make(H, W).to(x.device)
        mask = self.masks[sid]

        return x * mask[0]
        # import pdb; pdb.set_trace()
        # x 



class TimeCycle(nn.Module):
    def garg(self, k, d):
        return getattr(self.args, k) if hasattr(self.args, k) else d
        
    def __init__(self, args=None, vis=None):
        super(TimeCycle, self).__init__()
        
        self.args = args

        if args is not None:
            self.kldv_coef = getattr(args, 'kldv_coef', 0)
            self.xent_coef = getattr(args, 'xent_coef', 0)
            self.zero_diagonal = getattr(args, 'zero_diagonal', 0)
            self.dropout_rate = getattr(args, 'dropout', 0)
            self.featdrop_rate = getattr(args, 'featdrop', 0)
            self.model_type = getattr(args, 'model_type', 'scratch')
            self.temperature = getattr(args, 'temp', 1)
            self.shuffle = getattr(args, 'shuffle', 0)
        else:
            self.kldv_coef = 0
            self.xent_coef = 0
            self.zero_diagonal = 0
            self.dropout_rate = 0
            self.featdrop_rate = 0
            self.model_type = 'scratch'
            self.temperature = 1
            self.shuffle = False
        
        self.encoder = utils.make_encoder(self.model_type).cuda()

        self.infer_dims()

        self.selfsim_fc = self.make_head(depth=self.garg('head_depth', 0))
        self.selfsim_head = self.make_conv3d_head(depth=1)
        self.context_head = self.make_conv3d_head(depth=1)

        # self.selfsim_head = self.make_head([self.enc_hid_dim, 2*self.enc_hid_dim, self.enc_hid_dim])
        # self.context_head = self.make_head([self.enc_hid_dim, 2*self.enc_hid_dim, self.enc_hid_dim])

        import resnet3d, resnet2d
        if self.garg('cal_coef', 0) > 0:
            self.stack_encoder = utils.make_stack_encoder(self.enc_hid_dim)
            # self.aff_encoder = resnet2d.Bottleneck(1, 128,)

        # # assuming no fc pre-training
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data.normal_(0, 0.01)
        #         m.bias.data.zero_()

        self.edge = getattr(args, 'edgefunc', 'softmax')

        # self.kldv = torch.nn.KLDivLoss(reduction="batchmean")
        self.kldv = torch.nn.KLDivLoss(reduction="batchmean")
        self.xent = torch.nn.CrossEntropyLoss(reduction="none")

        self.target_temp = 1

        self._xent_targets = {}
        self._kldv_targets = {}
        
        if self.garg('restrict', 0) > 0:
            self.restrict = RestrictAttention(int(args.restrict))
        else:
            self.restrict =  None 

        self.dropout = torch.nn.Dropout(p=self.dropout_rate, inplace=False)
        self.featdrop = torch.nn.Dropout(p=self.featdrop_rate, inplace=False)

        self.viz = visdom.Visdom(port=8095, env='%s_%s' % (getattr(args, 'name', 'test'), '')) #int(time.time())))
        self.viz.close()

        if vis is not None:
            self._viz = vis
    
    def infer_dims(self):
        # if '2D' in str(type(self.encoder.conv1)):
        in_sz = 256
        dummy = torch.zeros(1, 3, 1, in_sz, in_sz).to(next(self.encoder.parameters()).device)
        # else:
        #     dummy = torch.Tensor(1, 3, 224, 224)
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
        # import pdb; pdb.set_trace()
        return A * mask

    def compute_affinity(self, x1, x2, do_dropout=True, zero_diagonal=None):
        B, C, N = x1.shape
        H = int(N**0.5)
        # assert x1.shape == x2.shape

        if do_dropout:
            x1, x2 = self.featdrop(x1), self.featdrop(x2)
            x1, x2 = F.normalize(x1, dim=1), F.normalize(x2, dim=1)

        # import pdb; pdb.set_trace()
        # import time
        t0 = time.time()
        
        # fast mm, pretty
        A = torch.einsum('bcn,bcm->bnm', x1, x2)
        # t1 = time.time()

        # # more flexible
        # from spatial_correlation_sampler import spatial_correlation_sample
        # # A_s = self.scorr(x1.view(B, C, int(N**0.5), int(N**0.5)).contiguous(), x2.view(B, C, int(N**0.5), int(N**0.5)).contiguous())     

        # utils.nn_field(A.view(*A.shape[:2], H, H))

        # import pdb; pdb.set_trace()

        # xx1, xx2 = x1.view(B, C, H, H).contiguous(), x2.view(B, C, H, H).contiguous()
        # A_s = spatial_correlation_sample(xx1, xx2, 1, H, 1, 0, 1)
                            #    kernel_size=1,
                            #    patch_size=H,
                            #    stride=1,
                            #    padding=0,
                            #    dilation_patch=1)

        # # cc = utils.Correlation(pad_size=0, kernel_size=1, max_displacement=H, stride1=1, stride2=1, corr_multiply=1)
        # # A_s = cc(x1.view(B, C, H, H).contiguous(), x2.view(B, C, H, H).contiguous())

        # t2 = time.time()
        # print(t2-t1, t1-t0)
        # import pdb; pdb.set_trace()

        if self.restrict is not None: #: and do_dropout:
            A = self.restrict(A)

        if (zero_diagonal is not False) and self.zero_diagonal:
            A = self.zeroout_diag(A)

        # if self.a_topk:
        #     A = sel
        # A12 = A.view(A.size(0), 1, H * H, W, W)
        # A21 = A.view(A.size(0), 1, H, H, W * W) 
        # A12  = F.softmax(A, dim=2)
        # A21  = F.softmax(A.transpose(1, 2), dim=2)

        A1, A2 = A, A.transpose(1, 2)        
        if do_dropout:
            A1, A2 = self.dropout(A1), self.dropout(A2)
    
        if self.edge == 'softmax':
            A1, A2 = F.softmax(A1/self.temperature, dim=-1), F.softmax(A2/self.temperature, dim=-1)
        else:
            if not hasattr(self, 'graph_bias'):
                self.graph_bias = nn.Parameter((torch.ones(*A.shape[-2:])) * 1e-2).to(next(self.encoder.parameters()).device)

            A1, A2 = F.normalize(F.relu(A1 + self.graph_bias)**2, dim=-1, p=1), F.normalize(F.relu(A2 + self.graph_bias.transpose(0,1))**2, dim=-1, p=1)

        AA = torch.matmul(A2, A1)
        log_AA = torch.log(AA + 1e-20)

        return A, AA, log_AA, A1, A2
    

    def pixels_to_nodes(self, x):
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
        ff = mm.sum(-1).sum(-1) / (H*W)
        # ff = torch.einsum('ijklm->ijk', ff) / ff.shape[-1]*ff.shape[-2] 

        ff = self.selfsim_fc(ff.transpose(-1, -2)).transpose(-1,-2)
        ff = F.normalize(ff, p=2, dim=1)
    
        # reshape to add back batch and num node dimensions
        ff = ff.view(B, N, ff.shape[1], T).permute(0, 2, 3, 1)
        mm = mm.view(B, N, *mm.shape[1:])

        return ff, mm


    def forward(self, x, orig=None, just_feats=False, visualize=False):
        # x = x.cpu() * 0 + torch.randn(x.shape)
        # x = x.cuda()
        # orig = orig.cpu() * 0 + torch.randn(orig.shape)
        # orig = orig.cuda()
        
        xents = [torch.tensor([0.]).cuda()]
        kldvs = [torch.tensor([0.]).cuda()]
        diags = dict(skip_accur=torch.tensor([0.]).cuda())

        # Assume input is B x T x N*C x H x W        
        B, T, C, H, W = x.shape
        _N, C = C//3, 3

        # x = F.dropout(x)
    
        if _N == 1 and visualize:
            # patchify with unfold
            _sz, res = 80, 10
            stride = utils.get_stride(H, _sz, res)
            B, C, H, W = orig.shape

            unfold = torch.nn.Unfold((_sz,_sz), dilation=1, padding=0, stride=(stride, stride))
            x = unfold(orig.view(B, C, H, W))
            x = x.view(B, 1, C, _sz, _sz, x.shape[-1]).permute(0, -1, 2, 1, 3, 4)
            # x = x.cuda()

            ff, mm = [], []
            _bsz = 100
            for i in range(0, x.shape[1], _bsz):
                fff, mmm = self.pixels_to_nodes(x[:, i:i+_bsz].cuda())
                ff.append(fff)
                mm.append(mmm)

            ff = torch.cat(ff, dim=-1)
            mm = torch.cat(mm, dim=1)

            # import pdb; pdb.set_trace()
        else:
            x = x.transpose(1,2).view(B, _N, C, T, H, W)

            if self.shuffle > np.random.random():
                shuffle = torch.randperm(B*_N)
                x = x.reshape(B*_N, C, T, H, W)[shuffle]
                x = x.view(B, _N, C, T, H, W)
                
            ff, mm = self.pixels_to_nodes(x)

        # _ff = ff.view(*ff.shape[:-1], h, w)
        if just_feats:
            h, w = int(np.ceil(x.shape[-2] / self.map_scale)), int(np.ceil(x.shape[-1] / self.map_scale))
            return ff, ff.view(*ff.shape[:-1], h, w)

        B, C, T, N = ff.shape

        A12s = []
        A21s = []
        AAs  = []
        As = []

        # produce A between all pairs of frames, store A for adjacent frames
        t_pairs = list(itertools.combinations(range(T), 2))
        L = len(t_pairs)

        if np.random.random() < 0.05 or visualize:
            if ff.device.index == 0:
                for i in range(B):
                    pg_win = 'patchgraph_%s'%i
                    # print('exists', self.viz.win_exists(pg_win, env=self.viz.env+'_pg'))
                    if not self.viz.win_exists(pg_win, env=self.viz.env+'_pg') or visualize:
                        tviz = 0
                        self.viz.clear_event_handlers(pg_win)
                        A, AA, log_AA, A12, A21 = self.compute_affinity(ff[i:i+1, :, tviz], ff[i:i+1, :, tviz+1], do_dropout=False)
                        pg = utils.PatchGraph(self.viz,
                            x[i, :, :, tviz:tviz+2].transpose(1, 2),
                            A[0], win=pg_win, orig=orig)

        if len(t_pairs) > 0:

            for (t1, t2) in t_pairs:
                f1, f2 = ff[:, :, t1], ff[:, :, t2]

                A, AA, log_AA, A12, A21 = self.compute_affinity(f1, f2)
                log_AA = log_AA.view(-1, log_AA.shape[-1])

                if self.garg('skip_coef', 0) > 0:
                    xent_loss, acc = self.compute_xent_loss(A, log_AA)
                    kldv_loss = self.compute_kldv_loss(A, log_AA)

                    # xents += xent_loss
                    # kldvs += kldv_loss
                    xents.append(xent_loss)
                    kldvs.append(kldv_loss)
                    diags['skip_accur'] += acc
                
                As.append(A12)
                if (t2 - t1) == 1:
                    A12s.append(A12)
                    A21s.append(A21)
                    AAs.append(AA)

                # _AA = AA.view(-1, H * W, H, W)
                if np.random.random() < (0.01 / len(t_pairs)) or visualize:
                    self.viz.text('%s %s' % (t1, t2), opts=dict(height=1, width=10000), win='div')
                    self.visualize_frame_pair(x, ff, mm, t1, t2)


            #########################################################
            # Affinity contrastive
            #########################################################

            if self.garg('cal_coef', 0) > 0:
                if not hasattr(self, 'aff_encoder'):
                    # downsample = nn.Conv3d(N, N//2,
                    #         kernel_size=1, stride=2, bias=False),
                    #     nn.BatchNorm2d(N//2)
                    # )
                    # self.aff_encoder = resnet2d.Bottleneck(A.shape[-1], 128, downsample=downsample)
                    aff_C = A.shape[-1]
                    self.aff_encoder = utils.make_aff_encoder(aff_C, self.enc_hid_dim).to(x.device)

                _As = torch.cat(As, dim=0)
                _As = _As.view(*_As.shape[:-1], int(N**0.5), int(N**0.5))
                a_con = self.aff_encoder(_As)
                a_con = a_con.sum(-1).sum(-1) / (a_con.shape[-1]*a_con.shape[-2])

                # _mm = mm.squeeze(-1).squeeze(-1).permute(0, 2, 3, 1)
                # _mm = mm.squeeze(-1).squeeze(-1).permute(0, 2, 3, 1) 
                # _mm = _mm.view(*_mm.shape[:-1], int(N**0.5), int(N**0.5))

                _ff = ff.view(*ff.shape[:-1], int(N**0.5), int(N**0.5))
                idxs = torch.Tensor(t_pairs).long()
                _ff = _ff[:, :, idxs].transpose(1, 2).flatten(0,1)

                a_hat = self.stack_encoder(_ff).squeeze(-3)
                a_hat = a_hat.sum(-1).sum(-1) / (a_hat.shape[-1]*a_hat.shape[-2])

                a_hat, a_con = F.normalize(a_hat, dim=-1), F.normalize(a_con, dim=-1)

                a_pred = torch.einsum('jk,lk->jl', a_hat, a_con) / self.temperature
                a_targ = torch.arange(0, a_pred.shape[-1]).to(a_pred.device)
                a_loss = self.xent(a_pred, a_targ).mean()
                xents.append(a_loss)
                diags['xent cont aff'] = a_loss.detach()


                # a_hat = a_hat.view(B, -1, *a_hat.shape[1:])
                # a_con = a_con.view(B, -1, *a_con.shape[1:])

                # for i, (t1, t2) in enumerate(t_pairs):
                #     _mm = mm.squeeze(-1).squeeze(-1).permute(0, 2, 3, 1)
                #     _mm = _mm.view(*_mm.shape[:-1], int(N**0.5), int(N**0.5))

                #     m1, m2 = _mm[:, :, t1], _mm[:, :, t2]
                #     ms = torch.stack([m1,m2], dim=2)

                    # a_hat = self.stack_encoder(ms)[..., 0, :, :]
                    # a_hat = a_hat.sum(-1).sum(-1) / (a_hat.shape[-1]*a_hat.shape[-2])

                # import pdb; pdb.set_trace()

            #########################################################


            # longer cycle:
            if self.garg('long_coef', 0) > 0:
                a12, a21 = A12s[0], A21s[0]
                for i in range(1, len(A12s)):
                    a12, a21 = torch.matmul(A12s[i], a12), torch.matmul(a21, A21s[i])
                    aa = torch.matmul(a21, a12)
                    log_aa = torch.log(aa + 1e-20).view(-1, aa.shape[-1])

                    xent_loss, acc = self.compute_xent_loss(aa, log_aa)
                    kldv_loss = self.compute_kldv_loss(aa, log_aa)

                    # xents += xent_loss
                    # kldvs += kldv_loss

                    xents.append(xent_loss)
                    kldvs.append(kldv_loss)
                    
                    diags['acc cyc %s' % str(i)] = acc
                    diags['xent cyc %s' % str(i)] = xent_loss.mean().detach()

            
        if _N > 1 and (np.random.random() < (0.01 / len(t_pairs)) or visualize):
            # all patches
            all_x = x.permute(0, 3, 1, 2, 4, 5)
            all_x = all_x.reshape(-1, *all_x.shape[-3:])
            all_f = ff.permute(0, 2, 3, 1).reshape(-1, ff.shape[1])
            all_f = all_f.reshape(-1, *all_f.shape[-1:])
            all_A = torch.einsum('ij,kj->ik', all_f, all_f)

            utils.nn_patches(self.viz, all_x, all_A[None])

        for k in diags:
            diags["%s %s" % (H, k)] = diags[k]
            del diags[k]

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
        if (len(x.shape) == 6 and x.shape[1] == 1):
            x = x.squeeze(1)

        if len(x.shape) < 6:
            # IMG VIZ
            # X here is B x C x T x H x W
            x1, x2 = x[0, :, t1].clone(), x[0, :, t2].clone()
            x1 -= x1.min(); x1 /= x1.max()
            x2 -= x2.min(); x2 /= x2.max()

            xx = torch.stack([x1, x2]).detach().cpu()
            self.viz.images(xx, win='imgs')
            # self._viz.patches(xx, A)

            # Keypoint Correspondences
            kp_corr = utils.draw_matches(f1[0], f2[0], x1, x2)

            self.viz.image(kp_corr, win='kpcorr')

            # # PCA VIZ
            spatialize = lambda xx: xx.view(*xx.shape[:-1], int(xx.shape[-1]**0.5), int(xx.shape[-1]**0.5))
            ff1 , ff2 = spatialize(f1[0]), spatialize(f2[0])
            pca_ff = utils.pca_feats(torch.stack([ff1,ff2]).detach().cpu())
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

