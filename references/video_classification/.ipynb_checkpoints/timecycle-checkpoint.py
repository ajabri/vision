import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter

import torchvision
import resnet as resnet3d
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
    def __init__(self, args=None):
        super(TimeCycle, self).__init__()
        # self.resnet = resnet3d.r3d_18(pretrained=False)
        self.resnet = resnet3d.r2d_10()
#         self.resnet = resnet3d.r2d_18(pretrained=True)

        self.resnet.fc, self.resnet.avgpool, self.resnet.layer4 = None, None, None

        self.infer_dims()
        # self.resnet_nchan = self.resnet.

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

        self._targets = {}
        
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

    def compute_affinity(self, x1, x2, do_dropout=True):
        N, C, T, H, W  = x1.shape

        # assert x1.shape == x2.shape
        
        # assuming xs: N, C, 1, H, W
        x1 = x1.transpose(3, 4).contiguous() # for the inlier counter
        x1_flat = x1.view(x1.size(0), x1.size(1), -1)
        x1_flat = x1_flat.transpose(1, 2)
        x2_flat = x2.transpose(3, 4).contiguous().view(x2.size(0), x2.size(1), -1)

        # import pdb; pdb.set_trace()

        A = torch.matmul(x1_flat, x2_flat)
        A = torch.div(A, C**0.5)

        #         A = torch.div(A, 1/C**0.5)

        if do_dropout:
            x1_flat, x2_flat = F.dropout(x1_flat, p=0.5), F.dropout(x2_flat, p=0.5)

        # import pdb; pdb.set_trace()

        # if self.dropout_rate > 0:
        #     A = self.dropout(A)

        if self.zero_diagonal:
            A[:, torch.eye(A.shape[1]).long().cuda()] = 0
            
            # A = 
        # import pdb; pdb.set_trace()
        # A12 = A.view(A.size(0), 1, H * H, W, W)
        # A21 = A.view(A.size(0), 1, H, H, W * W) 
        # A12  = F.softmax(A, dim=2)
        # A21  = F.softmax(A.transpose(1, 2), dim=2)
        A1, A2 = A, A.transpose(1, 2).clone()
        
        if do_dropout:
            A1, A2 = self.dropout(A1), self.dropout(A2)
            # A1, A2 = self.dropout(A), self.dropout(A.transpose(1, 2))

        A1 = F.softmax(A1, dim=2)
        A2 = F.softmax(A2, dim=2)

        AA = torch.matmul(A2, A1)

        # import pdb; pdb.set_trace()
        log_AA = torch.log(AA + 1e-20)

        return A, AA, log_AA
    
    def forward_affinity(self, x1, x2, encode=False):
        '''
        For computing similarity of things in X1 w.r.t X2
        As in, will return (n x H1*W1 x H2 x W2) sized affinity object
        '''

        if encode:
            x1 = self.resnet(x1)
            x2 = self.resnet(x2)
        
        N, C, T, H1, W1 = x1.shape
        H2, W2 = x2.shape[-2:]
        
        A, AA, log_AA = self.compute_affinity(x1, x2)
        
        A = A.view(*A.shape[:2], H2, W2)

        return A
    
    def forward_encoder(self, x):
        return self.resnet(x)
    
    def forward(self, x, just_feats=False):
        ff = self.resnet(x)
        ff = self.selfsim_head(ff)
#         ff = F.normalize(ff, p=2, dim=1)
        
        N, C, T, _H, _W = ff.shape
        _h, _w = _H // 4, _W // 4

        xents = torch.tensor([0.]).cuda()
        kldvs = torch.tensor([0.]).cuda()
        accur = torch.tensor([0.]).cuda()

        L = len(list(itertools.combinations(range(T), 2)))
        for (t1, t2) in itertools.combinations(range(T), 2):
            x1, x2 = ff[:, :, t1:t1+1, _h:-_h, _w:-_w].contiguous(), ff[:, :, t2:t2+1, _h:-_h, _w:-_w].contiguous()
            #ff[:, :, t2:t2+1, 2*_h:-2*_h, 2*_w:-2*_w].contiguous()
            
            # x1, x2 = ff[:, :, t1:t1+1, _h:-_h, _w:-_w].contiguous(), ff[:, :, t2:t2+1, 2*_h:-2*_h, 2*_w:-2*_w].contiguous()
            # x1, x2 = ff[:, :, t1:t1+1, _H//2-_h:_H//2+_h, _W//2-_w:_W//2+_w].contiguous(), \
            #     ff[:, :, t2:t2+1, _H//2-_h:_H//2+_h, _W//2-_w:_W//2+_w].contiguous()
            H, W  = x2.shape[-2:]

            # import pdb; pdb.set_trace()

            if H*W not in self._targets:
                self._targets[H*W] = self.make_smooth_target_2d(H, W)

            # Self similarity
            A, AA, log_AA = self.compute_affinity(x1, x2)
            
            target = torch.arange(AA.shape[1])[None].repeat(AA.shape[0], 1)
            target = (target).view(-1).cuda()
            # import pdb; pdb.set_trace()

            log_AA = log_AA.view(-1, log_AA.shape[1])

            # Cross Entropy
            if self.xent_coef > 0:
                _xent_loss = self.xent(log_AA, target)
                xents += _xent_loss.mean()
                # import pdb; pdb.set_trace()
                # print((torch.argmax(log_AA, dim=-1) == target).sum())
                accur += (torch.argmax(log_AA, dim=-1) == target).float().mean()

            # KL Div with Smoothed 2D Targets
            if self.kldv_coef > 0:
                I = self._targets[H*W][None].repeat(N, 1, 1).view(-1, A.shape[-1]).cuda()
                kldv_loss = self.kldv(log_AA, I)
                # print(kldv_loss, log_AA.min(), AA.min(), A.min())
                kldvs += kldv_loss

            # import pdb; pdb.set_trace()
            # self.viz.images()

            # _AA = AA.view(-1, H * W, H, W)
            if np.random.random() < 0.003:
                self.viz.text('%s %s' % (t1, t2), opts=dict(height=1, width=10000))

                # Self similarity
                A, AA, log_AA = self.compute_affinity(x1, x2, do_dropout=False)
                log_AA = log_AA.view(-1, log_AA.shape[1])
                _xent_loss = self.xent(log_AA, target)
                _AA = AA.view(-1, H * W, H, W)

                _A = A.view(*A.shape[:2], x1.shape[-1], -1)
                u, v = utils.compute_flow(_A[0:1])

                flows = torch.stack([u, v], dim=-1).cpu().numpy()
                flows = utils.draw_hsv(flows[0])
                # import pdb; pdb.set_trace()

                flows = cv2.resize(flows, (256, 256))
                self.viz.image((flows).transpose(2, 0, 1))

                # flows = [cv2.resize(flow.clip(min=0).astype(np.uint8), (256, 256)) for flow in flows]
                # self.viz.image((flows[0]).transpose(2, 0, 1))

                # import time
                # time.sleep(0.1)
                # import pdb; pdb.set_trace()

                xx = _xent_loss[:H*W]
                xx -= xx.min()
                xx /= xx.max()
                # xx = color(xx.detach().cpu().numpy())

                _img = torch.stack([x[0, :, t1], x[0, :, t2]])
                _img -= _img.min()
                _img /= _img.max()
                self.viz.images(_img)

                pca_ff = utils.pca_feats(torch.stack([ff[0, :, t1], ff[0, :, t2]]).detach().cpu())
                pca_ff = utils.make_gif(pca_ff, outname=None)
                self.viz.images(pca_ff.transpose(0, -1, 1, 2))

                img_grid = [cv2.resize(aa, (50,50), interpolation=cv2.INTER_NEAREST)[None] for aa in _AA[0, :, :, :, None].cpu().detach().numpy()]
                img_grid = [img_grid[_].repeat(3, 0) * np.array(color(xx[_].item()))[:3, None, None] for _ in range(H*W)]
                img_grid = [img_grid[_] / img_grid[_].max() for _ in range(H*W)]
                img_grid = torch.from_numpy(np.array(img_grid))
                
                img_grid = torchvision.utils.make_grid(img_grid, nrow=H, padding=1, pad_value=1)
                # img_grid = cv2.resize(img_grid.permute(1, 2, 0).cpu().detach().numpy(), (1000, 1000), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
                self.viz.images(img_grid)


        return ff, self.xent_coef * (xents/L), self.kldv_coef * (kldvs/L), accur/L

        # return dict(x=ff, xent_loss=xents, kldv_loss=kldvs)


    def forward2(self, x):
        iH, iW = x.shape[-2:]
        _ih, _iw = iH // 6, iW // 6
        base, query = x[:, :, 0:2], x[:, :, -1:, iH//2-_ih:iH//2+_ih, iW//2-_iw:iW//2+_iw]

        # import pdb; pdb.set_trace()
        X1, X2 = self.resnet(base), self.resnet(query)

        # ff = self.selfsim_head(ff)
        # ff = F.normalize(ff, p=2, dim=1)
        
        N, C = X1.shape[:2]
        # _h, _w = _H // 10, _W // 10

        xents = torch.tensor([0.]).cuda()
        kldvs = torch.tensor([0.]).cuda()
        accur = torch.tensor([0.]).cuda()

        # L = len(list(itertools.combinations(range(T), 2)))
        # for (t1, t2) in itertools.combinations(range(T), 2):
        L = 1
        for _ in range(L):
            x1, x2 = X1[:, :, 0:1], X2
            H, W  = x2.shape[-2:]


            if H*W not in self._targets:
                self._targets[H*W] = self.make_smooth_target_2d(H, W)

            # Self similarity
            A, AA, log_AA = self.compute_affinity(x1, x2)
            
            target = torch.arange(AA.shape[1])[None].repeat(AA.shape[0], 1)
            target = (target).view(-1).cuda()
            # import pdb; pdb.set_trace()

            log_AA = log_AA.view(-1, log_AA.shape[1])

            # Cross Entropy
            if self.xent_coef > 0:
                _xent_loss = self.xent(log_AA, target)
                xents += _xent_loss.mean()
                # import pdb; pdb.set_trace()
                # print((torch.argmax(log_AA, dim=-1) == target).sum())
                accur += (torch.argmax(log_AA, dim=-1) == target).float().mean()

            # KL Div with Smoothed 2D Targets
            if self.kldv_coef > 0:
                I = self._targets[H*W][None].repeat(N, 1, 1).view(-1, A.shape[-1]).cuda()
                kldv_loss = self.kldv(log_AA, I)
                # print(kldv_loss, log_AA.min(), AA.min(), A.min())
                kldvs += kldv_loss

            # import pdb; pdb.set_trace()
            # self.viz.images()

            # _AA = AA.view(-1, H * W, H, W)
            if np.random.random() < 0.01:

                # Self similarity
                A, AA, log_AA = self.compute_affinity(x1, x2, do_dropout=False)
                log_AA = log_AA.view(-1, log_AA.shape[1])
                _xent_loss = self.xent(log_AA, target)
                _AA = AA.view(-1, H * W, H, W)

                import pdb; pdb.set_trace()
                xx = _xent_loss[:H*W]
                xx -= xx.min()
                xx /= xx.max()
                # xx = color(xx.detach().cpu().numpy())

                _img = torch.stack([x[0, :, 0], x[0, :, -1]])
                _img -= _img.min()
                _img /= _img.max()
                self.viz.text('%s %s' % (0, -1), opts=dict(height=1, width=10000))
                self.viz.images(_img)

                # import pdb; pdb.set_trace()
                pca_ff = utils.pca_feats(X1[0, :].transpose(0, 1).detach().cpu())
                pca_ff = utils.make_gif(pca_ff, outname=None)
                self.viz.images(pca_ff.transpose(0, -1, 1, 2))

                pca_ff = utils.pca_feats(X2[0, :].transpose(0, 1).detach().cpu())
                pca_ff = utils.make_gif(pca_ff, outname=None)
                self.viz.image(pca_ff.transpose(0, -1, 1, 2)[0])

                img_grid = [cv2.resize(aa, (50,50), interpolation=cv2.INTER_NEAREST)[None] for aa in _AA[0, :, :, :, None].cpu().detach().numpy()]
                img_grid = [img_grid[_].repeat(3, 0) * np.array(color(xx[_].item()))[:3, None, None] for _ in range(H*W)]
                img_grid = [img_grid[_] / img_grid[_].max() for _ in range(H*W)]
                img_grid = torch.from_numpy(np.array(img_grid))
                
                img_grid = torchvision.utils.make_grid(img_grid, nrow=H, padding=1, pad_value=1)
                # img_grid = cv2.resize(img_grid.permute(1, 2, 0).cpu().detach().numpy(), (1000, 1000), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
                self.viz.images(img_grid)

        return x1, self.xent_coef * (xents/L), self.kldv_coef * (kldvs/L), accur/L
