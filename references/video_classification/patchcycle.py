import torch
import numpy as np
import pdb
from skimage.util.shape import view_as_windows

from torch import nn
import torch.nn.functional as F
import utils
    
class RelHead(nn.Module):
    def __init__(self, enc, fdim, vdim=128, K=10, vis=None):
        super(RelHead, self).__init__()
        self.enc = enc
        self.K = K
        self.fdim = fdim
        self.vdim = vdim
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(fdim, vdim)

        self.head1 = nn.Linear(vdim, vdim*self.K)
        self.head2 = nn.Linear(vdim, vdim*self.K)

        self.temp = 0.5
        self.vis = vis

    def map_heads(self, f, head, B, N):
        # apply all k-heads
        v = head(f)
        # relation-specific transformed features
        vv = v.view(B, N, self.K, self.vdim).transpose(1, 2)  # B, K, N, C

        return vv

    # def forward(self, x, func='hydra'):
    #     return getattr(self, func)(x)
    
    def map2vec(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

    def zero_diag(self, A, zero=0):
        mask = (torch.eye(A.shape[-1]).unsqueeze(0).repeat(A.shape[0], 1, 1).bool() < 1).float().cuda()
        # import pdb; pdb.set_trace()
        return A * mask

    def cam_grid(self, x, f):
        # import pdb; pdb.set_trace()
        # xx = torch.einsum("ijk,ilk->ijlk", x, x)
        mapped = torch.einsum("ijk,ioklm->ijolm", f_norm, f_maps)
        

    def forward(self, x):
        B, C, H, W = x.shape
        N, C = C//3, 3

        x = x.view(B*N, C, *x.shape[-2:])
        f = self.enc(x)
        f_maps = f

        if len(f.shape) > 3 and N == 1:
            f = f.view(*f.shape[:-2], -1)
            N = f.shape[-1]
            f = f.transpose(-1, -2)
        else:
            f = self.map2vec(f)
            
        f = f.view(B, N, -1) # B x N x d
        f_maps = f_maps.view(B, N, *f_maps.shape[1:])

        f_norm = F.normalize(f, dim=-1, p=2) # f_norm = f

        # cam = self.cam_grid(f_norm, f_maps)
        # import pdb; pdb.set_trace()
        
        A = torch.matmul(f_norm, f_norm.transpose(-1,-2))
        Atemp = torch.div(A, 1/128**0.5)
        # self.zero_diag(A, zero=-1e10)

        A1  = F.softmax(Atemp, dim=-2)        
        A1  = F.normalize(F.dropout(self.zero_diag(A1), p=0.5), dim=-2, p=1)
        A2t = F.softmax(Atemp.transpose(-1,-2), dim=-2)
        A2t = F.normalize(F.dropout(self.zero_diag(A2t), p=0.5), dim=-2, p=1)

        AtA = torch.matmul(A2t, A1).transpose(-1, -2)
        targs = torch.arange(AtA.shape[-1]).unsqueeze(0).repeat(AtA.shape[0], 1)
        # loss = F.cross_entropy(torch.log(AtA), targs.cuda())
        loss = F.nll_loss(AtA, targs.cuda())
        # import pdb; pdb.set_trace()

        if np.random.random() < 0.5 and self.vis is not None and torch.cuda.current_device() == 0:
            self.vis.log([
                ('loss', (-1, loss.item())), 
                ('AtA.min', (-1, AtA.min().item())),
                ('AtA.max', (-1, AtA.max().item())),

                ('v_w.max_norm', (-1, f.norm(dim=-1).max().item())),
                ('v_w.norm_mu', (-1, f.norm(dim=-1).mean().item())),
                ('v_w.norm_std', (-1, f.norm(dim=-1).std().item()))
            ])
            print('ACC', (targs.cuda() == torch.argmax(AtA, dim=-1)).sum().item() / (targs.shape[0]*targs.shape[1]))

            if np.random.random() < 0.1:
                self.vis.patches( x.view(B, N, C, *x.shape[-2:])[0], A[0].unsqueeze(0).transpose(-1, -2), prefix='2')

                # import pdb; pdb.set_trace()

                pca_feats = utils.pca_feats(f_maps[0].detach().cpu())
                pca_feats = utils.make_gif(pca_feats, outname=None)
                self.vis.viz.images(pca_feats.transpose(0, -1, 1, 2), nrow=int(N**0.5), win='pca_viz')
                xi = x[0:N]
                self.vis.viz.images((xi-xi.min()) / ((xi-xi.min()).max()), nrow=int(N**0.5), win='pca_viz_imgs')


        return (f, A), loss

    def hydra(self, x): 
        B, C, H, W = x.shape
        N, C = C//3, 3
        x = x.view(B*N, C, *x.shape[-2:])

        f = self.enc(x)
        f = self.map2vec(x)
        f = f.view(B, N, -1) # B x N x d
        # f = f.view(B, N, *f.shape[-3:]) # B x N x d x h (1) x w (1)
        # f = torch.flatten(f, 2) # B x N x d

        # apply all k-heads
        v1 = self.map_heads(f, self.head1, B, N)
        v2 = self.map_heads(f, self.head2, B, N) # B, K, N, C
        v2 = v2.transpose(2, 3) # B, K, C, N

        # replicate original features for all dot products
        # v2 = f.view(B, N, 1, self.vdim).repeat(1, 1, self.K, 1)
        # v2 = v2.transpose(1,2).transpose(2, 3) # B, K, C, N

        # import pdb; pdb.set_trace()
        v1_norm = F.normalize(v1, dim=-1)
        v2_norm = F.normalize(v2, dim=-2)

        A_k = torch.matmul(v1_norm, v2_norm) 
        B, R, N, _ = A_k.shape
        
        # node-wise normalization
        # maxes = A_k.max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1).detach()
        # expA = torch.exp(A_k - maxes) + 1e-10 # - A_k.max()) + 1e-6
        # expA = torch.exp(A_k)
        # p_r_y__x = expA / expA.sum(1).sum(1).unsqueeze(1).unsqueeze(1)   # per node joint R, Y distribution
        p_r_y__x = F.softmax(A_k.view(B, -1, N), dim=1).view(B, R, N, N)

        p_r__x = p_r_y__x.sum(2) 
        p_y__x = p_r_y__x.sum(1)
        
        p_r__x_y = p_r_y__x / p_y__x.unsqueeze(1)
        p_y__x_r = p_r_y__x / p_r__x.unsqueeze(2)

        # import pdb; pdb.set_trace()

        H_R__X = (-1 * p_r__x * torch.log(p_r__x)).sum(1).mean()
        H_R__X_Y = (-1 * p_r__x_y * torch.log(p_r__x_y)).sum(1).mean()

        H_Y__X = (-1 * p_y__x * torch.log(p_y__x)).sum(1).mean()
        H_Y__X_R = (-1 * p_y__x_r * torch.log(p_y__x_r)).sum(2).mean()

        # import pdb; pdb.set_trace()
        if np.random.random() < 0.01:
            print('H(R|X):', H_R__X.item(), 'H(R|X,Y):', H_R__X_Y.item(), 'I(R;Y|X):', H_R__X.item() - H_R__X_Y.item(), H_Y__X.item() - H_Y__X_R.item())
            print(p_r__x.sum(1).mean().item(), p_r__x.min().item(), p_r__x.max().item(), p_r__x_y.sum(1).mean().item(), p_r__x_y.min(), p_r__x_y.max().item())
        
        if np.random.random() < 0.1 and self.vis is not None and torch.cuda.current_device() == 0:
            self.vis.log([
                ('H(R|X):', (-1, H_R__X.item())),
                ('H(R|X,Y):', (-1, H_R__X_Y.item())),
                ('I(R;Y|X):', (-1, H_R__X.item() - H_R__X_Y.item())),
                ('I(Y;R|X):', (-1, H_Y__X.item() - H_Y__X_R.item())),
                
                ('A_k.min', (-1, A_k.min().item()*self.temp)),
                ('A_k.max', (-1, A_k.max().item()*self.temp)),

                ('v_w.max_norm', (-1, v1.norm(dim=-1).max().item())),
                ('v_w.norm_mu', (-1, v1.norm(dim=-1).mean().item())),
                ('v_w.norm_std', (-1, v1.norm(dim=-1).std().item()))
            ])

            if np.random.random() < 0.05:
                self.vis.patches(
                    x.view(B, N, C, *x.shape[-2:])[0],
                    p_r_y__x[0])

        loss  = - (H_R__X - 1 * H_R__X_Y)
        loss += - (H_Y__X - 1 * H_Y__X_R)

        # import pdb; pdb.set_trace()
# 
        return (v1, A_k), loss
    

