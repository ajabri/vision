# Video's features
import numpy as np
from sklearn.decomposition import PCA
import cv2
import imageio as io

import visdom
import time
import PIL
import torchvision
import torch
from . import transforms as T
import skimage

from matplotlib import cm

def pca_feats(ff, solver='auto', img_normalize=True):
    ## expect ff to be   N x C x H x W
        
    N, C, H, W = ff.shape
    pca = PCA(
        n_components=3,
        svd_solver=solver,
        whiten=True
    )
#     print(ff.shape)
    ff = ff.transpose(1, 2).transpose(2, 3)
#     print(ff.shape)
    ff = ff.reshape(N*H*W, C).numpy()
#     print(ff.shape)
    
    pca_ff = torch.Tensor(pca.fit_transform(ff))
    pca_ff = pca_ff.view(N, H, W, 3)
#     print(pca_ff.shape)
    pca_ff = pca_ff.transpose(3, 2).transpose(2, 1)

    if img_normalize:
        pca_ff = (pca_ff - pca_ff.min()) / (pca_ff.max() - pca_ff.min())


    return pca_ff


def make_gif(video, outname='/tmp/test.gif', sz=256):
    if hasattr(video, 'shape'):
        video = video.cpu()
        if video.shape[0] == 3:
            video = video.transpose(0, 1)

        video = video.numpy().transpose(0, 2, 3, 1)
        # print(video.min(), video.max())
        
        video = (video*255).astype(np.uint8)
#         video = video.chunk(video.shape[0])
        
    video = [cv2.resize(vv, (sz, sz)) for vv in video]

    if outname is None:
        return np.stack(video)

    io.mimsave(outname, video, duration = 0.2)


from matplotlib import cm
import time
import cv2

def draw_matches(x1, x2, i1, i2):
    # x1, x2 = f1, f2/
    detach = lambda x: x.detach().cpu().numpy().transpose(1,2,0) * 255
    i1, i2 = detach(i1), detach(i2)
    i1, i2 = cv2.resize(i1, (400, 400)), cv2.resize(i2, (400, 400))

    for check in [True]:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=check)
        # matches = bf.match(x1.permute(0,2,1).view(-1, 128).cpu().detach().numpy(), x2.permute(0,2,1).view(-1, 128).cpu().detach().numpy())

        h = int(x1.shape[-1]**0.5)
        matches = bf.match(x1.t().cpu().detach().numpy(), x2.t().cpu().detach().numpy())

        scale = i1.shape[-2] / h
        grid = torch.stack([torch.arange(0, h)[None].repeat(h, 1), torch.arange(0, h)[:, None].repeat(1, h)])
        
        grid = grid.view(2, -1)
        grid = grid * scale + scale//2

        kps = [cv2.KeyPoint(grid[0][i], grid[1][i], 1) for i in range(grid.shape[-1])]

        matches = sorted(matches, key = lambda x:x.distance)

        # img1 = img2 = np.zeros((40, 40, 3))
        out = cv2.drawMatches(i1.astype(np.uint8), kps, i2.astype(np.uint8), kps,matches[:], None, flags=2).transpose(2,0,1)

    return out

class PatchGraph(object):
    
    color = cm.get_cmap('jet')
    pad = 0
    
    def blend(self, i):

        y, x = i // self.W, i % self.W
        cx, cy = [int((self.w + self.pad) * (x  + 0.5)), int((self.h + self.pad) * (y  + 0.5))]

        def _blend(img, mask):
            img = img[...,:-self.pad, :-self.pad] if self.pad > 0 else img
            img = (0.5 * mask[i] + 0.5 * img).copy() * 255
            # import pdb; pdb.set_trace()

            return img

        img1 = self.grid[0]*255.0
        img1[:, cy-5:cy+5, cx-5:cx+5] = 255

        key_imgs = [_blend(self.grid[j+1], self.maps[j]) for j in range(0, len(self.maps))]

        return np.concatenate([img1] + key_imgs, axis=-1), None

    def update(self):
        if self.viz is not None:
            self.viz.image(self.curr[0], win=self.win_id, env=self.viz.env+'_pg')
            # self.viz.image(self.curr[1], win=self.win_id2, env=self.viz.env+'_pg')

    def make_canvas(self, I, orig, N):
        # import pdb; pdb.set_trace()
        # if N == 1:
        #     grid = [cv2.resize(o.numpy().transpose(1,2,0), (800, 800), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1) for o in orig]
        # else:
        grid = []
        for i in range(I.shape[1]):
            grid += [torchvision.utils.make_grid(I[:, i], nrow=int(N**0.5), padding=self.pad, pad_value=0).cpu().numpy()]
        
        for i in range(len(grid)):
            grid[i] -= grid[i].min()
            grid[i] /= grid[i].max()
        
        # if orig is not None:
        #     self.orig = cv2.resize(orig[0].numpy().transpose(1,2,0), self.grid.shape[-2:], interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        #     self.orig -= self.orig.min()
        #     self.orig /= self.orig.max()
        # else:
        #     self.orig = None
        
        return grid

    def __init__(self, I, A, viz=None, win='patchgraph', orig=None):
        self._win = win
        self.viz = viz
        self._birth = time.time()

        if self.viz is not None:
            self.viz.close(self.viz.env+'_pg')

        P, T, C, h, w = I.shape  # P is patches per image, 1 means whole image
        N = A.shape[-1]
        H = W = int(N ** 0.5)

        self.N, self.H, self.W, self.h, self.w = N, H, W, h, w
        if P == 1:
            self.w, self.h = self.w // W, self.h // H

        I = I.cpu()
        orig = orig.cpu()

        A = A.view(A.shape[0], H * W, H, W).cpu() #.transpose(-1, -2)

        # psize = min(2000 // H, I.shape[-1])
        # if psize < I.shape[-1]:
        #     I = [cv2.resize(ii, (psize, psize)) for ii in I]

        ####################################################################
        # Construct image data

        self.grid = self.make_canvas(I, orig, N)

        ####################################################################
        # Construct map data

        # pstride = utils.get_stride(orig.shape[-1], h, H)   # stride length used to gen patches, h is patch size, H is n patch per side
        # map_sz_ratio = (pstride * H ) / orig.shape[-1]     # compute percentage of image spanned by affinity map overlay
        # map_sz = int(map_sz_ratio * self.orig.shape[-1])
        # lpad = int(((h-pstride)//2 / orig.shape[-1]) * self.orig.shape[-1])
        # rpad = self.orig.shape[-1] - map_sz - lpad

        map_sz = self.grid[0].shape[-1]
        lpad, rpad = 0, 0

        zeros = np.zeros(self.grid[0].shape).transpose(1,2,0)
        maps = []

        for A_t in A[..., None].numpy():
            maps.append([])
            for a in A_t:
                _a = cv2.resize(a, (map_sz, map_sz), interpolation=cv2.INTER_NEAREST)
                _a = _a**10
                _a /= _a.max()
                _a = self.color(_a * 255.0)[...,:3]
                a = zeros.copy()
                if lpad > 0 and rpad > 0:
                    a[lpad:-rpad, lpad:-rpad, :] = _a
                else:
                    a = _a
                
                maps[-1].append(a)
        
        self.maps = np.array(maps).transpose(0, 1, -1, 2, 3)

        ####################################################################
        # Set first image

        self.curr_id = (H//2) * W + W//2
        self.curr = self.blend(self.curr_id)
        # viz.text('', opts=dict(width=10000, height=2), env=viz.env+'_pg')
        
        self.win_id = self._win 
        self.win_id2 = self._win+'2'
        self.win_id_text = self._win+'_text'

        self.update()
        ####################################################################

        def str2inttuple(s):
            try:
                ss = s.split(',')
                assert(len(ss) == 2)
                return int(ss[0]), int(ss[1])
            except:
                return False

        def callback(event):
            # nonlocal win_id #, win_id_text
            # print(event['event_type'])

            #TODO make the enter key recompute the A under a
            if event['event_type'] == 'KeyPress':
                # print(event['key'], 'KEYYYYY')

                if 'Arrow' in event['key']:
                    self.curr_id += {'ArrowLeft':-1, 'ArrowRight': 1, 'ArrowUp': -self.W, 'ArrowDown': self.W}[event['key']]
                    # print('hello!!', self.curr_id)
                    self.curr_id = min(max(self.curr_id, 0), N)
                    self.curr = self.blend(self.curr_id)
                    self.update()

                # curr_txt = event['pane_data']['content']

                # print(event['key'], 'KEYYYYY')
                # if event['key'] == 'Enter':
                #     itup = str2inttuple(curr_txt)
                #     if itup:
                #         self.curr = self.blend(itup[0]*H + itup[1])
                #         viz.image(self.curr, win=self.win_id, env=viz.env+'_pg')
                #         curr_txt='Set %s' % curr_txt
                #     else:
                #         curr_txt='Invalid position tuple'

                # elif event['key'] == 'Backspace':
                #     curr_txt = curr_txt[:-1]
                # elif event['key'] == 'Delete':
                #     curr_txt = ''
                # elif len(event['key']) == 1:
                #     curr_txt += event['key']
                

                # viz.text(curr_txt, win=self.win_id_text, env=viz.env+'_pg')

            if event['event_type'] == 'Click':
                # print(event.keys())
                # import pdb; pdb.set_trace()
                # viz.text(event)
                coords = "x: {}, y: {};".format(
                    event['image_coord']['x'], event['image_coord']['y']
                )
                x, y = event['image_coord']['x'], event['image_coord']['y']
                self.curr_id = int( (y // self.h) * self.W + (x // self.w))
                self.curr = self.blend(self.curr_id)
                self.update()

        if viz is not None:
            viz.register_event_handler(callback, self.win_id)
        # viz.register_event_handler(callback, self.win_id_text)
        # import pdb; pdb.set_trace()

import wandb

class Visualize(object):
    def __init__(self, args, suffix='metrics', log_interval=2*60):

        self._env_name = "%s-%s" % (args.name, suffix)
        self.vis = visdom.Visdom(
            port=args.port,
            server='http://%s' % args.server,
            env=self._env_name,
        )
        self.data = dict()
        self.args = args

        self.log_interval = log_interval
        self._last_flush = time.time()
        self._init = False

    def wandb_init(self):
        if not self._init:
            self._init = True
            wandb.init(project="palindromes", group="release", config=self.args)

    def log(self, key_vals):
        return wandb.log(key_vals)

    def log2(self, key, value):
        if not key in self.data:
            self.data[key] = [[],[]]

        if isinstance(value, tuple):
            self.data[key][0].append(value[0])
            self.data[key][1].append(value[1])
        else:
            self.data[key][1].append(value)
            self.data[key][0].append(len(self.data[key][1]) * 1.0)
            # import pdb; pdb.set_trace()

        if (time.time() - self._last_flush) > (self.log_interval):
            for k in self.data:
                self.vis.line(
                    X=np.array(self.data[k][0]),
                    Y=np.array(self.data[k][1]),
                    win=k,
                    opts=dict( title=k )
                )
            self._last_flush = time.time()
            if np.random.random() < 0.1:
                self.save()
            
    def nn_patches(self, P, A_k, prefix='', N=10, K=20):
        nn_patches(self.vis, P, A_k, prefix, N, K)

    def save(self):
        self.vis.save([self._env_name])

def get_stride(im_sz, p_sz, res):
    stride = (im_sz - p_sz)//(res-1)
    return stride

def nn_patches(vis, P, A_k, prefix='', N=10, K=20):
    # produces nearest neighbor visualization of N patches given an affinity matrix with K channels

    P = P.cpu().detach().numpy()
    P -= P.min()
    P /= P.max()

    A_k = A_k.cpu().detach().numpy() #.transpose(-1,-2).numpy()
    # assert np.allclose(A_k.sum(-1), 1)

    A = np.sort(A_k, axis=-1)
    I = np.argsort(-A_k, axis=-1)
    
    vis.text('', opts=dict(width=10000, height=1), win='%s_patch_header' %(prefix))


    for n,i in enumerate(np.random.permutation(P.shape[0])[:N]):
        p = P[i]
        vis.text('', opts=dict(width=10000, height=1), win='%s_patch_header_%s' % (prefix, n))
        # vis.image(p,  win='%s_patch_query_%s' % (prefix, n))

        for k in range(I.shape[0]):
            vis.images(P[I[k, i, :K]], nrow=min(I.shape[-1], 20), win='%s_patch_values_%s_%s' % (prefix, n, k))
            vis.bar(A[k, i][::-1][:K], opts=dict(height=150, width=500), win='%s_patch_affinity_%s_%s' % (prefix, n, k))


def compute_flow(corr):
    # assume corr is shape N x H * W x W x H
    h = w = int(corr.shape[-1] ** 0.5)

    # x1 -> x2
    corr = corr.transpose(-1, -2).view(*corr.shape[:-1], h, w)
    nnf = corr.argmax(dim=1)

    u = nnf % w # nnf.shape[-1]
    v = nnf / h # nnf.shape[-2] # nnf is an IntTensor so rounds automatically

    rr = torch.arange(u.shape[-1])[None].long().cuda()

    for i in range(u.shape[-1]):
        u[:, i] -= rr

    for i in range(v.shape[-1]):
        v[:, :, i] -= rr

    return u, v


def frame_pair(x, ff, mm, t1, t2, AA, xent_loss, viz):
    normalize = lambda x: (x-x.min()) / (x-x.min()).max()
    N = AA.shape[-1]
    H = W = int(N**0.5)
    AA = AA.view(-1, H * W, H, W)

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

    ##############################################
    ## Visualize PCA of Embeddings, Correspondences
    ##############################################

    # import pdb; pdb.set_trace()
    if (len(x.shape) == 6 and x.shape[1] == 1):
        x = x.squeeze(1)

    if len(x.shape) < 6:   # Single image input, no patches
        # X here is B x C x T x H x W
        x1, x2 = x[0, :, t1].clone(), x[0, :, t2].clone()
        x1, x2 = normalize(x1), normalize(x2)

        xx = torch.stack([x1, x2]).detach().cpu()
        viz.images(xx, win='imgs')

        # Keypoint Correspondences
        kp_corr = draw_matches(f1[0], f2[0], x1, x2)
        viz.image(kp_corr, win='kpcorr')

        # # PCA VIZ
        spatialize = lambda xx: xx.view(*xx.shape[:-1], int(xx.shape[-1]**0.5), int(xx.shape[-1]**0.5))
        ff1 , ff2 = spatialize(f1[0]), spatialize(f2[0])
        pca_ff = pca_feats(torch.stack([ff1,ff2]).detach().cpu())
        pca_ff = make_gif(pca_ff, outname=None)
        viz.images(pca_ff.transpose(0, -1, 1, 2), win='pcafeats')

    else:  # Patches as input
        # X here is B x N x C x T x H x W
        x1, x2 =  x[0, :, :, t1],  x[0, :, :, t2]
        m1, m2 = mm[0, :, :, t1], mm[0, :, :, t2]

        pca_ff = pca_feats(torch.cat([m1, m2]).detach().cpu())
        pca_ff = make_gif(pca_ff, outname=None, sz=64).transpose(0, -1, 1, 2)
        
        pca1 = torchvision.utils.make_grid(torch.Tensor(pca_ff[:N]), nrow=int(N**0.5), padding=1, pad_value=1)
        pca2 = torchvision.utils.make_grid(torch.Tensor(pca_ff[N:]), nrow=int(N**0.5), padding=1, pad_value=1)
        img1 = torchvision.utils.make_grid(normalize(x1)*255, nrow=int(N**0.5), padding=1, pad_value=1)
        img2 = torchvision.utils.make_grid(normalize(x2)*255, nrow=int(N**0.5), padding=1, pad_value=1)
        viz.images(torch.stack([pca1,pca2]), nrow=4, win='pca_viz_combined1')
        viz.images(torch.stack([img1.cpu(),img2.cpu()]), nrow=4, win='pca_viz_combined2')
    
    ##############################################
    # LOSS VIS
    ##############################################
    color = cm.get_cmap('winter')

    xx = normalize(xent_loss[:H*W])
    img_grid = [cv2.resize(aa, (50,50), interpolation=cv2.INTER_NEAREST)[None] 
                for aa in AA[0, :, :, :, None].cpu().detach().numpy()]
    img_grid = [img_grid[_].repeat(3, 0) * np.array(color(xx[_].item()))[:3, None, None] for _ in range(H*W)]
    img_grid = [img_grid[_] / img_grid[_].max() for _ in range(H*W)]
    img_grid = torch.from_numpy(np.array(img_grid))
    img_grid = torchvision.utils.make_grid(img_grid, nrow=H, padding=1, pad_value=1)
    
    # img_grid = cv2.resize(img_grid.permute(1, 2, 0).cpu().detach().numpy(), (1000, 1000), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
    viz.images(img_grid, win='lossvis')





# #################################################################
# # PatchGraph
# #################################################################

# if visualize and self.viz is not None: #np.random.random() < 0.05 and visualize:
#     with torch.no_grad():

#         if ff.device.index == 0:
#             for i in range(B):
#                 pg_win = 'patchgraph_%s'%i
#                 # print('exists', self.viz.win_exists(pg_win, env=self.viz.env+'_pg'))
#                 if not self.viz.win_exists(pg_win, env=self.viz.env+'_pg') or visualize:
#                     tviz = 0
#                     self.viz.clear_event_handlers(pg_win)
#                     fff = ff[i].transpose(0,1)
#                     A_traj = self.compute_affinity(ff[i].transpose(0,1)[:-1] , ff[i].transpose(0,1)[1:], do_dropout=False)

#                     A_t = [self.stoch_mat(A_traj[0])]
#                     for A_tp1 in A_traj[1:]:
#                         A_t.append(self.stoch_mat(A_tp1) @ A_t[-1])

#                     # A_t = [F.softmax(A_traj[0], dim=-1)]
#                     # for A_tp1 in A_traj[1:]:
#                     #     A_t.append(F.softmax(A_tp1, dim=-1) @ A_t[-1])

#                     A_t = torch.stack(A_t)
#                     pg = utils.PatchGraph(x[i, :, :].transpose(1, 2),
#                         A_t, viz=self.viz,win=pg_win, orig=orig) 
