import torch
import numpy as np
from matplotlib import cm
import cv2

def vis_pose(oriImg, points):

    pa = np.zeros(15)
    pa[2] = 0
    pa[12] = 8
    pa[8] = 4
    pa[4] = 0
    pa[11] = 7
    pa[7] = 3
    pa[3] = 0
    pa[0] = 1
    pa[14] = 10
    pa[10] = 6
    pa[6] = 1
    pa[13] = 9
    pa[9] = 5
    pa[5] = 1

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170,0,255],[255,0,255]]
    canvas = oriImg
    stickwidth = 4
    x = points[0, :]
    y = points[1, :]

    for n in range(len(x)):
        pair_id = int(pa[n])

        x1 = int(x[pair_id])
        y1 = int(y[pair_id])
        x2 = int(x[n])
        y2 = int(y[n])

        if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
            cv2.line(canvas, (x1, y1), (x2, y2), colors[n], 8)

    return canvas


def hard_prop(predlbls):
    pred_max = predlbls.max(axis=0)[0]
    predlbls[predlbls <  pred_max] = 0
    predlbls[predlbls >= pred_max] = 1
    predlbls /= predlbls.sum(0)[None]
    return predlbls

def process_pose(predlbls, lbl_set, topk=3):
    # generate the coordinates:
    predlbls = predlbls[..., 1:]
    flatlbls = predlbls.flatten(0,1)
    topk = min(flatlbls.shape[0], topk)
    
    vals, ids = torch.topk(flatlbls, k=topk, dim=0)
    vals /= vals.sum(0)[None]
    xx, yy = ids % predlbls.shape[1], ids // predlbls.shape[1]

    current_coord = torch.stack([(xx * vals).sum(0), (yy * vals).sum(0)], dim=0)
    current_coord[:, flatlbls.sum(0) == 0] = -1

    predlbls_val_sharp = np.zeros((*predlbls.shape[:2], 3))

    for t in range(len(lbl_set) - 1):
        x = int(current_coord[0, t])
        y = int(current_coord[1, t])

        if x >=0 and y >= 0:
            predlbls_val_sharp[y, x, :] = lbl_set[t + 1]

    return current_coord.cpu(), predlbls_val_sharp


def dump_predictions(predlbls, lbl_set, img_now, prefix):
    sz = img_now.shape[:-1]

    predlbls_cp = predlbls.copy()
    predlbls_cp = cv2.resize(predlbls_cp, sz[::-1])[:]
    
    predlbls_val = np.zeros((*sz, 3))

    ids = np.argmax(predlbls_cp[:, :, 1 : len(lbl_set)], 2)

    predlbls_val = np.argmax(predlbls_cp, axis=-1)
    predlbls_val = np.array(lbl_set, dtype=np.int32)[predlbls_val]      

    predlbls_val2 = cv2.resize(predlbls_val, (img_now.shape[1], img_now.shape[0]), interpolation=cv2.INTER_NEAREST)

    # activation_heatmap = cv2.applyColorMap(predlbls, cv2.COLORMAP_JET)
    img_with_heatmap =  np.float32(img_now) * 0.5 + np.float32(predlbls_val2) * 0.5

    predlbls_soft = predlbls_cp[..., 1]
    predlbls_soft = cv2.resize(predlbls_soft, (img_now.shape[1], img_now.shape[0]), interpolation=cv2.INTER_NEAREST)
    predlbls_soft = cm.jet(predlbls_soft)[..., :3] * 255.0
    img_with_heatmap2 =  np.float32(img_now) * 0.5 + np.float32(predlbls_soft) * 0.5

    imname  = prefix + '_blend.jpg'
    imageio.imwrite(imname, np.uint8(img_with_heatmap))

    if prefix[-4] != '.':
        imname2 = prefix + '_mask.png'
        # skimage.io.imsave(imname2, np.uint8(predlbls_val))
    else:
        imname2 = prefix.replace('jpg','png')
        
        # predlbls_val = np.uint8(predlbls_val)

        # if predlbls_val.max() > 20:#: or :
        #     import pdb; pdb.set_trace()
    
        # skimage.io.imsave(imname2.replace('jpg','png'), predlbls_val)

    imageio.imwrite(imname2, np.uint8(predlbls_val))

    return img_with_heatmap, predlbls_val, img_with_heatmap2



def context_index_bank(n_context, long_mem, N):
    ll = []
    
    for t in long_mem:
        idx = torch.zeros(N, 1).long()
        if t > 0:
            assert t < N
            idx += t + (n_context+1)
            idx[:n_context+t+1] = 0

        ll.append(idx)

    ss = [(torch.arange(n_context)[None].repeat(N, 1) +  torch.arange(N)[:, None])[:, :]]

    return ll + ss


def mem_efficient_batched_affinity(query, keys, mask, temperature, topk, long_mem, device):
    bsize, pbsize = 2, 100 #keys.shape[2] // 2
    Ws, Is = [], []

    for b in range(0, keys.shape[2], bsize):
        _k, _q = keys[:, :, b:b+bsize].to(device), query[:, :, b:b+bsize].to(device)
        w_s, i_s = [], []

        for pb in range(0, _k.shape[-1], pbsize):
            A = torch.einsum('ijklm,ijkn->iklmn', _k, _q[..., pb:pb+pbsize]) 
            A[0, :, len(long_mem):] += mask[..., pb:pb+pbsize].to(device)

            _, N, T, h1w1, hw = A.shape
            A = A.view(N, T*h1w1, hw)
            A /= temperature

            weights, ids = torch.topk(A, topk, dim=-2)
            weights = F.softmax(weights, dim=-2)
            
            w_s.append(weights.cpu())
            i_s.append(ids.cpu())

        weights = torch.cat(w_s, dim=-1)
        ids = torch.cat(i_s, dim=-1)
        Ws += [w for w in weights]
        Is += [ii for ii in ids]

def batched_affinity(query, keys, mask, temperature, topk, long_mem, device):
    bsize = 2
    Ws, Is = [], []
    for b in range(0, keys.shape[2], bsize):
        _k, _q = keys[:, :, b:b+bsize].to(device), query[:, :, b:b+bsize].to(device)
        w_s, i_s = [], []

        A = torch.einsum('ijklmn,ijkop->iklmnop', _k, _q) / temperature
        
        # Mask
        A[0, :, len(long_mem):] += mask.to(device)

        _, N, T, h1w1, hw = A.shape
        A = A.view(N, T*h1w1, hw)
        A /= temperature

        weights, ids = torch.topk(A, topk, dim=-2)
        weights = F.softmax(weights, dim=-2)
            
        Ws += [w for w in weights]
        Is += [ii for ii in ids]

    return Ws, Is

