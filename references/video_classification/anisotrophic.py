# Guided Anisotropic Diffusion algorithm
# Rodrigo Caye Daudt
# https://rcdaudt.github.io
# 
# Caye Daudt, Rodrigo, Bertrand Le Saux, Alexandre Boulch, and Yann Gousseau. "Guided anisotropic diffusion and iterative learning for weakly supervised change detection." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 0-0. 2019.

import torch


def g(x, K=5):
    return 1.0 / (1.0 + (torch.abs((x*x))/(K*K)))


def c(I, K =5):
    cv = g(torch.unsqueeze(torch.mean(I[:,:,1:,:] - I[:,:,:-1,:], 1), 1), K)
    ch = g(torch.unsqueeze(torch.mean(I[:,:,:,1:] - I[:,:,:,:-1], 1), 1), K)
    
    return cv, ch
        
    
def diffuse_step(cv, ch, I, l=0.24):
    dv = I[:,:,1:,:] - I[:,:,:-1,:]
    dh = I[:,:,:,1:] - I[:,:,:,:-1]
    
    tv = l * cv * dv # vertical transmissions
    c  = I[:,:, 1:,:] - tv
    c2 = c[:,:,:-1,:] + tv[:, :, 1:] 
    x = torch.cat([
        I[:, :, :1]   + tv[:, :, :1],
        c2,
        c[:, :, -1:] 
        ], dim=-2)

    # I[:,:,1:,:]  = I[:,:,1:,:]  - tv
    # I[:,:,:-1,:] = I[:,:,:-1,:] + tv
    
    # if I.max() > 0:
    #     import pdb; pdb.set_trace()
    # # del(dv,tv)
    
    th = l * ch * dh # horizontal transmissions

    # I[:,:,:,1:]  = I[:,:,:,1:]  - th
    # I[:,:,:,:-1] = I[:,:,:,:-1] + th
    # I2 = I

    I = x
    c  = I[:,:, :, 1:] - th
    c2 = c[:,:, :, :-1] + th[:, :, :, 1:]
    x = torch.cat([
        I[:, :, :, :1]  + th[:, :, :, :1],
        c2,
        c[:, :, :, -1:] 
        ], dim=-1)

    # assert (I2 - x).norm() < 1e-6
    # if I.max() > 0:
    #     import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()

    # del(dh,th)
    
    return x


def anisotropic_diffusion(I, N=3, l=0.24, K=5, is_log=False, verbose=False):
    if is_log:
        I = torch.exp(I)
    
    for t in range(N): 
        if verbose:
            print('Iteration {}'.format(t))

        cv, ch = c(I, K=K)
        I = diffuse_step(cv, ch, I, l=l)

        # del(cv,ch)
        
    return I
