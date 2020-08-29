import os
import numpy as np
import scipy.misc
import cv2

from PIL import Image

jpglist = []





import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out_folder', default='/scratch/ajabri/davis_results/', type=str)
parser.add_argument('-i', '--in_folder', default='/scratch/ajabri/davis_results_mask_sep/', type=str)
parser.add_argument('-d', '--dataset', default='/scratch/ajabri/data/davis/', type=str)

args = parser.parse_args()

annotations_folder = args.dataset + '/Annotations/480p/'
f1 = open(args.dataset + '/ImageSets/2017/val.txt', 'r')
for line in f1:
    line = line[:-1]
    jpglist.append(line)
f1.close()

out_folder = args.out_folder
current_folder = args.in_folder
palette_list = 'davis/palette.txt'

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

f = open(palette_list, 'r')
palette = np.zeros((256, 3))
cnt = 0
for line in f:
    rows = line.split()
    palette[cnt][0] = int(rows[0])
    palette[cnt][1] = int(rows[1])
    palette[cnt][2] = int(rows[2])
    cnt = cnt + 1

f.close()
palette = palette.astype(np.uint8)
def color2id(c):
    return np.arange(0, palette.shape[0])[np.all(palette == c, axis=-1)]

topk = 0

def convert_dir(i):
    fname = jpglist[i]
    gtfolder = annotations_folder + fname + '/'
    outfolder = out_folder + fname + '/'

    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    files = [_ for _ in os.listdir(gtfolder) if _[-4:] == '.png']

    firstim = gtfolder + "{:05d}.png".format(0)
    lblimg  = cv2.imread(firstim)

    height = lblimg.shape[0]
    width  = lblimg.shape[1]
    # scipy.misc.imsave(outfolder + "{:05d}.png".format(0), np.uint8(lblimg))


    # lblimg = Image.fromarray(np.uint8(lblimg))
    # lblimg = lblimg.convert('P')
    # lblimg.save(outfolder + "{:05d}.png".format(0), format='PNG')


    for j in range(len(files)):
        outname = outfolder + "{:05d}.png".format(j + 1)
        inname  = current_folder + str(i) + '_' + str(j + topk) + '_mask.png'

        # print(inname, outname)
        lblimg  = cv2.imread(inname)
        # print(inname)
        flat_lblimg = lblimg.reshape(-1, 3)
        lblidx  = np.zeros((lblimg.shape[0], lblimg.shape[1]))
        lblidx2  = np.zeros((lblimg.shape[0], lblimg.shape[1]))

        colors = np.unique(flat_lblimg, axis=0)

        # for h in range(lblimg.shape[0]):
        #     for w in range(lblimg.shape[1]):
        #         nowlbl = lblimg[h, w, :]
        #         idx = 0
        #         for t in range(len(palette)):
        #             if palette[t][0] == nowlbl[0] and palette[t][1] == nowlbl[1] and palette[t][2] == nowlbl[2]:
        #                 idx = t
        #                 break
        #         lblidx[h, w] = idx

        # import pdb; pdb.set_trace()
        for c in colors:
            cid = color2id(c)
            if len(cid) > 0:
                lblidx2[np.all(lblimg == c, axis=-1)] = cid

        lblidx = lblidx2
        # assert (lblidx != lblidx2).sum() == 0
        # import pdb; pdb.set_trace()

        lblidx = lblidx.astype(np.uint8)
        lblidx = cv2.resize(lblidx, (width, height), interpolation=cv2.INTER_NEAREST)
        lblidx = lblidx.astype(np.uint8)

        # lblidx  = scipy.misc.imresize( lblidx, (height, width), 'nearest' )
        # lblidx  = lblidx.reshape((height, width, 1))
        im = Image.fromarray(lblidx)
        im.putpalette(palette.ravel())
        im.save(outname, format='PNG')


        # scipy.misc.imsave(outname, np.uint8(lblimg))

import multiprocessing as mp
pool = mp.Pool(processes=10)
results = pool.map(convert_dir, range(len(jpglist)))
# print(results)
