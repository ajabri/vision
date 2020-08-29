import argparse
import os
import torch
import random

def common_args(parser):
    return parser

def test_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Label Propagation')

    # Datasets
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--manualSeed', type=int, default=777, help='manual seed')

    #Device options
    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--batchSize', default=1, type=int,
                        help='batchSize')
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='temperature')

    parser.add_argument('--topk_vis', default=20, type=int,
                        help='k for kNN')
    parser.add_argument('--radius', default=12, type=float,
                        help='spatial radius to consider neighbors from')
    parser.add_argument('--all-nn', default=False, action='store_true',
                        help='do nn over all context frames')
    parser.add_argument('--videoLen', default=4, type=int,
                        help='number of context frames')

    parser.add_argument('--cropSize', default=320, type=int,
                        help='size of test image')

    parser.add_argument('--filelist', default='/scratch/ajabri/data/davis/val2017.txt', type=str)
    parser.add_argument('--save-path', default='./results', type=str)

    parser.add_argument('--visdom', default=False, action='store_true')
    parser.add_argument('--visdom-server', default='localhost', type=str)

    # Model Details
    parser.add_argument('--model-type', default='scratch', type=str)
    parser.add_argument('--head-depth', default=0, type=int,
                        help='')

    parser.add_argument('--no-maxpool', default=False, action='store_true', help='')
    parser.add_argument('--use-res4', default=False, action='store_true', help='')
    parser.add_argument('--no-l2', default=False, action='store_true', help='')

    parser.add_argument('--long-mem', default=[0], type=int, nargs='*', help='')
    parser.add_argument('--texture', default=False, action='store_true', help='')
    parser.add_argument('--round', default=False, action='store_true', help='')

    parser.add_argument('--norm_mask', default=False, action='store_true', help='')

    parser.add_argument('--finetune', default=0, type=int, help='')

    args = parser.parse_args()

    # CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Using GPU', args.gpu_id)
        args.device = 'cuda:%s' % args.gpu_id
    else:
        args.device = 'cpu'

    # Set seed
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

    return args

def train_args():
    parser = argparse.ArgumentParser(description='CRaWl Training')

    parser.add_argument('--data-path', default='/data/ajabri/kinetics/',
        help='/home/ajabri/data/places365_standard/train/ | /data/ajabri/kinetics/')

    parser.add_argument('--model', default='r3d_18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--clip-len', default=8, type=int, metavar='N',
                        help='number of frames per clip')
    parser.add_argument('--clips-per-video', default=5, type=int, metavar='N',
                        help='maximum number of clips per video to consider')
    parser.add_argument('-b', '--batch-size', default=24, type=int)
    parser.add_argument('--epochs', default=45, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--steps-per-epoch', default=1e10, type=int, help='max number of batches per epoch')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--lr-milestones', nargs='+', default=[20, 30, 40], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.3, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=0, type=int, help='number of warmup epochs')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='auto', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--reload', default='', help='reload net from checkpoint without optimizer state')
    parser.add_argument('--partial-reload', default='', help='reload net from checkpoint, ignoring keys that are not in current model')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    
    parser.add_argument( "--cache-dataset", dest="cache_dataset", help="Cache the datasets for quicker initialization. It also serializes the transforms", action="store_true", )
    parser.add_argument( "--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true", )
    parser.add_argument( "--test-only", dest="test_only", help="Only test the model", action="store_true", )

    parser.add_argument( "--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo", action="store_true", )

    parser.add_argument( "--data-parallel", dest="data_parallel", help="", action="store_true", )

    parser.add_argument( "--zero-diagonal", dest="zero_diagonal", help="", action="store_true", )

    parser.add_argument( "--fast-test", dest="fast_test", help="", action="store_true", )

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str, help='For apex mixed precision training' 'O0 for FP32 training, O1 for mixed precision training.' 'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet' )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')


    # my args
    parser.add_argument('--xent-coef', default=1, type=float, help='initial learning rate')
    parser.add_argument('--kldv-coef', default=0, type=float, help='initial learning rate')
    parser.add_argument('--dropout', default=0, type=float, help='dropout rate on A')
    parser.add_argument('--name', default='', type=str, help='')

    parser.add_argument('--frame-transforms', default='crop', type=str,
        help='blur ^ crop ^ cj ^ flip')
    parser.add_argument('--frame-aug', default='', type=str,
        help='(randpatch | grid) + cj ^ flip')

    parser.add_argument('--frame-skip', default=8, type=int, help='number of frames to skip when sampling')

    parser.add_argument('--img-size', default=256, type=int, help='number of patches sampled')
    parser.add_argument('--patch-size', default=[64, 64, 3], type=int, nargs="+", help='number of patches sampled')
    parser.add_argument('--nrel', default=10, type=int, help='number of heads')

    parser.add_argument('--port', default=8095, type=int, help='number of heads')
    parser.add_argument('--server', default='localhost', type=str, help='number of heads')
    parser.add_argument('--npatch', default=5, type=int, help='number of patches sampled')

    # patchifying args
    parser.add_argument('--pstride', default=[0.5, 0.5], nargs=2,
        type=float, help='random sample patchify stride from [this*patch_size, patch_size]')
    parser.add_argument('--npatch-scale', default=[0.2, 0.8], nargs=2,
        type=float, help='range from which to same patch sizes for random patch sampling')
    parser.add_argument('--edgefunc', default='softmax', type=str, help='softmax | relu')

    parser.add_argument('--model-type', default='scratch', type=str, help='scratch | imagenet | moco')
    parser.add_argument('--optim', default='adam', type=str, help='adam | sgd')

    parser.add_argument('--temp', default=0.08,
        type=float, help='softmax temperature when computing affinity')

    parser.add_argument('--featdrop', default=0.0,
        type=float, help='dropout on features')

    parser.add_argument('--restrict', default=-1,
        type=int, help='dropout on features')

    parser.add_argument('--head-depth', default=0,
        type=int, help='depth of head mlp')

    parser.add_argument('--visualize', default=False,
        action='store_true', help='visualize trained model')

    parser.add_argument('--long-coef', default=1,
        type=float, help='long cycle loss coef')
    parser.add_argument('--skip-coef', default=0,
        type=float, help='skip cycle loss coef')
    parser.add_argument('--cal-coef', default=0.0,
        type=float, help='contrastive affinity')

    parser.add_argument('--shuffle', default=0.0,
        type=float, help='shuffle patches across instances for different negatives')

    parser.add_argument('--xent-weight', default=False, action='store_true',
        help='use out-going entropy * max similarity as a loss gate')

    parser.add_argument('--no-maxpool', default=False, action='store_true',
        help='')
    parser.add_argument('--use-res4', default=False, action='store_true',
        help='')

    args = parser.parse_args()

    if args.fast_test:
        args.batch_size = 1
        args.workers = 0
        args.data_parallel = False

    if args.output_dir == 'auto':
        args.dataset = 'kinetics' if 'kinetics' in args.data_path else 'pennaction'
        keys = {
            'dropout':'drop', 'clip_len': 'len', 'frame_transforms': 'ftrans', 'frame_aug':'faug', 'zero_diagonal':'zdiag', 
            'pstride':'pstride', 'optim':'optim', 'temp':'temp',
            'featdrop':'fdrop', 'lr':'lr', 'skip_coef':'skip', 'head_depth':'mlp'
        }
        name = '-'.join(["%s%s" % (keys[k], getattr(args, k) if not isinstance(getattr(args, k), list) else '-'.join([str(s) for s in getattr(args, k)])) for k in keys])
        args.output_dir = "checkpoints/%s_%s/" % (args.name, name)

        import datetime
        dt = datetime.datetime.today()
        args.name = "%s-%s--%s_%s" % (str(dt.month), str(dt.day), args.name, name)

    return args




