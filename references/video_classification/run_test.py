import os
import yaml
import socket

def test(model, L=5, K=2, T=0.01, R=12, opts=[], gpu=0, force=False):
    #--radius 40  --all-nn 
    # MODEL=moco

    # MODEL=
    # MODEL="--resume ${MODEL}"
    # model="--model-type imagenet"

    R = int(R)

    if os.path.exists(model) and 'vince_weights' not in model:
        model_str = "--model-type scratch --resume %s" % model
        model_name = '_'.join(model.split('/')[1:]) #.replace('/', '_')
    else:
        model_str = '--model-type %s' % model
        model_name = model
        # import pdb; pdb.set_trace()

    hostname = socket.gethostname()

    if hostname.startswith('kiwi'):
        outdir = '/data/ajabri/davis_dump/'   
        datapath = '/data/ajabri/davis/DAVIS/'
        davis2017path = '/data/ajabri/davis-2017/'
    elif hostname.startswith('em'):
        outdir = '/tmp/'   
        # datapath = '/home/ajabri/data/davis/DAVIS'
        datapath = '/scratch/ajabri/data/davis/'
        # davis2017path = '/home/ajabri/data/davis-2017/'
        davis2017path = '/scratch/ajabri/data/davis-2017/'
    elif hostname.startswith('b5'):
        outdir = '/data/yusun/ajabri/dump/'   
        datapath = '/data/yusun/ajabri/DAVIS/' 
        davis2017path = '/data/yusun/ajabri/davis-2017/'

    # model_name = "hardprop_fixedtemp_truerenorm_all_L%s_K%s_T%s_opts%s_M%s" %(L, K, T, ''.join(opts), model_name) 
    model_name = "fix_IN_L%s_K%s_T%s_R%s_opts%s_M%s" %(L, K, T, R, ''.join(opts), model_name) 

    if 'nopool' in model_name:
        opts += ['--no-maxpool']

    # if 'res4' in model_name or '425-noxw' in model_name: # or 'vince' in model_name:
    #     # if not 'res4-nocj-noblur' in model_name: # or True:
    #         opts += ['--use-res4']
    
    opts = ' '.join(opts)
        
    cmd = ""

    outfile = f"{outdir}/converted_{model_name}/results.yaml"
    online_str = '_online' if '--finetune' in opts else ''

    if (not os.path.isfile(outfile)) or force:
        print('Testing', model_name)
        if (not os.path.isdir(f"{outdir}/results_{model_name}")) or force:# or True:
            cmd += f'''
                python test_mem_online.py --filelist {datapath}/vallist.txt {model_str} \
                    --topk_vis {K} --radius {R}  --videoLen {L} --temperature {T} --save-path {outdir}/results_{model_name} \
                    --workers 5  {opts} --gpu-id {gpu} && \
                '''

        cmd += f'''
            python eval/davis/convert_davis.py --in_folder {outdir}/results_{model_name}/ --out_folder {outdir}/converted_{model_name}/ \
                --dataset {datapath} \
                    \
            && python {davis2017path}/python/tools/eval.py \
                -i {outdir}/converted_{model_name}/ -o {outdir}/converted_{model_name}/results.yaml \
                    --year 2017 --phase val
        '''

        print(cmd)

        os.system(f"export PYTHONPATH={davis2017path}/python/lib/; " + cmd)

    return yaml.load(open(outfile))['dataset']


def sweep(models, L, K, T, R, size, finetune, multiprocess=False, slurm=False, force=False, gpu=-1):
    import itertools

    # opts = [['--head-depth', str(-1)]] #['--radius', str(10)], ['--radius', str(5)], ['--radius', str(2.5)]] #, ['--all-nn']]
    base_opts = ['--cropSize', str(size), '--all-nn', # '--norm_mask'
    #   '--no-maxpool',
    #   '--use-res4'
    ]

    if finetune > 0:
        base_opts += ['--head-depth', str(0), '--use-res4', '--finetune', str(finetune)]
    else:
        base_opts += ['--head-depth', str(-1)]


    # opts = [base_opts + ['--radius', str(10)], base_opts + ['--radius', str(40)], base_opts + ['--radius', str(5)]]
    # opts = [base_opts + ['--radius', str(10), '--long-mem','0', '5', '10']] #, base_opts + ['--radius', str(5)]]
    opts = [base_opts]# + ['--radius', str(12)]]#, '--long-mem','0', ]] #, base_opts + ['--radius', str(5)]]
    # opts = [base_opts + ['--radius', str(10)]] #, base_opts + ['--radius', str(5)]]

    prod = list(itertools.product(models, L, K, T, R, opts))
    
    if multiprocess:
        import multiprocessing
        pool = multiprocessing.Pool(3)

        for i in range(0, len(prod), 3):
            results = []

            for j in range(3):
                if i+j < len(prod):
                    print((*prod[i+j], j))
                    result = pool.apply_async(test, (*prod[i+j], j, force))
                    results.append(result)

                    # test(*prod[i+j], j)

            [result.wait() for result in results]

    elif slurm:
        for p in prod:

            cmd = f"sbatch --export=model_path={p[0]},L={p[1]},K={p[2]},T={p[3]},R={p[4]},size={size},finetune={finetune} /home/ajabri/slurm/davis_test.sh"
            print(cmd)

            os.system(cmd)
            # import pdb; pdb.set_trace()
            # test(*prod, 0, force)
    else:
        print(prod)
        for i in range(0, len(prod)):
            test(*prod[i], 0 if gpu == -1 else gpu, force)



def serve(checkpoint_dir):
    import time

    while True:
        time.sleep(10)

        for f in os.listdir(checkpoint_dir):
            model_path = "%s/%s" %(checkpoint_dir, f)
            
            sweep(model_path, multiprocess=False)


                        

if __name__ == '__main__':
    # MODEL=checkpoints/longcycle_mlp_datasetkinetics_dropout0.1_clip_len6_frame_transformscrop+cj_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.1_lr0.0003/model_1.pth
    # MODEL = "checkpoints/vlog_datasetpennaction_dropout0.1_clip_len6_frame_transformscj+crop_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.1_lr0.0003/model_11.pth"
    #MODEL=checkpoints/vlog_datasetpennaction_dropout0.0_clip_len6_frame_transformscj+crop
    # _frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.5_lr0.0003/model_10.pth


    # models += 
    models = [
        # "checkpoints/longcycle_mlp_datasetkinetics_dropout0.25_clip_len6_frame_transformscrop+cj_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.1_lr0.0003/model_4.pth",
        # "checkpoints/vlog_datasetpennaction_dropout0.1_clip_len6_frame_transformscj+crop_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.1_lr0.0003/model_%s.pth" \
        # "checkpoints/vlog_datasetpennaction_dropout0.0_clip_len6_frame_transformscj+crop_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.5_lr0.0003/model_%s.pth"
        #     % i for i in [1, 4, 7, 10, 13, 16]
        # "checkpoints/vlog_datasetpennaction_dropout0.1_clip_len6_frame_transformscj+crop_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.1_lr0.0003/model_11.pth",
        # 'imagenet', 'moco',    
        "checkpoints/test-xentweight_drop0.1-len6-ftranscj+crop+blur-fauggrid-zdiagFalse-pstride0.5-0.5-optimadam-temp0.08-fdrop0.1-lr0.0001-skip0.0-mlp0/model_0.pth",
        # "checkpoints/alt1-spacel2-flipped-dropcon_drop0.1-len6-ftranscj+crop+blur-fauggrid-zdiagFalse-pstride0.5-0.5-optimadam-temp0.08-fdrop0.1-lr0.0003-skip0.0-mlp0/model_0.pth"
    ]
    # models = [
    #     "checkpoints/longcycle_reload_lr1e-4_datasetkinetics_dropout0.1_clip_len6_frame_transformscrop+cj_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.1_lr0.0001/model_2.pth"
    # ]

    # models = [
    #     # "checkpoints/test_no_shuff_datasetpennaction_dropout0.1_clip_len4_frame_transformscj+crop+blur_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.25-0.25_edgefuncsoftmax_optimsgd_temp0.08_featdrop0.0_lr0.0003/model_5.pth"
    #     "checkpoints/test_no_shuff_datasetpennaction_dropout0.1_clip_len3_frame_transformscj+crop+blur_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.0_lr0.0003/model_4.pth"
    # ]

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint-dir', default='./test_checkpoints/', type=str)
    parser.add_argument('--model-path', default=[], type=str, nargs='+',)

    parser.add_argument('--slurm', default=False, action='store_true')
    
    parser.add_argument('--multiprocess', default=False, action='store_true')
    parser.add_argument('--force', default=False, action='store_true')

    parser.add_argument('--L', default=[3], type=int, nargs='+')
    parser.add_argument('--K', default=[2], type=int, nargs='+')
    parser.add_argument('--T', default=[0.1], type=float, nargs='+')
    parser.add_argument('--R', default=[12], type=float, nargs='+')
    parser.add_argument('--finetune', default=0, type=int)

    parser.add_argument('--cropSize', default=480, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    
    args = parser.parse_args()
    
    if len(args.model_path) == 0:
        args.model_path = models

    if args.slurm:
        sweep(args.model_path, args.L, args.K, args.T, args.R, args.cropSize, args.finetune,
            slurm=args.slurm,
            force=args.force)
        
    else:
        sweep(args.model_path, args.L, args.K, args.T, args.R, args.cropSize, args.finetune,
            multiprocess=args.multiprocess,
            force=args.force,
            gpu=args.gpu)
