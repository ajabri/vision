import os
import yaml
import socket

def test(model, L=5, K=2, T=0.01, opts=[], gpu=0):
    #--radius 40  --all-nn 
    # MODEL=moco

    # MODEL=
    # MODEL="--resume ${MODEL}"
    # model="--model-type imagenet"

    if os.path.exists(model):
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
    else:
        outdir = '/scratch/ajabri/data/davis_dump/'   
        datapath = '/scratch/ajabri/data/davis/'
        davis2017path = '/scratch/ajabri/data/davis-2017/'

    model_name = "L%s_K%s_T%s__M%s_%s" %(L, K, T, model_name, ''.join(opts)) 

    opts = ' '.join(opts)
    cmd = ""

    if not os.path.isdir(f"{outdir}/results_{model_name}") or True:
        cmd += f'''
            python test.py --filelist {datapath}/vallist.txt {model_str} \
                --topk_vis {K}   --videoLen {L} --temperature {T} --save-path {outdir}/results_{model_name} \
                --workers 5  {opts} --head-depth -1 --gpu-id {gpu} && \
            '''
            #.format(model_str=model_str, model_name=model_name, K=K, L=L, T=T, gpu=gpu, outdir=outdir, opts=opts)

    cmd += f'''
         python davis/convert_davis.py --in_folder {outdir}/results_{model_name}/ --out_folder {outdir}/converted_{model_name}/ \
            --dataset {datapath} \
                \
        && python {davis2017path}/python/tools/eval.py \
            -i {outdir}/converted_{model_name}/ -o {outdir}/converted_{model_name}/results.yaml \
                --year 2017 --phase val
    '''#.format(model_str=model_str, model_name=model_name, K=K, L=L, T=T, gpu=gpu, outdir=outdir)

    os.system(f"PYTHONPATH={davis2017path}/python/lib/ " + cmd)

    outfile = f"{outdir}/converted_{model_name}/results.yaml"

    return yaml.load(open(outfile))['dataset']

def sweep(models):
    import itertools


    L = [3]
    K = [2]
    T = [0.1]#, 0.05]

    # opts = [['--head-depth', str(-1)]] #['--radius', str(10)], ['--radius', str(5)], ['--radius', str(2.5)]] #, ['--all-nn']]
    opts = [['--cropSize', str(320), '--head-depth', str(-1)]]
    prod = list(itertools.product(models, L, K, T, opts))

    import multiprocessing
    pool = multiprocessing.Pool(3)

    for i in range(0, len(prod), 3):
        results = []

        for j in range(3):
            if i+j < len(prod):
                print((*prod[i+j], j))
                result = pool.apply_async(test, (*prod[i+j], 1))
                results.append(result)

                # test(*prod[i+j], j)

        [result.wait() for result in results]



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
        # "checkpoints/alt1_drop0.2-len6-ftranscj+crop+blur-fauggrid-zdiagFalse-pstride0.5-0.5-optimadam-temp0.08-fdrop0.1-lr0.0003-skip0.0-mlp0/checkpoint.pth"
        "checkpoints/alt1_drop0.1-len6-ftranscj+crop+blur-fauggrid-zdiagFalse-pstride0.5-0.5-optimadam-temp0.08-fdrop0.1-lr0.0003-skip0.0-mlp0/model_0.pth"
        # 'imagenet', 'moco',    
    ]
    # models = [
    #     "checkpoints/longcycle_reload_lr1e-4_datasetkinetics_dropout0.1_clip_len6_frame_transformscrop+cj_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.1_lr0.0001/model_2.pth"
    # ]

    # models = [
    #     # "checkpoints/test_no_shuff_datasetpennaction_dropout0.1_clip_len4_frame_transformscj+crop+blur_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.25-0.25_edgefuncsoftmax_optimsgd_temp0.08_featdrop0.0_lr0.0003/model_5.pth"
    #     "checkpoints/test_no_shuff_datasetpennaction_dropout0.1_clip_len3_frame_transformscj+crop+blur_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.0_lr0.0003/model_4.pth"
    # ]
    sweep(models)
