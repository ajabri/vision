import os

def test(model, L, K, T, opts, gpu=0):
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


    outdir = '/data/ajabri/davis_dump/'
    model_name = "L%s_K%s_T%s__M%s_%s" %(L, K, T, model_name, ''.join(opts)) 

    opts = ' '.join(opts)

    cmd = ""

    if not os.path.isdir(f"{outdir}/results_{model_name}"):# or True:
        cmd += '''
            python test.py --filelist /data/ajabri/davis/DAVIS/vallist.txt {model_str} \
                --topk_vis {K}   --videoLen {L} --temperature {T} --save-path {outdir}/results_{model_name} \
                --workers 5  {opts} --head-depth -1 --gpu-id {gpu} && \
            '''.format(model_str=model_str, model_name=model_name, K=K, L=L, T=T, gpu=gpu, outdir=outdir, opts=opts)

    cmd += '''
         python davis/convert_davis.py --in_folder {outdir}/results_{model_name}/ --out_folder {outdir}/converted_{model_name}/ \
            --dataset /data/ajabri/davis/DAVIS/ \
                \
        && python /data/ajabri/davis-2017/python/tools/eval.py \
            -i {outdir}/converted_{model_name}/ -o {outdir}/converted_{model_name}/results.yaml \
                --year 2017 --phase val
    '''.format(model_str=model_str, model_name=model_name, K=K, L=L, T=T, gpu=gpu, outdir=outdir)

    # print(cmd)
    os.system('export PYTHONPATH=/data/ajabri/davis-2017/python/lib/')
    os.system(cmd)

def sweep(models):
    import itertools


    L = [5]
    K = [2]#, 3]
    T = [0.01]#, 0.05]
    opts = [['']]#, ['--all-nn']]

    prod = list(itertools.product(models, L, K, T, opts))

    import multiprocessing
    pool = multiprocessing.Pool(3)

    for i in range(0, len(prod), 3):
        results = []

        for j in range(3):
            if i+j < len(prod):
                print((*prod[i+j], j))
                result = pool.apply_async(test, (*prod[i+j], j))
                results.append(result)

        [result.wait() for result in results]

# MODEL=checkpoints/longcycle_mlp_datasetkinetics_dropout0.1_clip_len6_frame_transformscrop+cj_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.1_lr0.0003/model_1.pth
# MODEL = "checkpoints/vlog_datasetpennaction_dropout0.1_clip_len6_frame_transformscj+crop_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.1_lr0.0003/model_11.pth"
#MODEL=checkpoints/vlog_datasetpennaction_dropout0.0_clip_len6_frame_transformscj+crop_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.5_lr0.0003/model_10.pth


# models += 
models = [
    # "checkpoints/longcycle_mlp_datasetkinetics_dropout0.25_clip_len6_frame_transformscrop+cj_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.1_lr0.0003/model_4.pth",
    # "checkpoints/vlog_datasetpennaction_dropout0.1_clip_len6_frame_transformscj+crop_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.1_lr0.0003/model_%s.pth" \
    "checkpoints/vlog_datasetpennaction_dropout0.0_clip_len6_frame_transformscj+crop_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.5_lr0.0003/model_%s.pth"
        % i for i in [1, 4, 7, 10, 13, 16]
    # "checkpoints/vlog_datasetpennaction_dropout0.1_clip_len6_frame_transformscj+crop_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.1_lr0.0003/model_24.pth",
    # 'imagenet', 'moco',    
]
# models = [
#     "checkpoints/longcycle_reload_lr1e-4_datasetkinetics_dropout0.1_clip_len6_frame_transformscrop+cj_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.1_lr0.0001/model_2.pth"
# ]
sweep(models)