#!/bin/bash
# MODEL=checkpoints/longcycle_mlp_datasetkinetics_dropout0.1_clip_len6_frame_transformscrop+cj_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.1_lr0.0003/model_1.pth
 MODEL=checkpoints/vlog_datasetpennaction_dropout0.1_clip_len6_frame_transformscj+crop_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.1_lr0.0003/model_11.pth
#MODEL=checkpoints/vlog_datasetpennaction_dropout0.0_clip_len6_frame_transformscj+crop_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.5_lr0.0003/model_10.pth
 #--radius 40  --all-nn 
# MODEL=moco

# MODEL=
# MODEL="--resume ${MODEL}"
MODEL="--model-type imagenet"

for len in 5
do
  for temp in 0.01 0.001
  do
    for topk in 1 2 3
      
      python test.py --filelist /data/ajabri/davis/DAVIS/vallist.txt   ${MODEL} \
        --topk_vis 2   --videoLen ${len} --temperature 0.001 \
        --visdom   --workers 5 --head-depth -1 \
      && python davis/convert_davis.py --in_folder ./results/ --out_folder ./result_converted/ --dataset /data/ajabri/davis/DAVIS/ \
      && python /data/ajabri/davis-2017/python/tools/eval.py -i ./result_converted -o ./result_converted/results.yaml --year 2017 --phase val

    done
  done
done