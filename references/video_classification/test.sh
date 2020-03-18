MODEL=checkpoints/longcycle_mlp_datasetkinetics_dropout0.1_clip_len6_frame_transformscrop+cj_frame_auggrid_zero_diagonalFalse_npatch5_nrel10_pstride0.5-0.5_edgefuncsoftmax_optimadam_temp0.08_featdrop0.1_lr0.0003/model_1.pth

python test.py --filelist /data/ajabri/davis/DAVIS/vallist.txt  --resume ${MODEL} \
  --topk_vis 2   --videoLen 5 --temperature 0.001 --radius 4 --all-nn --visdom   --workers 5 \
&& python davis/convert_davis.py --in_folder ./results/ --out_folder ./result_converted/ --dataset /data/ajabri/davis/DAVIS/ \
&& python /data/ajabri/davis-2017/python/tools/eval.py -i ./result_converted -o ./result_converted/results.yaml --year 2017 --phase val