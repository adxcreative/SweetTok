export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=0 python scripts/eval_sweettok.py config=configs/training/stage2/SweetTok_Compact_ucf101.yaml \
    experiment.init_weight="./checkpoints/SweetTok_Compact_ucf/checkpoint-50000/" \
    training.per_gpu_batch_size=4



# python3 ./evaluation/fvd_external.py --dataset ucf --gen_dir '/data/tanzhentao/titok_diff_LLM_t_s_xueben_v2_qwen/ucf_classcond_eval4096_0.9_cfg0.5_noscale/topp0.90_topk4096_2' --gt_dir '/data/tanzhentao/UCF-101/videos_split/train'  --resolution 128 --num_videos 2048
#python3 fvd_external.py --dataset ucf/k600 --gen_dir {PATH_TO_GENERATED_VIDEOS} --gt_dir {PATH_TO_GT_VIDEOS} --resolution 128/64 --num_videos 2048
