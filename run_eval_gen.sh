export PYTHONPATH=$(pwd)

python3 -m torch.distributed.run --nproc_per_node=8 --master_port 42914 transformer_eval.py --inference_type "video" \
                      --gpt_ckpt GEN_CKPT_PATH --seed 2000 \
                      --batch_size 1 --save SAVE_PATH --n_sample 10000 --class_cond --cfg_ratio 0.5 --no_scale_cfg \
                      --top_k 4096 --top_p 0.9  --config 'configs/training/transformer_generate.yaml' --config_model 'configs/training/stage2/SweetTok_Compact_ucf101.yaml'

