export PYTHONPATH=$(pwd)

export TORCH_DISTRIBUTED_DEBUG=DETAIL



WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=8 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9997 --same_network \
    scripts/train_sweettok.py config=configs/training/stage2/SweetTok_Compact_ucf101.yaml \
