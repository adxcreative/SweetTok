export PYTHONPATH=$(pwd)

WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=8 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=2331 --same_network \
    scripts/train_transformer.py config=configs/training/transformer_generate.yaml \
    experiment.project="titok_generation" \
    experiment.name="titok_b64_maskgit" \
    experiment.output_dir="./runs/generate_ucf_72" \

