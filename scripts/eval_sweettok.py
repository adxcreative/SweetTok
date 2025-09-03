import json
import os
import time
import math
from pathlib import Path
import pprint
import glob
from collections import defaultdict
from tqdm import tqdm
import imageio

from accelerate import Accelerator

from data import SimpleImageDataset, SimpleVideoDataset
import torch
from omegaconf import OmegaConf
from modeling.modules import EMAModel, ReconstructionLoss_Stage1, ReconstructionLoss_Stage2
from modeling.sweettok import SweetTok_Compact, PretrainedTokenizer_Omni, SweetTok_base
from evaluator import VQGANEvaluator

from utils.viz_utils import make_viz_from_samples
from torchinfo import summary
from utils.train_utils import get_config
from OmniTokenizer.fvd.fvd import get_fvd_logits, frechet_distance, load_fvd_model

def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x

def eval():

    config = get_config()
    accelerator = Accelerator(
        mixed_precision = config.training.mixed_precision
    )
    device = 'cuda:0'
    print(config.experiment.init_weight)
    use_accelerate = False
    if config.eval.accelerate_load:
        use_accelerate = True
    model = SweetTok_Compact(config)
    if use_accelerate:
        model = accelerator.prepare(model)
        accelerator.load_state(config.experiment.init_weight, strict=True)
        model = accelerator.unwrap_model(model)
    else:
        model_weight = torch.load(config.experiment.init_weight, map_location="cpu")
        msg = model.load_state_dict(model_weight, strict=True)
        model.to(device)

    model.eval()
    
    if config.model.vq_model.finetune_decoder:
        pretrained_tokenizer = None
    else:
        pretrained_tokenizer = PretrainedTokenizer_Omni(config.model.vq_model.pretrained_tokenizer_weight)
        pretrained_tokenizer.to(device)

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    dataset = SimpleVideoDataset(
        train_shards_path=dataset_config.train_shards_path_or_url,
        eval_shards_path=dataset_config.eval_shards_path_or_url,
        num_train_examples=config.experiment.max_train_examples,
        per_gpu_batch_size=config.training.per_gpu_batch_size,
        global_batch_size=config.training.per_gpu_batch_size,
        num_workers_per_gpu=dataset_config.num_workers_per_gpu,
        resize_shorter_edge=preproc_config.resize_shorter_edge,
        crop_size=preproc_config.crop_size,
        random_crop=preproc_config.random_crop,
        random_flip=preproc_config.random_flip,
        normalize_mean=preproc_config.normalize_mean,
        normalize_std=preproc_config.normalize_std
    )
    train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader
    
    bias = 0.5
    save_gt = False

    bi = 0
    i3d = load_fvd_model(device)
    real_embeddings = []
    fake_embeddings = []

    for batch in tqdm(eval_dataloader):
        
        bi += 1

        images = batch["image"].to(
            device, memory_format=torch.contiguous_format, non_blocking=True
        )
        fnames = batch["__key__"]

        original_images = torch.clone(images)
        # reconstructed_images, model_dict = model(images)
        # with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        enc_tokens, encoder_dict = model.encode(original_images)
        reconstructed_images = model.decode(enc_tokens)
        if pretrained_tokenizer is not None:
            reconstructed_images = pretrained_tokenizer.decode(reconstructed_images.argmax(1))

        original_images = original_images + bias
        reconstructed_images = reconstructed_images + bias

        reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
        real_embeddings.append(get_fvd_logits(shift_dim(torch.clamp(original_images, 0.0, 1.0) * 255, 1, -1).byte().data.cpu().numpy(), i3d=i3d, device=device))
        fake_embeddings.append(get_fvd_logits(shift_dim(reconstructed_images * 255, 1, -1).byte().data.cpu().numpy(), i3d=i3d, device=device))
        reconstructed_images = torch.round(reconstructed_images * 255.0)
        original_images = torch.clamp(original_images, 0.0, 1.0) * 255.0

        # print(reconstructed_images.shape, original_images.shape)
        reconstructed_images = reconstructed_images.cpu().detach().float()
        original_images = original_images.cpu().detach().float()

        original_images_saving = [original_images[i].permute(1,0,2,3).numpy().astype('uint8') for i in range(original_images.shape[0])]
        reconstructed_images_saving = [reconstructed_images[i].permute(1,0,2,3).numpy().astype('uint8') for i in range(reconstructed_images.shape[0])]

        root = Path(config.experiment.output_dir) / "eval_images_gt"
        os.makedirs(root, exist_ok=True)

        if save_gt:
            for i,img in enumerate(original_images_saving):
                filename = f"{fnames[i]}.gif"
                path = os.path.join(root, filename)
                seqs = [img[i].transpose((1, 2, 0)) for i in range(img.shape[0])]
                imageio.mimsave(path, seqs, fps=4)

        root = Path(config.experiment.output_dir) / "eval_images_recon"
        os.makedirs(root, exist_ok=True)

        # if (bi -1) % 200 == 0:
        #     for i,img in enumerate(reconstructed_images_saving):
        #         filename = f"{fnames[i]}.gif"
        #         path = os.path.join(root, filename)
        #         seqs = [img[i].transpose((1, 2, 0)) for i in range(img.shape[0])]
        #         imageio.mimsave(path, seqs, fps=4)

        if (bi-1) % 200 == 0:
            real_embeddings_ = torch.cat(real_embeddings, 0)     
            fake_embeddings_ = torch.cat(fake_embeddings, 0) 
            print('FVD = %.2f'%(frechet_distance(fake_embeddings_, real_embeddings_)))
        

    real_embeddings = torch.cat(real_embeddings, 0)     
    fake_embeddings = torch.cat(fake_embeddings, 0) 
    print('FVD = %.2f'%(frechet_distance(fake_embeddings, real_embeddings)))

if __name__ == "__main__":
    eval()