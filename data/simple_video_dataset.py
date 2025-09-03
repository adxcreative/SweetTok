import math
from typing import List, Union, Text
import webdataset as wds
import torch
from torch.utils.data import default_collate
from torchvision import transforms
import torch.distributed as dist

from .video_dataset import DecordVideoDataset, DecordVideoDataset_indice


class SimpleVideoDataset:
    def __init__(
        self,
        train_shards_path: Union[Text, List[Text]],
        eval_shards_path: Union[Text, List[Text]],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers_per_gpu: int = 12,
        resize_shorter_edge: int = 256,
        crop_size: int = 256,
        random_crop = True,
        random_flip = True,
        normalize_mean: List[float] = [0., 0., 0.],
        normalize_std: List[float] = [1., 1., 1.],
    ):
        """Initializes the WebDatasetReader class.

        Args:
            train_shards_path: A string or list of string, path to the training data shards in webdataset format.
            eval_shards_path: A string or list of string, path to the evaluation data shards in webdataset format.
            num_train_examples: An integer, total number of training examples.
            per_gpu_batch_size: An integer, number of examples per GPU batch.
            global_batch_size: An integer, total number of examples in a batch across all GPUs.
            num_workers_per_gpu: An integer, number of workers per GPU.
            resize_shorter_edge: An integer, the shorter edge size to resize the input image to.
            crop_size: An integer, the size to crop the input image to.
            random_crop: A boolean, whether to use random crop augmentation during training.
            random_flip: A boolean, whether to use random flipping augmentation during training.
            normalize_mean: A list of float, the normalization mean used to normalize the image tensor.
            normalize_std: A list of float, the normalization std used to normalize the image tensor.
        """

        num_batches = math.ceil(num_train_examples / global_batch_size)
        num_worker_batches = math.ceil(num_train_examples / 
            (global_batch_size * num_workers_per_gpu))
        num_batches = num_worker_batches * num_workers_per_gpu
        num_samples = num_batches * global_batch_size

        # Each worker is iterating over the complete dataset.
        self._train_dataset = DecordVideoDataset(train_shards_path[0], train_shards_path[1], -1, 17, 
                                    train=True, resolution=crop_size, resizecrop=False,
                                    normalize_mean=normalize_mean, normalize_std=normalize_std)

                                    
        if dist.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                self._train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
            )
        else:
            sampler = None
        self._train_dataloader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=per_gpu_batch_size,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            sampler=sampler,
            shuffle=False
        )

        # Add meta-data to dataloader instance for convenience.
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

        # Create eval dataset and loader.
        self._eval_dataset = DecordVideoDataset(eval_shards_path[0], eval_shards_path[1], -1, 17, 
                                    train=False, resolution=crop_size, resizecrop=False,
                                    normalize_mean=normalize_mean, normalize_std=normalize_std)
                                    
        if dist.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                self._eval_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
            )
        else:
            sampler = None
        self._eval_dataloader = torch.utils.data.DataLoader(
            self._eval_dataset,
            batch_size=per_gpu_batch_size,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            sampler=sampler,
            shuffle=False
        )

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def eval_dataset(self):
        return self._eval_dataset

    @property
    def eval_dataloader(self):
        return self._eval_dataloader


class SimpleVideoGenerationDataset:
    def __init__(
        self,
        train_shards_path: Union[Text, List[Text]],
        eval_shards_path: Union[Text, List[Text]],
        indice_path: Union[Text, List[Text]],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers_per_gpu: int = 12,
        resize_shorter_edge: int = 256,
        crop_size: int = 256,
        random_crop = True,
        random_flip = True,
        normalize_mean: List[float] = [0., 0., 0.],
        normalize_std: List[float] = [1., 1., 1.],
    ):
        """Initializes the WebDatasetReader class.

        Args:
            train_shards_path: A string or list of string, path to the training data shards in webdataset format.
            eval_shards_path: A string or list of string, path to the evaluation data shards in webdataset format.
            num_train_examples: An integer, total number of training examples.
            per_gpu_batch_size: An integer, number of examples per GPU batch.
            global_batch_size: An integer, total number of examples in a batch across all GPUs.
            num_workers_per_gpu: An integer, number of workers per GPU.
            resize_shorter_edge: An integer, the shorter edge size to resize the input image to.
            crop_size: An integer, the size to crop the input image to.
            random_crop: A boolean, whether to use random crop augmentation during training.
            random_flip: A boolean, whether to use random flipping augmentation during training.
            normalize_mean: A list of float, the normalization mean used to normalize the image tensor.
            normalize_std: A list of float, the normalization std used to normalize the image tensor.
        """

        num_batches = math.ceil(num_train_examples / global_batch_size)
        num_worker_batches = math.ceil(num_train_examples / 
            (global_batch_size * num_workers_per_gpu))
        num_batches = num_worker_batches * num_workers_per_gpu
        num_samples = num_batches * global_batch_size

        # Each worker is iterating over the complete dataset.

        self._train_dataset = DecordVideoDataset_indice(train_shards_path[0], train_shards_path[1], indice_path, -1, 17, 
                                    train=True, resolution=crop_size, resizecrop=False,
                                    normalize_mean=normalize_mean, normalize_std=normalize_std)
                                    
        if dist.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                self._train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
            )
        else:
            sampler = None
        self._train_dataloader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=per_gpu_batch_size,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            sampler=sampler,
            shuffle=False
        )

        # Add meta-data to dataloader instance for convenience.
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

        # Create eval dataset and loader.
        self._eval_dataset = DecordVideoDataset_indice(eval_shards_path[0], eval_shards_path[1], indice_path,-1, 17, 
                                    train=False, resolution=crop_size, resizecrop=False,
                                    normalize_mean=normalize_mean, normalize_std=normalize_std)
        if dist.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                self._eval_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
            )
        else:
            sampler = None
        self._eval_dataloader = torch.utils.data.DataLoader(
            self._eval_dataset,
            batch_size=per_gpu_batch_size,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            sampler=sampler,
            shuffle=False
        )

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def eval_dataset(self):
        return self._eval_dataset

    @property
    def eval_dataloader(self):
        return self._eval_dataloader