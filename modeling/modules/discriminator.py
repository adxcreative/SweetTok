"""This file contains some base implementation for discrminators.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

TODO: Add reference to Mark Weber's tech report on the improved discriminator architecture.
"""
import functools
import math
from typing import Tuple
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

from .maskgit_vqgan import Conv2dSame

class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise=None):
        if x.ndim == 4:
            # image: B C H W
            if noise is None:
                noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)
        
        else:
            # video: B C T H W
            if noise is None:
                noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), x.size(4), device=x.device, dtype=x.dtype)

            return x + self.weight.view(1, -1, 1, 1, 1) * noise.to(x.device)

def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)


class BlurBlock(torch.nn.Module):
    def __init__(self,
                 kernel: Tuple[int] = (1, 3, 3, 1)
                 ):
        super().__init__()

        kernel = torch.tensor(kernel, dtype=torch.float32, requires_grad=False)
        kernel = kernel[None, :] * kernel[:, None]
        kernel /= kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)

    def calc_same_pad(self, i: int, k: int, s: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ic, ih, iw = x.size()[-3:]
        pad_h = self.calc_same_pad(i=ih, k=4, s=2)
        pad_w = self.calc_same_pad(i=iw, k=4, s=2)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        weight = self.kernel.expand(ic, -1, -1, -1)

        out = F.conv2d(input=x, weight=weight, stride=2, groups=x.shape[1])
        return out


class Blur2d(nn.Module):
    def __init__(self, f=[1,2,1], normalize=True, flip=False, stride=1):
        """
            depthwise_conv2d:
            https://blog.csdn.net/mao_xiao_feng/article/details/78003476
        """
        super(Blur2d, self).__init__()
        assert isinstance(f, list) or f is None, "kernel f must be an instance of python built_in type list!"

        if f is not None:
            f = torch.tensor(f, dtype=torch.float32)
            f = f[:, None] * f[None, :]
            f = f[None, None]
            if normalize:
                f = f / f.sum()
            if flip:
                # f = f[:, :, ::-1, ::-1]
                f = torch.flip(f, [2, 3])
            self.f = f
        else:
            self.f = None
        self.stride = stride

    def forward(self, x):
        if self.f is not None:
            # expand kernel channels
            kernel = self.f.expand(x.size(1), -1, -1, -1).to(x.device)
            x = F.conv2d(
                x,
                kernel,
                stride=self.stride,
                padding=int((self.f.size(2)-1)/2),
                groups=x.size(1)
            )
            return x
        else:
            return x

class NLayerDiscriminator(torch.nn.Module):
    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 128,
        num_stages: int = 3,
        blur_resample: bool = True,
        blur_kernel_size: int = 4
    ):
        """ Initializes the NLayerDiscriminator.

        Args:
            num_channels -> int: The number of input channels.
            hidden_channels -> int: The number of hidden channels.
            num_stages -> int: The number of stages.
            blur_resample -> bool: Whether to use blur resampling.
            blur_kernel_size -> int: The blur kernel size.
        """
        super().__init__()
        assert num_stages > 0, "Discriminator cannot have 0 stages"
        assert (not blur_resample) or (blur_kernel_size >= 3 and blur_kernel_size <= 5), "Blur kernel size must be in [3,5] when sampling]"

        in_channel_mult = (1,) + tuple(map(lambda t: 2**t, range(num_stages)))
        init_kernel_size = 5
        activation = functools.partial(torch.nn.LeakyReLU, negative_slope=0.1)

        self.block_in = torch.nn.Sequential(
            Conv2dSame(
                num_channels,
                hidden_channels,
                kernel_size=init_kernel_size
            ),
            activation(),
        )

        BLUR_KERNEL_MAP = {
            3: (1,2,1),
            4: (1,3,3,1),
            5: (1,4,6,4,1),
        }

        discriminator_blocks = []
        for i_level in range(num_stages):
            in_channels = hidden_channels * in_channel_mult[i_level]
            out_channels = hidden_channels * in_channel_mult[i_level + 1]
            block = torch.nn.Sequential(
                Conv2dSame(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                ),
                torch.nn.AvgPool2d(kernel_size=2, stride=2) if not blur_resample else BlurBlock(BLUR_KERNEL_MAP[blur_kernel_size]),
                torch.nn.GroupNorm(32, out_channels),
                activation(),
            )
            discriminator_blocks.append(block)

        self.blocks = torch.nn.ModuleList(discriminator_blocks)

        self.pool = torch.nn.AdaptiveMaxPool2d((16, 16))

        self.to_logits = torch.nn.Sequential(
            Conv2dSame(out_channels, out_channels, 1),
            activation(),
            Conv2dSame(out_channels, 1, kernel_size=5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            x -> torch.Tensor: The input tensor.

        Returns:
            output -> torch.Tensor: The output tensor.
        """
        hidden_states = self.block_in(x)
        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.pool(hidden_states)

        return self.to_logits(hidden_states)

# class NLayerDiscriminator(nn.Module):
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_type="batch", use_sigmoid=False, getIntermFeat=True, activation="leaky_relu", apply_blur=False, apply_noise=False):
#     # def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
#         super(NLayerDiscriminator, self).__init__()
#         self.getIntermFeat = getIntermFeat
#         self.n_layers = n_layers

#         if apply_noise:
#             self.noise = ApplyNoise(channels=input_nc)
#         else:
#             self.noise = nn.Identity()
        

#         activation_func = nn.LeakyReLU(0.2, True) if activation == "leaky_relu" else nn.Tanh() 
        
#         kw = 4
#         padw = int(np.ceil((kw-1.0)/2))
#         sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), activation_func]]

#         nf = ndf
#         for n in range(1, n_layers):
#             nf_prev = nf
#             nf = min(nf * 2, 512)
#             sequence += [[
#                 Blur2d(f=None) if apply_blur else nn.Identity(),
#                 nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
#                 Normalize(nf, norm_type), 
#                 activation_func
#             ]]

#         nf_prev = nf
#         nf = min(nf * 2, 512)
#         sequence += [[
#             nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
#             Normalize(nf, norm_type),
#             activation_func
#         ]]

#         sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

#         if use_sigmoid:
#             sequence += [[nn.Sigmoid()]]

#         if getIntermFeat:
#             for n in range(len(sequence)):
#                 setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
#         else:
#             sequence_stream = []
#             for n in range(len(sequence)):
#                 sequence_stream += sequence[n]
#             self.model = nn.Sequential(*sequence_stream)

#     def forward(self, input, apply_diffaug=False):
#         if self.getIntermFeat:
#             trans_input = self.noise(input)
#             if apply_diffaug:
#                 trans_input = DiffAugment(trans_input, policy='color,translation,cutout')
            
#             res = [trans_input]
#             for n in range(self.n_layers+2):
#                 model = getattr(self, 'model'+str(n))
#                 res.append(model(res[-1]))
#             return res[-1], res[1:]
#         else:
#             return self.model(input), _





class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type="batch", use_sigmoid=False, getIntermFeat=True, activation="leaky_relu", apply_blur=False, apply_noise=False):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        if apply_noise:
            self.noise = ApplyNoise(channels=input_nc)
        else:
            self.noise = nn.Identity()

        activation_func = nn.LeakyReLU(0.2, True) if activation == "leaky_relu" else nn.Tanh() 

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), activation_func]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                Blur2d(f=None) if apply_blur else nn.Identity(),
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                Normalize(nf, norm_type), 
                activation_func
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            Normalize(nf, norm_type),
            activation_func
        ]]

        sequence += [[
            nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw),
            Normalize(1, norm_type),
            activation_func
        ]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input, apply_diffaug=False):
        if self.getIntermFeat:
            trans_input = self.noise(input)
            if apply_diffaug:
                B = input.shape[0]
                trans_input = rearrange(trans_input, "b c t h w -> (b t) c h w") # B C T H W
                trans_input = DiffAugment(trans_input, policy='color,translation,cutout')
                trans_input = rearrange(trans_input, "(b t) c h w -> b c t h w", b=B)
            
            res = [trans_input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
                    
            return res[-1], res[1:]
        else:
            return self.model(input), _