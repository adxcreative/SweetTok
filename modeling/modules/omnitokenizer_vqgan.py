import math
import argparse
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_
from .omnitokenizer_attention_modules import Transformer

from enum import unique
import torch.distributed as dist

def divisible_by(numer, denom):
    return (numer % denom) == 0

def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret

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

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)
class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim, no_random_restart=False, restart_thres=1.0, usage_sigma=0.99, fp32_quant=False):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())
        self.register_buffer('codebook_usage', torch.zeros(n_codes))

        self.call_cnt = 0
        self.usage_sigma = usage_sigma

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres

        self.fp32_quant = fp32_quant

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))
    

    def calculate_batch_codebook_usage_percentage(self, batch_encoding_indices):
        # Flatten the batch of encoding indices into a single 1D tensor
        all_indices = batch_encoding_indices.flatten()
        
        # Obtain the total number of encoding indices in the batch to calculate percentages
        total_indices = all_indices.numel()
        
        # Initialize a tensor to store the percentage usage of each code
        codebook_usage_percentage = torch.zeros(self.n_codes, device=all_indices.device)
        
        # Count the number of occurrences of each index and get their frequency as percentages
        unique_indices, counts = torch.unique(all_indices, return_counts=True)
        # Calculate the percentage
        percentages = (counts.float() / total_indices)
        
        # Populate the corresponding percentages in the codebook_usage_percentage tensor
        codebook_usage_percentage[unique_indices.long()] = percentages
        
        return codebook_usage_percentage
    


    def forward(self, z):
        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2) # [bthw, c]
        
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * flat_inputs @ self.embeddings.t() \
                    + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True) # [bthw, c]

        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs) # [bthw, ncode]
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:]) # [b, t, h, w, ncode]

        embeddings = F.embedding(encoding_indices, self.embeddings) # [b, t, h, w, c]
        embeddings = shift_dim(embeddings, -1, 1) # [b, c, t, h, w]

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            if not self.no_random_restart:
                usage = (self.N.view(self.n_codes, 1) >= self.restart_thres).float()
                self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        try:
            usage = self.calculate_batch_codebook_usage_percentage(encoding_indices)
        except:
            usage = torch.zeros(self.n_codes, device=encoding_indices.device)
        

        # print(usage.shape, torch.zeros(self.n_codes).shape)

        if self.call_cnt == 0:
            self.codebook_usage.data = usage
        else:
            self.codebook_usage.data = self.usage_sigma * self.codebook_usage.data + (1 - self.usage_sigma) * usage

        self.call_cnt += 1
        # avg_distribution = self.codebook_usage.data.sum() / self.n_codes
        avg_usage = (self.codebook_usage.data > (1/self.n_codes)).sum() / self.n_codes
            
        return dict(embeddings=embeddings_st, encodings=encoding_indices,
                    commitment_loss=commitment_loss, perplexity=perplexity, avg_usage=avg_usage, batch_usage=usage)

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings


class OmniTokenizer_Encoder(nn.Module):
    def __init__(self, image_size, patch_embed, norm_type, block='tttt', window_size=4, spatial_pos="rel",
                    image_channel=3, patch_size=16, temporal_patch_size=2, defer_temporal_pool=False, defer_spatial_pool=False,
                    spatial_depth=4, temporal_depth=4, causal_in_temporal_transformer=False, dim=512, 
                    causal_in_peg=True, dim_head=64, heads=8, attn_dropout=0., ff_dropout=0., ff_mult=4., initialize=False, sequence_length=17):
        super().__init__()
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size
        self.temporal_patch_size = temporal_patch_size
        self.block = block

        # self.spatial_rel_pos_bias = ContinuousPositionBias(
        #     dim=dim, heads=heads)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (
            image_width % patch_width) == 0

        if patch_embed == 'linear':
            if defer_temporal_pool:
                temporal_patch_size //= 2
                self.temporal_patch_size = temporal_patch_size
                self.temporal_pool = nn.AvgPool3d(kernel_size=(2, 1, 1))
            else:
                self.temporal_pool = nn.Identity()
            
            if defer_spatial_pool:
                self.patch_size =  pair(patch_size // 2)
                patch_height, patch_width = self.patch_size
                self.spatial_pool = nn.AvgPool3d(kernel_size=(1, 2, 2))
            else:
                self.spatial_pool = nn.Identity()

            self.to_patch_emb_first_frame = nn.Sequential(
                Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)',
                        p1=patch_height, p2=patch_width),
                nn.LayerNorm(image_channel * patch_width * patch_height),
                nn.Linear(image_channel * patch_width * patch_height, dim),
                nn.LayerNorm(dim)
            )

            self.to_patch_emb = nn.Sequential(
                Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)',
                        p1=patch_height, p2=patch_width, pt=temporal_patch_size),
                nn.LayerNorm(image_channel * patch_width *
                            patch_height * temporal_patch_size),
                nn.Linear(image_channel * patch_width *
                        patch_height * temporal_patch_size, dim),
                nn.LayerNorm(dim)
            )
        elif patch_embed == 'cnn':
            self.to_patch_emb_first_frame = nn.Sequential(
                # SamePadConv3d(image_channel, dim, kernel_size=(1, patch_height, patch_width), stride=(1, patch_height, patch_width)),
                nn.Conv3d(image_channel, dim, kernel_size=(1, patch_height, patch_width), stride=(1, patch_height, patch_width)),
                Normalize(dim, norm_type),
                Rearrange('b c t h w -> b t h w c'),
            )

            self.to_patch_emb = nn.Sequential(
                # SamePadConv3d(image_channel, dim, kernel_size=(temporal_patch_size, patch_height, patch_width), stride=(temporal_patch_size, patch_height, patch_width)),
                nn.Conv3d(image_channel, dim, kernel_size=(temporal_patch_size, patch_height, patch_width), stride=(temporal_patch_size, patch_height, patch_width)),
                Normalize(dim, norm_type),
                Rearrange('b c t h w -> b t h w c'),
            )

            self.temporal_pool, self.spatial_pool = nn.Identity(), nn.Identity()

        else:
            raise NotImplementedError

        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=causal_in_peg,
            ff_mult=ff_mult
        )

        self.enc_spatial_transformer = Transformer(depth=spatial_depth, block=block, window_size=window_size, spatial_pos=spatial_pos, **transformer_kwargs)

        
        if causal_in_temporal_transformer:
            transformer_kwargs["causal"] = True

        self.enc_temporal_transformer = Transformer(
            depth=temporal_depth, block='t' * temporal_depth, **transformer_kwargs)
        
        if initialize:
            self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]
    

    def encode(
        self,
        tokens, 
    ):
        b = tokens.shape[0]  # batch size
        # h, w = self.patch_height_width  # patch h,w
        is_image = tokens.shape[1] == 1

        # video shape, last dimension is the embedding size
        video_shape = tuple(tokens.shape[:-1])
        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        
        # encode - spatial
        tokens = self.enc_spatial_transformer(tokens, video_shape=video_shape, is_spatial=True)

        hw = tokens.shape[1]
        new_h, new_w = int(math.sqrt(hw)), int(math.sqrt(hw))
        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, h=new_h, w=new_w)

        # encode - temporal
        video_shape2 = tuple(tokens.shape[:-1])
        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')
        tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape2, is_spatial=False)
        # tokens = self.enc_temporal_transformer(tokens)

        # codebook expects:  [b, c, t, h, w]
        tokens = rearrange(tokens, '(b h w) t d -> b d t h w', b=b, h=new_h, w=new_w)
        tokens = self.spatial_pool(tokens)

        if tokens.shape[2] > 1:
            first_frame_tokens = tokens[:, :, 0:1]
            rest_frames_tokens = tokens[:, :, 1:]
            rest_frames_tokens = self.temporal_pool(rest_frames_tokens)
            tokens = torch.cat([first_frame_tokens, rest_frames_tokens], dim=2)

        return tokens

    
    def forward(self, video, is_image, mask=None):
        # 4 is BxCxHxW (for images), 5 is BxCxFxHxW
        assert video.ndim in {4, 5}

        if is_image:  # add temporal channel to 1 for images only
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert mask is None

        _, _, f, *image_dims = *video.shape, 

        # assert tuple(image_dims) == self.image_size
        assert mask is None or mask.shape[-1] == f
        assert divisible_by(
            f - 1, self.temporal_patch_size), f'number of frames ({f}) minus one ({f - 1}) must be divisible by temporal patch size ({self.temporal_patch_size})'

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]

        # derive patches
        first_frame_tokens = self.to_patch_emb_first_frame(first_frame)

        if rest_frames.shape[2] != 0:
            rest_frames_tokens = self.to_patch_emb(rest_frames)
            # simple cat
            tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim=1)

        else:
            tokens = first_frame_tokens

        return self.encode(tokens)


class OmniTokenizer_Decoder(nn.Module):
    def __init__(self, image_size, patch_embed, norm_type, block='tttt', window_size=4, spatial_pos="rel",
                    image_channel=3, patch_size=16, temporal_patch_size=2, defer_temporal_pool=False, defer_spatial_pool=False,
                    spatial_depth=4, temporal_depth=4, causal_in_temporal_transformer=False, dim=512, 
                    causal_in_peg=True, dim_head=64, heads=8, attn_dropout=0., ff_dropout=0., ff_mult=4., gen_upscale=None, initialize=False,
                    sequence_length=17):
        super().__init__()
        self.gen_upscale = gen_upscale
        if gen_upscale is not None:
            patch_size *= gen_upscale


        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size
        self.block = block

        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=causal_in_peg,
            ff_mult=ff_mult
        )

        #self.spatial_rel_pos_bias = ContinuousPositionBias(
        #    dim=dim, heads=heads) # HACK this: whether shared pos encoding is better or on the contrary

        self.dec_spatial_transformer = Transformer(
            depth=spatial_depth, block=block, window_size=window_size, spatial_pos=spatial_pos, **transformer_kwargs)
        
        if causal_in_temporal_transformer:
            transformer_kwargs["causal"] = True

        self.dec_temporal_transformer = Transformer(
            depth=temporal_depth, block='t' * temporal_depth, **transformer_kwargs)

        if patch_embed == "linear":
            if defer_temporal_pool:
                temporal_patch_size //= 2
                self.temporal_patch_size = temporal_patch_size
                self.temporal_up = nn.Upsample(scale_factor=(2, 1, 1), mode="nearest") # AvgPool3d(kernel_size=(2, 1, 1))
            else:
                self.temporal_up = nn.Identity()
            
            if defer_spatial_pool:
                self.patch_size =  pair(patch_size // 2)
                patch_height, patch_width = self.patch_size
                self.spatial_up = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest") # nn.AvgPool3d(kernel_size=(1, 2, 2))
            else:
                self.spatial_up = nn.Identity()
            
            # b 1 nhnw dim -> b 1 phpw 3phpw
            self.to_pixels_first_frame = nn.Sequential(
                nn.Linear(dim, image_channel * patch_width * patch_height),
                Rearrange('b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)',
                        p1=patch_height, p2=patch_width)
            )

            self.to_pixels = nn.Sequential(
                nn.Linear(dim, image_channel * patch_width *
                        patch_height * temporal_patch_size),
                Rearrange('b t h w (c pt p1 p2) -> b c (t pt) (h p1) (w p2)',
                        p1=patch_height, p2=patch_width, pt=temporal_patch_size),
            )

        elif patch_embed == "cnn":
            # torch.Size([1, 1, 8, 8, 512])
            self.to_pixels_first_frame = nn.Sequential(
                Rearrange('b 1 h w dim -> b dim 1 h w', h=image_size//patch_size),
                # SamePadConvTranspose3d(dim, image_channel, kernel_size=(1, patch_height, patch_width), stride=(1, patch_height, patch_width)),
                nn.ConvTranspose3d(dim, image_channel, kernel_size=(1, patch_height, patch_width), stride=(1, patch_height, patch_width)),
                Normalize(image_channel, norm_type)
            )

            self.to_pixels = nn.Sequential(
                Rearrange('b t h w dim -> b dim t h w', h=image_size//patch_size),
                # SamePadConvTranspose3d(dim, image_channel, kernel_size=(temporal_patch_size, patch_height, patch_width), stride=(temporal_patch_size, patch_height, patch_width)),
                nn.ConvTranspose3d(dim, image_channel, kernel_size=(temporal_patch_size, patch_height, patch_width), stride=(temporal_patch_size, patch_height, patch_width)),
                Normalize(image_channel, norm_type)
            )
            self.temporal_up = nn.Identity()
            self.spatial_up = nn.Identity()
        
        else:
            raise NotImplementedError
    
        if initialize:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @property
    def patch_height_width(self):
        if self.gen_upscale is None:
            return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]
        else:
            return int(self.image_size[0] // self.patch_size[0] * self.gen_upscale), int(self.image_size[1] // self.patch_size[1] * self.gen_upscale)

    def decode(
        self,
        tokens,
    ):
        b = tokens.shape[0]
        # h, w = self.patch_height_width
        is_image = tokens.shape[1] == 1
        video_shape = tuple(tokens.shape[:-1]) # b t h' w' d
        h = tokens.shape[2]
        w = tokens.shape[3]


        # decode - temporal
        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')
        tokens = self.dec_temporal_transformer(tokens, video_shape=video_shape, is_spatial=False)
        # tokens = self.dec_temporal_transformer(tokens)

        # might spatial downsampling here
        down_op = self.block.count('n') + self.block.count('r')
        down_ratio = int(2 ** down_op)

        # decode - spatial
        tokens = rearrange(tokens, '(b h w) t d -> (b t) (h w) d', b=b, h=h//down_ratio, w=w//down_ratio)
        #tokens = self.dec_spatial_transformer(
        #    tokens, attn_bias_func=self.spatial_rel_pos_bias, video_shape=video_shape)
        tokens = self.dec_spatial_transformer(tokens, video_shape=video_shape, is_spatial=True)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)

        # to pixels
        first_frame_token, rest_frames_tokens = tokens[:, :1], tokens[:, 1:]
        first_frame = self.to_pixels_first_frame(first_frame_token)

        if rest_frames_tokens.shape[1] != 0:
            rest_frames = self.to_pixels(rest_frames_tokens)
            recon_video = torch.cat((first_frame, rest_frames), dim=2)
        else:
            recon_video = first_frame
        
        return recon_video
    

    def forward(self, tokens, is_image, mask=None):
        # expected input: b d t h w -> b t h w d
        if tokens.shape[2] > 1:
            first_frame_tokens = tokens[:, :, 0:1]
            rest_frames_tokens = tokens[:, :, 1:]
            rest_frames_tokens = self.temporal_up(rest_frames_tokens)
            tokens = torch.cat([first_frame_tokens, rest_frames_tokens], dim=2)

        tokens = self.spatial_up(tokens)
        tokens = tokens.permute(0, 2, 3, 4, 1).contiguous()

        recon_video = self.decode(tokens)

        # handle shape if we are training on images only
        returned_recon = rearrange(
            recon_video, 'b c 1 h w -> b c h w') if is_image else recon_video.clone()

        return returned_recon


