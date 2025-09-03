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


class SweetTok_base_Encoder(nn.Module):
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


class SweetTok_base_Decoder(nn.Module):
    def __init__(self, image_size, patch_embed, norm_type, block='tttt', window_size=4, spatial_pos="rel",
                    image_channel=3, patch_size=16, temporal_patch_size=2, defer_temporal_pool=False, defer_spatial_pool=False,
                    spatial_depth=4, temporal_depth=4, causal_in_temporal_transformer=False, dim=512, 
                    causal_in_peg=True, dim_head=64, heads=8, attn_dropout=0., ff_dropout=0., ff_mult=4., gen_upscale=None, initialize=False,
                    sequence_length=17):
        super().__init__()
        self.gen_upscale = gen_upscale
        if gen_upscale is not None:
            patch_size *= gen_upscale

        self.dim = dim
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

        self.ln_post = nn.LayerNorm(self.dim)

        self.ffn = nn.Sequential(
            nn.Conv2d(self.dim, 2 * self.dim, 1, padding=0, bias=True),
            nn.Tanh(),
            nn.Conv2d(2 * self.dim, 8192, 1, padding=0, bias=True),
        )
        self.conv_out = nn.Identity()

        self.temporal_up = nn.Identity()
        self.spatial_up = nn.Identity()

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

        tokens = rearrange(tokens, '(b t) (h w) d -> b (t h w) d', b=b, h=h, w=w)

        x = self.ln_post(tokens)
        # N L D -> N D H W
        # x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        x = x.permute(0, 2, 1).reshape(b, self.dim, 5, 32**2)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x) # [B, D, T, H*W]
        x = x.reshape(b, x.shape[1], 5, 32, 32).contiguous()
        
        return x
    

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
        # returned_recon = rearrange(
        #     recon_video, 'b c 1 h w -> b c h w') if is_image else recon_video.clone()

        return recon_video



class SweetTok_Compact_Encoder(nn.Module):
    def __init__(self, config,image_size, patch_embed, norm_type, block='tttt', window_size=4, spatial_pos="rel",
                    image_channel=3, patch_size=16, temporal_patch_size=2, defer_temporal_pool=False, defer_spatial_pool=False,
                    spatial_depth=4, temporal_depth=4, causal_in_temporal_transformer=False, dim=512, 
                    causal_in_peg=True, dim_head=64, heads=8, attn_dropout=0., ff_dropout=0., ff_mult=4., initialize=False, sequence_length=17,
                    use_temporal_transformer=False):
        super().__init__()
        self.config = config
        self.dim = dim 
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size
        self.temporal_patch_size = temporal_patch_size
        self.block = block
        self.num_spatial_latent_tokens = config.model.vq_model.num_spatial_latent_tokens
        self.num_temporal_latent_tokens = config.model.vq_model.num_temporal_latent_tokens
        self.num_intermediate_temporal_tokens = config.model.vq_model.num_intermediate_temporal_tokens
        self.token_size = config.model.vq_model.token_size
        self.use_temporal_transformer = use_temporal_transformer

        # positional embedding for latent tokens
        scale = self.dim ** -0.5
        self.spatial_latent_tokens_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_spatial_latent_tokens, self.dim))
        if self.use_temporal_transformer:
            self.temporal_latent_tokens_positional_embedding = nn.Parameter(
                scale * torch.randn(self.num_temporal_latent_tokens, self.dim)) # [32*32]


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
            ff_mult=ff_mult,
        )

        self.enc_spatial_transformer = Transformer(depth=spatial_depth, block=block, window_size=window_size, spatial_pos=spatial_pos, **transformer_kwargs)

        latent_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=False,
            peg_causal=causal_in_peg,
            ff_mult=ff_mult,
            has_cross_attn=True,
            attn_num_null_kv=0,
        )
        self.enc_latent_spatial_transformer = Transformer(depth=spatial_depth, block=block, window_size=window_size, spatial_pos=spatial_pos, **latent_kwargs)
        
        if causal_in_temporal_transformer:
            transformer_kwargs["causal"] = True

        if self.use_temporal_transformer:

            temporal_latent_kwargs = dict(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                peg=False,
                peg_causal=causal_in_peg,
                ff_mult=ff_mult,
                has_cross_attn=True,
                attn_num_null_kv=0,
            )
            self.enc_temporal_transformer = Transformer(
                depth=temporal_depth, block='t' * temporal_depth, spatial_pos=None, **transformer_kwargs)
            
            self.enc_latent_temporal_transformer = Transformer(depth = temporal_depth, block= 't'* temporal_depth, spatial_pos=None, **temporal_latent_kwargs)


        self.ln_post_spatial = nn.LayerNorm(self.dim)
        if self.use_temporal_transformer:
            self.ln_post_temporal = nn.LayerNorm(self.dim)
        self.conv_out = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=True)
        self.conv_out_temporal = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=True)

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
        spatial_latent_tokens,
        temporal_latent_tokens
    ):
        # tokens = tokens[:, 0:1, ...]
        h = tokens.shape[2]
        w = tokens.shape[3]
        b, t, hw = tokens.shape[0], tokens.shape[1], tokens.shape[2]*tokens.shape[3]  # batch size
        # h, w = self.patch_height_width  # patch h,w
        is_image = tokens.shape[1] == 1

        # latent tokens: spatial 256, temporal: 1024
        spatial_latent_tokens = _expand_token(spatial_latent_tokens, b).to(tokens.dtype)
        spatial_latent_tokens = spatial_latent_tokens + self.spatial_latent_tokens_positional_embedding.to(tokens.dtype)


        temporal_latent_tokens = _expand_token(temporal_latent_tokens, b).to(tokens.dtype)
        temporal_latent_tokens = temporal_latent_tokens + self.temporal_latent_tokens_positional_embedding.to(tokens.dtype)

        # video shape, last dimension is the embedding size
        first_frame_tokens = tokens
        video_shape = tuple(first_frame_tokens.shape[:-1])
        # encode - spatial
        sp_tokens = self.enc_spatial_transformer(rearrange(first_frame_tokens, 'b t h w d -> (b t) (h w) d'), video_shape=video_shape, is_spatial=True, return_inter = True)
        pooled_sp_tokens = []
        tshape_sp_tokens = []
        for sp_token in sp_tokens:
            sp_token = rearrange(sp_token, '(b t) l d -> b t l d', t=t)
            pooled_sp_tokens.append(sp_token.mean(1))
            tshape_sp_tokens.append(sp_token)
        spatial_latent_tokens = self.enc_latent_spatial_transformer(spatial_latent_tokens, context = pooled_sp_tokens, is_spatial = True, multi_cross = True)


        # # encode - temporal
        token_diffs = tokens[:, 1:, :, :, :] - tokens[:, :-1, :, :, :]        
        video_shape2 = tuple(token_diffs.shape[:-1])
        token_diffs = rearrange(token_diffs, 'b t h w d -> (b h w) t d')
        te_tokens = self.enc_temporal_transformer(token_diffs, video_shape=video_shape2, is_spatial=False, return_inter = True)


        l = temporal_latent_tokens.shape[1]
        temporal_latent_tokens = rearrange(temporal_latent_tokens, 'b (n t) d -> (b n) t d', t=1)
        temporal_latent_tokens = self.enc_latent_temporal_transformer(temporal_latent_tokens, context = te_tokens, is_spatial = False, multi_cross = True)
        temporal_latent_tokens = rearrange(temporal_latent_tokens, '(b l) t d -> b (t l) d', l=1024)


        #latent token: b * (256 + 1024) * d
        spatial_latent_tokens = self.ln_post_spatial(spatial_latent_tokens)
        spatial_latent_tokens = rearrange(spatial_latent_tokens, 'b l d -> b d l 1')
        spatial_latent_tokens = self.conv_out(spatial_latent_tokens)
        spatial_latent_tokens = rearrange(spatial_latent_tokens,'b d l 1 -> b d 1 l')

        temporal_latent_tokens = self.ln_post_temporal(temporal_latent_tokens)
        temporal_latent_tokens = rearrange(temporal_latent_tokens, 'b l d -> b d l 1')
        temporal_latent_tokens = self.conv_out_temporal(temporal_latent_tokens)
        temporal_latent_tokens = rearrange(temporal_latent_tokens,'b d l 1 -> b d 1 l')

        return tokens, spatial_latent_tokens, temporal_latent_tokens

    
    def forward(self, video, is_image, spatial_latent_tokens, temporal_latent_tokens, mask=None):
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
        return self.encode(tokens, spatial_latent_tokens, temporal_latent_tokens)


class SweetTok_Compact_Decoder(nn.Module):
    def __init__(self, config ,image_size, patch_embed, norm_type, block='tttt', window_size=4, spatial_pos="rel",
                    image_channel=3, patch_size=16, temporal_patch_size=2, defer_temporal_pool=False, defer_spatial_pool=False,
                    spatial_depth=4, temporal_depth=4, causal_in_temporal_transformer=False, dim=512, 
                    causal_in_peg=True, dim_head=64, heads=8, attn_dropout=0., ff_dropout=0., ff_mult=4., gen_upscale=None, initialize=False,
                    sequence_length=17, use_temporal_transformer=False):
        super().__init__()

        self.config= config
        self.gen_upscale = gen_upscale
        if gen_upscale is not None:
            patch_size *= gen_upscale

        self.dim = dim
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size
        self.grid_size = 32
        self.num_spatial_latent_tokens = config.model.vq_model.num_spatial_latent_tokens
        self.num_temporal_latent_tokens = config.model.vq_model.num_temporal_latent_tokens
        self.num_intermediate_temporal_tokens = config.model.vq_model.num_intermediate_temporal_tokens
        self.token_size = config.model.vq_model.token_size
        self.tube_size = config.model.vq_model.dec_tube_size
        self.output_size = config.model.vq_model.dec_output_size
        self.batch_size = config.training.per_gpu_batch_size
        self.block = block
        self.use_temporal_transformer = use_temporal_transformer

        scale = self.dim ** -0.5
        self.spatial_latent_tokens_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_spatial_latent_tokens, self.dim))
        
        if self.use_temporal_transformer:
            self.temporal_latent_tokens_positional_embedding = nn.Parameter(
                scale * torch.randn(self.num_temporal_latent_tokens, self.dim))

        self.positional_embedding = nn.Parameter(
                scale * torch.randn(32 ** 2, self.dim))
        
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.dim)) 
        
        self.decoder_embed = nn.Linear(
            self.dim, self.dim, bias=True)
        self.decoder_embed_temporal = nn.Linear(
            self.dim, self.dim, bias=True
        )

        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=causal_in_peg,
            ff_mult=ff_mult,
            has_cross_attn=True
        )

        #self.spatial_rel_pos_bias = ContinuousPositionBias(
        #    dim=dim, heads=heads) # HACK this: whether shared pos encoding is better or on the contrary

        self.dec_spatial_transformer = Transformer(
            depth=spatial_depth, block=block, window_size=window_size, spatial_pos=spatial_pos, **transformer_kwargs)

        latent_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=False,
            peg_causal=False,
            ff_mult=ff_mult,
            attn_num_null_kv = 0,
        )

        self.dec_spatial_latent_transformer = Transformer(
            depth=spatial_depth, block=block, window_size=window_size, spatial_pos=spatial_pos, **latent_kwargs)

        if causal_in_temporal_transformer:
            transformer_kwargs["causal"] = True

        if self.use_temporal_transformer:

            temporal_latent_kwargs = dict(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                peg=False,
                peg_causal=causal_in_peg,
                ff_mult=ff_mult,
                has_cross_attn=False,
                attn_num_null_kv=0,
            )

            self.dec_temporal_transformer = nn.ModuleList([])
            for _ in range(temporal_depth):
                self.dec_temporal_transformer.append(Transformer(
                    depth=1, block='c', spatial_pos= None, **transformer_kwargs))

            self.dec_temporal_latent_transformer = Transformer(
                depth=temporal_depth, block='t' * temporal_depth, spatial_pos= None, **temporal_latent_kwargs)

        self.ln_post = nn.LayerNorm(self.dim)

        self.ffn = nn.Sequential(
            nn.Conv2d(self.dim, 2 * self.dim, 1, padding=0, bias=True),
            nn.Tanh(),
            nn.Conv2d(2 * self.dim, self.output_size, 1, padding=0, bias=True),
        )
        self.conv_out = nn.Identity()

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
        
        # else:
        #     raise NotImplementedError
        #     
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
        z_quantized,
    ):
        b, d, h, w = z_quantized[0].shape
        _, d_ ,_ ,w_ = z_quantized[1].shape

        hw = self.grid_size**2
        
        if self.use_temporal_transformer:
            assert h == 1 and w + w_ == (self.num_spatial_latent_tokens + self.num_temporal_latent_tokens), f"{h}, {w}, {(self.num_spatial_latent_tokens)}"
        else:
            assert h == 1 and w == (self.num_spatial_latent_tokens), f"{h}, {w}, {(self.num_spatial_latent_tokens)}"
        z_quantized_spatial = z_quantized[0].reshape(b, d*h, w).permute(0, 2, 1) # NLD
        z_quantized_temporal = z_quantized[1].reshape(b, d_*h, w_).permute(0, 2, 1)

        spatial_latent_tokens = self.decoder_embed(z_quantized_spatial)
        temporal_latent_tokens = self.decoder_embed_temporal(z_quantized_temporal)
        
        h,w = self.grid_size, self.grid_size
        spatial_latent_tokens = spatial_latent_tokens + self.spatial_latent_tokens_positional_embedding.to(z_quantized_spatial.dtype)
        temporal_latent_tokens = temporal_latent_tokens + self.temporal_latent_tokens_positional_embedding.to(z_quantized_spatial.dtype)

        mask_tokens = self.mask_token.repeat(b, self.grid_size**2, 1).to(z_quantized_spatial.dtype)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        mask_tokens = rearrange(mask_tokens,'b (t h w) d -> b t h w d', t=1, h = self.grid_size, w = self.grid_size)

        mask_tokens = mask_tokens.repeat(1, self.tube_size, 1, 1, 1)
        
        video_shape = tuple(mask_tokens.shape[:-1])

        # # might spatial downsampling here

        down_ratio = 1

        # decode - temporal
        temporal_latent_tokens = rearrange(temporal_latent_tokens, 'b (n t) d -> (b n) t d', t=1)
        tp_latent_tokens = self.dec_temporal_latent_transformer(temporal_latent_tokens, is_spatial=False, return_inter=True)

        # decode - spatial
        spatial_latent_tokens = spatial_latent_tokens.unsqueeze(1).repeat(1, self.tube_size, 1, 1)
        spatial_latent_tokens = rearrange(spatial_latent_tokens, 'b t l d -> (b t) l d')
        sp_latent_tokens = self.dec_spatial_latent_transformer(spatial_latent_tokens, is_spatial=True, return_inter = True)
        mask_tokens = rearrange(mask_tokens, ' b t h w d -> (b t) (h w) d', b=b, h=h//down_ratio, w=w//down_ratio)
        mask_tokens = self.dec_spatial_transformer(mask_tokens, context = sp_latent_tokens, video_shape = video_shape, 
            is_spatial=True, multi_cross=True, external_temporal_layer=self.dec_temporal_transformer, 
            external_temporal_inputs=tp_latent_tokens, external_temporal_layer_nums=[1, 3, 5, 7])
        mask_tokens = rearrange(mask_tokens, '(b t) (h w) d -> b (t h w) d', b=b, t=self.tube_size, h=h, w=w)

        x = self.ln_post(mask_tokens)
        # N L D -> N D H W

        x = x.permute(0, 2, 1).reshape(b, self.dim, self.tube_size, self.grid_size**2)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x) # [B, D, T, H*W]
        x = x.reshape(b, x.shape[1], self.tube_size, self.grid_size, self.grid_size).contiguous()
        return x
    

    def forward(self, tokens, is_image, mask=None):
        recon_video = self.decode(tokens)

        return recon_video



