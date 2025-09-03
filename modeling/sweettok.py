import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from modeling.modules.base_model import BaseModel
from modeling.modules.blocks import TiTokEncoder, TiTokDecoder
from modeling.quantizer.quantizer import VectorQuantizer, MLC_quantizer_noun, MLC_quantizer_verb
from modeling.modules.maskgit_vqgan import Encoder as Pixel_Eecoder
from modeling.modules.maskgit_vqgan import Decoder as Pixel_Decoder
from modeling.modules.maskgit_vqgan import VectorQuantizer as Pixel_Quantizer
from modeling.modules.omnitokenizer_vqgan import OmniTokenizer_Encoder, \
        OmniTokenizer_Decoder, Codebook
from modeling.modules.sweettok_modules import SweetTok_base_Encoder, SweetTok_base_Decoder, SweetTok_Compact_Encoder, SweetTok_Compact_Decoder
import json
from omegaconf import OmegaConf
from pathlib import Path

from huggingface_hub import PyTorchModelHubMixin


class PretrainedTokenizer(nn.Module):
    def __init__(self, pretrained_weight):
        super().__init__()
        conf = OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
            "num_resolutions": 5,
            "dropout": 0.0,
            "hidden_channels": 128,
            "num_channels": 3,
            "num_res_blocks": 2,
            "resolution": 256,
            "z_channels": 256})
        self.encoder = Pixel_Eecoder(conf)
        self.decoder = Pixel_Decoder(conf)
        self.quantize = Pixel_Quantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
        # Load pretrained weights
        self.load_state_dict(torch.load(pretrained_weight, map_location=torch.device("cpu")), strict=True)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode(self, x):
        is_image = x.ndim == 4
        if not is_image:
            B = x.shape[0]
            T = x.shape[2]
            x = rearrange(x, 'B D T H W -> (B T) D H W')
        hidden_states = self.encoder(x)
        quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states)
        if not is_image:
            codebook_indices = rearrange(codebook_indices, '(B T) S -> B T S', T = T)
        return codebook_indices.detach()
    
    @torch.no_grad()
    def decode(self, codes):
        is_image = codes.shape == 3
        if not is_image:
            B = codes.shape[0]
            T = codes.shape[1]
            codes = rearrange(codes, 'B T H W -> (B T) H W')
        quantized_states = self.quantize.get_codebook_entry(codes)
        rec_images = self.decoder(quantized_states)
        if not is_image:
            rec_images = rearrange(rec_images, '(B T) D H W -> B D T H W', B = B)
        rec_images = torch.clamp(rec_images, 0.0, 1.0)
        return rec_images.detach()

class PretrainedTokenizer_Omni(nn.Module):
    def __init__(self, pretrained_weight):
        super().__init__()

        self.encoder = OmniTokenizer_Encoder(
            image_size = 256, image_channel=3, norm_type= 'batch', 
            block= 'ttww' , window_size=8, spatial_pos= 'rope',
            patch_embed = 'linear', patch_size = 8, temporal_patch_size= 4, defer_temporal_pool= False, defer_spatial_pool= False,
            spatial_depth=4, temporal_depth=4, causal_in_temporal_transformer=True, causal_in_peg=True, 
            dim = 512, dim_head=64, heads=8, attn_dropout=0.0, ff_dropout=0.0, ff_mult=4.0,
            initialize=False, sequence_length=17,
        )
        
        self.decoder = OmniTokenizer_Decoder(
            image_size = 256, image_channel=3, norm_type='batch', 
            block='tttt', window_size=8, spatial_pos= 'rope',
            patch_embed = 'linear', patch_size = 8, temporal_patch_size= 4, defer_temporal_pool= False, defer_spatial_pool=False,
            spatial_depth=4, temporal_depth=4, causal_in_temporal_transformer=True, causal_in_peg=True, 
            dim = 512, dim_head=64, heads=8, attn_dropout=0.0, ff_dropout=0.0, ff_mult=4.0, 
            gen_upscale=None, initialize=False, sequence_length=17,
        )
        self.l2_code = True
        self.resolution = 256
        self.patch_size = 8

        self.codebook = Codebook(8192, 8, no_random_restart=True, restart_thres=1.0)
        self.pre_vq_conv = nn.Sequential(
                    Rearrange("b c t h w -> b t h w c"),
                    nn.Linear(512, 8),
                    Rearrange("b t h w c -> b c t h w")
                )
        self.post_vq_conv = nn.Sequential(
                    Rearrange("b c t h w -> b t h w c"),
                    nn.Linear(8, 512),
                    Rearrange("b t h w c -> b c t h w")
                )
        # Load pretrained weights
        state_dict = torch.load(pretrained_weight, map_location=torch.device("cpu"))['state_dict']
        w_keys = list(state_dict.keys())
        for k in w_keys:
            if 'image_discriminator' in k or \
                'video_discriminator' in k or \
                'perceptual_model' in k or \
                    'context_norm' in k :
                state_dict.pop(k, None)

        self.load_state_dict(state_dict, strict=True)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode(self, x, is_image=False, include_embeddings=False):
        h = self.pre_vq_conv(self.encoder(x, is_image))
        h = F.normalize(h, p=2, dim=1)
        vq_output = self.codebook(h)

        return vq_output['encodings'].detach() # [B, T, H, W]

    @torch.no_grad()
    def decode(self, encodings, is_image=False):
        z = F.embedding(encodings, self.codebook.embeddings)
        if z.ndim == 3:
            h = self.resolution // self.patch_size
            w = h
            z = rearrange(z, "b (t h w) c -> b c t h w", h=h, w=w)
        else:
            z = rearrange(z, "b t h w c -> b c t h w")
        
        z = self.post_vq_conv(z)

        return self.decoder(z, is_image).detach()
        
class SweetTok_base(BaseModel, PyTorchModelHubMixin, tags=["arxiv:2304.12244", "image-tokenization"], license="mit"):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        # This should be False for stage1 and True for stage2.
        self.finetune_decoder = config.model.vq_model.get("finetune_decoder", True)
        self.encoder = SweetTok_base_Encoder( 
            image_size = 256, image_channel=3, norm_type= 'batch', 
            block= 'tttt' , window_size=8, spatial_pos= 'rope',
            patch_embed = 'linear', patch_size = 8, temporal_patch_size= 4, defer_temporal_pool= False, defer_spatial_pool= False,
            spatial_depth=4, temporal_depth=4, causal_in_temporal_transformer=True, causal_in_peg=True, 
            dim = 512, dim_head=64, heads=8, attn_dropout=0.0, ff_dropout=0.0, ff_mult=4.0,
            initialize=False, sequence_length=17,
        )
        self.decoder = SweetTok_base_Decoder( 
            image_size = 256, image_channel=3, norm_type='batch', 
            block='tttt', window_size=8, spatial_pos='rope',
            patch_embed = None, patch_size = 8, temporal_patch_size= 4, defer_temporal_pool= False, defer_spatial_pool=False,
            spatial_depth=4, temporal_depth=4, causal_in_temporal_transformer=True, causal_in_peg=True, 
            dim = 512, dim_head=64, heads=8, attn_dropout=0.0, ff_dropout=0.0, ff_mult=4.0, 
            gen_upscale=None, initialize=False, sequence_length=17,
        )

        # self.quantize = Codebook(8192, 8, no_random_restart=True, restart_thres=1.0)
        self.pre_vq_conv = nn.Sequential(
                    Rearrange("b c t h w -> b t h w c"),
                    nn.Linear(512, config.model.vq_model.token_size),
                    Rearrange("b t h w c -> b c t h w")
                )
        self.post_vq_conv = nn.Sequential(
                    Rearrange("b c t h w -> b t h w c"),
                    nn.Linear(config.model.vq_model.token_size, 512),
                    Rearrange("b t h w c -> b c t h w")
        )
        
        # latent tokens settings
        
        self.apply(self._init_weights)

        self.quantize = VectorQuantizer(
            codebook_size=config.model.vq_model.codebook_size,
            token_size=config.model.vq_model.token_size,
            commitment_cost=config.model.vq_model.commitment_cost,
            use_l2_norm=config.model.vq_model.use_l2_norm,)
        
        if self.finetune_decoder:
            # Freeze encoder/quantizer/latent tokens
            self.latent_tokens.requires_grad_(False)
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            self.quantize.eval()
            self.quantize.requires_grad_(False)

            # Include MaskGiT-VQGAN's quantizer and decoder
            self.pixel_quantize = Pixel_Quantizer(
                num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
            self.pixel_decoder = Pixel_Decoder(OmegaConf.create(
                {"channel_mult": [1, 1, 2, 2, 4],
                "num_resolutions": 5,
                "dropout": 0.0,
                "hidden_channels": 128,
                "num_channels": 3,
                "num_res_blocks": 2,
                "resolution": 256,
                "z_channels": 256}))
        
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        if self.finetune_decoder:
            with torch.no_grad():
                self.encoder.eval()
                self.quantize.eval()
                z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
                z_quantized, result_dict = self.quantize(z)
                result_dict["quantizer_loss"] *= 0
                result_dict["commitment_loss"] *= 0
                result_dict["codebook_loss"] *= 0
        else:
            h = self.pre_vq_conv(self.encoder(x, False))
            h = F.normalize(h, p=2, dim=1)
            h = rearrange(h, 'b d t h w -> b d 1 (t h w)')
            z_quantized, result_dict = self.quantize(h)
            z_quantized = rearrange(z_quantized, 'b d 1 (t h w) -> b d t h w', h = 32, w = 32)
        return z_quantized, result_dict
    
    def decode(self, z_quantized):
        decoded = self.decoder(self.post_vq_conv(z_quantized), False)
        if self.finetune_decoder:
            quantized_states = torch.einsum(
                'nchw,cd->ndhw', decoded.softmax(1),
                self.pixel_quantize.embedding.weight)
            decoded = self.pixel_decoder(quantized_states)
        return decoded
    
    def decode_tokens(self, tokens):
        tokens = tokens.squeeze(1)
        batch, seq_len = tokens.shape # B x N
        z_quantized = self.quantize.get_codebook_entry(
            tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        decoded = self.decode(z_quantized)
        return decoded
    
    def forward(self, x):
        z_quantized, result_dict = self.encode(x)
        decoded = self.decode(z_quantized)

        return decoded, result_dict


class SweetTok_Compact(BaseModel, PyTorchModelHubMixin, tags=["arxiv:2304.12244", "image-tokenization"], license="mit"):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        # This should be False for stage1 and True for stage2.
        self.finetune_decoder = config.model.vq_model.get("finetune_decoder", True)
        self.encoder = SweetTok_Compact_Encoder( config = self.config,
            image_size = 256, image_channel=3, norm_type= 'batch', 
            block= 'ttwwtttt' , window_size=8, spatial_pos= 'rope',
            patch_embed = 'linear', patch_size = 8, temporal_patch_size= 4, defer_temporal_pool= False, defer_spatial_pool= False,
            spatial_depth=8, temporal_depth=4, causal_in_temporal_transformer=True, causal_in_peg=True, 
            dim = 512, dim_head=64, heads=8, attn_dropout=0.0, ff_dropout=0.0, ff_mult=4.0,
            initialize=False, sequence_length=17, use_temporal_transformer=True
        )
        self.decoder = SweetTok_Compact_Decoder( config = self.config,
            image_size = 256, image_channel=3, norm_type='batch', 
            block='tttttttt', window_size=8, spatial_pos='rope',
            patch_embed = None, patch_size = 8, temporal_patch_size= 4, defer_temporal_pool= False, defer_spatial_pool=False,
            spatial_depth=8, temporal_depth=4, causal_in_temporal_transformer=True, causal_in_peg=True, 
            dim = 512, dim_head=64, heads=8, attn_dropout=0.0, ff_dropout=0.0, ff_mult=4.0, 
            gen_upscale=None, initialize=False, sequence_length=17, use_temporal_transformer=True
        )

        
        # latent tokens settings
        self.num_spatial_latent_tokens = config.model.vq_model.num_spatial_latent_tokens
        self.num_temporal_latent_tokens = config.model.vq_model.num_temporal_latent_tokens
        self.num_intermediate_temporal_tokens = config.model.vq_model.num_intermediate_temporal_tokens
        scale = self.encoder.dim ** -0.5
        self.latent_spatial_tokens = nn.Parameter(
            scale * torch.randn(self.num_spatial_latent_tokens, self.encoder.dim))

        self.latent_temporal_tokens = nn.Parameter(
            scale * torch.randn(self.num_temporal_latent_tokens, self.encoder.dim))


        self.pre_vq_conv_ = nn.Sequential(
                    Rearrange("b c t h w -> b t h w c"),
                    nn.Linear(512, 512),
                    Rearrange("b t h w c -> b c t h w")
                )

        self.pre_vq_conv_temporal = nn.Sequential(
                    Rearrange("b c t h w -> b t h w c"),
                    nn.Linear(512, 512),
                    Rearrange("b t h w c -> b c t h w")
                )

        self.post_vq_conv_ = nn.Sequential(
                    Rearrange("b c t h w -> b t h w c"),
                    nn.Linear(512, 512),
                    Rearrange("b t h w c -> b c t h w")
        )

        self.post_vq_conv_temporal = nn.Sequential(
                    Rearrange("b c t h w -> b t h w c"),
                    nn.Linear(512, 512),
                    Rearrange("b t h w c -> b c t h w")
        )

        
        self.apply(self._init_weights)

        self.quantize = MLC_quantizer_noun(
            n_e=1024,
            e_dim=256,
            beta = 0.25,
            topk=2,)

        self.quantize_temporal = MLC_quantizer_verb(
            n_e=1024,
            e_dim=256,
            beta = 0.25,
            topk=2,)

        if config.experiment.train_stage == 1:
            state_dict =  torch.load(config.model.vq_model.image_stage1_weight, map_location=torch.device("cpu"))
            w_keys = list(state_dict.keys())
            self.load_state_dict(state_dict, strict=False)

        
        if self.finetune_decoder:
            # Freeze encoder/quantizer/latent tokens
            self.latent_spatial_tokens.requires_grad_(False)
            self.latent_temporal_tokens.requires_grad_(False)
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            self.quantize.eval()
            self.quantize.requires_grad_(False)
            self.quantize_temporal.eval()
            self.quantize_temporal.requires_grad_(False)
            self.pre_vq_conv_.eval()
            self.pre_vq_conv_.requires_grad_(False)
            self.pre_vq_conv_temporal.eval()
            self.pre_vq_conv_temporal.requires_grad_(False)

            # Include MaskGiT-VQGAN's quantizer and decoder
            self.pixel_decoder = OmniTokenizer_Decoder(
            image_size = 256, image_channel=3, norm_type='batch', 
            block='tttt', window_size=8, spatial_pos='rope',
            patch_embed = 'linear', patch_size = 8, temporal_patch_size= 4, defer_temporal_pool= False, defer_spatial_pool=False,
            spatial_depth=4, temporal_depth=4, causal_in_temporal_transformer=True, causal_in_peg=True, 
            dim = 512, dim_head=64, heads=8, attn_dropout=0.0, ff_dropout=0.0, ff_mult=4.0, 
            gen_upscale=None, initialize=False, sequence_length=17,
            )

            self.l2_code = True
            self.resolution = 256
            self.patch_size = 8

            self.pixel_codebook = Codebook(8192, 8, no_random_restart=True, restart_thres=1.0)
            self.pixel_post_vq_conv = nn.Sequential(
                    Rearrange("b c t h w -> b t h w c"),
                    nn.Linear(8, 512),
                    Rearrange("b t h w c -> b c t h w")
                )

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        if self.finetune_decoder:
            with torch.no_grad():
                self.encoder.eval()
                self.quantize.eval()
                self.quantize_temporal.eval()
                self.pre_vq_conv_.eval()
                self.pre_vq_conv_temporal.eval()
                z, spatial_latent_tokens, temporal_latent_tokens = self.encoder(x, is_image = False, spatial_latent_tokens=self.latent_spatial_tokens, temporal_latent_tokens=self.latent_temporal_tokens)
                spatial_latent_tokens = rearrange(spatial_latent_tokens, 'b d h w -> b d 1 h w')
                spatial_latent_tokens = self.pre_vq_conv_(spatial_latent_tokens)
                spatial_latent_tokens = F.normalize(spatial_latent_tokens, p=2, dim=1)
                spatial_latent_tokens = rearrange(spatial_latent_tokens, 'b d t h w -> b d 1 (t h w)')
                z_quantized, result_dict = self.quantize(spatial_latent_tokens)
                temporal_latent_tokens = rearrange(temporal_latent_tokens, 'b d h w -> b d 1 h w')
                temporal_latent_tokens = self.pre_vq_conv_temporal(temporal_latent_tokens)
                temporal_latent_tokens = F.normalize(temporal_latent_tokens, p=2, dim=1)
                temporal_latent_tokens = rearrange(temporal_latent_tokens, 'b d t h w -> b d 1 (t h w)')
                z_quantized_temporal, result_dict_temporal = self.quantize_temporal(temporal_latent_tokens)
                z_quantized = rearrange(z_quantized, 'b d 1 h -> b d 1 1 h')
                z_quantized_temporal = rearrange(z_quantized_temporal, 'b d 1 h -> b d 1 1 h')

                for k in result_dict.keys():
                    if 'loss' in k:
                        result_dict[k] += result_dict_temporal[k]
                    elif 'indices' in k:
                        result_dict[k] = torch.cat([result_dict[k], result_dict_temporal[k]], -1)
                        # result_dict[k] = result_dict_temporal[k]
                result_dict["quantizer_loss"] *= 0
                result_dict["commitment_loss"] *= 0
                result_dict["codebook_loss"] *= 0
        else:
            z, spatial_latent_tokens, temporal_latent_tokens = self.encoder(x, is_image = False, spatial_latent_tokens=self.latent_spatial_tokens, temporal_latent_tokens=self.latent_temporal_tokens)
            spatial_latent_tokens = rearrange(spatial_latent_tokens, 'b d h w -> b d 1 h w')
            spatial_latent_tokens = self.pre_vq_conv_(spatial_latent_tokens)
            spatial_latent_tokens = F.normalize(spatial_latent_tokens, p=2, dim=1)
            spatial_latent_tokens = rearrange(spatial_latent_tokens, 'b d t h w -> b d 1 (t h w)')
            z_quantized, result_dict = self.quantize(spatial_latent_tokens)
            temporal_latent_tokens = rearrange(temporal_latent_tokens, 'b d h w -> b d 1 h w')
            temporal_latent_tokens = self.pre_vq_conv_temporal(temporal_latent_tokens)
            temporal_latent_tokens = F.normalize(temporal_latent_tokens, p=2, dim=1)
            temporal_latent_tokens = rearrange(temporal_latent_tokens, 'b d t h w -> b d 1 (t h w)')
            z_quantized_temporal, result_dict_temporal = self.quantize_temporal(temporal_latent_tokens)
            z_quantized = rearrange(z_quantized, 'b d 1 h -> b d 1 1 h')
            z_quantized_temporal = rearrange(z_quantized_temporal, 'b d 1 h -> b d 1 1 h')

            for k in result_dict.keys():
                if 'loss' in k:
                    result_dict[k] += result_dict_temporal[k]
                elif 'indices' in k:
                    result_dict[k] = torch.cat([result_dict[k], result_dict_temporal[k]], -1)
                    # result_dict[k] = result_dict_temporal[k]
                
        return (z_quantized, z_quantized_temporal), result_dict
    
    def decode(self, z_quantized):
        spatial_latent_tokens = self.post_vq_conv_(z_quantized[0])
        spatial_latent_tokens = rearrange(spatial_latent_tokens,'b d 1 1 h -> b d 1 h')
        temporal_latent_tokens = self.post_vq_conv_temporal(z_quantized[1])
        temporal_latent_tokens = rearrange(temporal_latent_tokens,'b d 1 1 h -> b d 1 h')
        decoded = self.decoder((spatial_latent_tokens, temporal_latent_tokens), is_image = False)
        if self.finetune_decoder:
            quantized_states = torch.einsum(
                'ncthw,cd->ndthw', decoded.softmax(1),
                self.pixel_codebook.embeddings)
            quantized_states = self.pixel_post_vq_conv(quantized_states)
            decoded = self.pixel_decoder(quantized_states, False)
        return decoded
    
    def decode_tokens(self, tokens):
        # tokens = tokens.squeeze(1)
        batch, seq_len = tokens.shape # B x N
        spatial_tokens = tokens[:,:768]
        temporal_tokens = tokens[:,768:]
        z_quantized = self.quantize.get_codebook_entry(
            spatial_tokens.reshape(-1))
        z_quantized_temporal = self.quantize_temporal.get_codebook_entry(
            temporal_tokens.reshape(-1))
        z_quantized = rearrange(z_quantized, 'l d -> 1 d 1 1 l')
        z_quantized_temporal = rearrange(z_quantized_temporal, 'l d -> 1 d 1 1 l')
        decoded = self.decode((z_quantized, z_quantized_temporal))
        return decoded
    
    def forward(self, x):
        z_quantized, result_dict = self.encode(x)
        decoded = self.decode(z_quantized)

        return decoded, result_dict
