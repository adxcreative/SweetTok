import argparse
import random

import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.scheduler.cosine_lr import CosineLRScheduler

from .gpt import GPT
from .encoders import Labelator, SOSProvider, Identity
from modeling.modules.base_model import BaseModel
from pathlib import Path


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Net2NetTransformer(BaseModel):
    def __init__(self,
                 config,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="video",
                 cond_stage_key="label",
                 pkeep=1.0,
                 sos_token=0,
                 ):
        super().__init__()
        self.config = config.model.generator
        self.class_cond_dim = self.config.condition_num_classes
        self.be_unconditional = self.config.unconditional
        self.sos_token = sos_token
        self.spatial_start = 0
        self.spatial_end = 1
        self.temporal_start = 2
        self.temporal_end = 3
        self.sms = 4
        self.smn = 5
        self.tms = 6
        self.tmn = 7

        self.first_stage_key = self.config.first_stage_key
        self.cond_stage_key = cond_stage_key
        self.vtokens = self.config.vtokens
        self.sample_every_n_latent_frames = getattr(self.config, 'sample_every_n_latent_frames', 0)
        
        # self.init_first_stage_from_ckpt(self.config)
        self.first_stage_vocab_size = 10481 + 11139
        self.init_cond_stage_from_ckpt(self.config)

        # if not hasattr(self.config, "starts_with_sos"):
        #     self.config.starts_with_sos = False
        
        # if not hasattr(self.config, "p_drop_cond"):
        #     self.config.p_drop_cond = False

        # if not hasattr(self.config, "class_first"):
        #     self.config.class_first = False
        
        self.starts_with_sos = self.config.starts_with_sos
        self.sos_provider = SOSProvider(self.sos_token)
        self.ss_provider = SOSProvider(self.spatial_start)
        self.sn_provider = SOSProvider(self.spatial_end)
        self.ts_provider = SOSProvider(self.temporal_start)
        self.tn_provider = SOSProvider(self.temporal_end)
        self.sms_provider = SOSProvider(self.sms)
        self.smn_provider = SOSProvider(self.smn)
        self.tms_provider = SOSProvider(self.tms)
        self.tmn_provider = SOSProvider(self.tmn)
        self.p_drop_cond = None
        self.class_first = self.config.class_first

        # import pdb; pdb.set_trace()
        if self.be_unconditional:
            self.starts_with_sos = False

        gpt_vocab_size = self.first_stage_vocab_size + self.cond_stage_vocab_size + 8
        if self.starts_with_sos:
            gpt_vocab_size += 8
        


        self.config.transformer_dropout = 0.
        
        self.transformer = GPT(self.config, gpt_vocab_size, 3848, n_layer=24, n_head=16, 
                                n_embd=1536, vtokens_pos=self.config.vtokens_pos, n_unmasked=0. , embd_pdrop=self.config.transformer_dropout, resid_pdrop=self.config.transformer_dropout, attn_pdrop=self.config.transformer_dropout)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.pkeep = pkeep
        # self.save_hyperparameters()
        
        self.automatic_optimization = False
        self.grad_accumulates = 1
        self.grad_clip_val = 1.0
    
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

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_cond_stage_from_ckpt(self, args):
        if self.cond_stage_key=='label' and not self.be_unconditional:
            model = Labelator(n_classes=args.condition_num_classes)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model
            self.cond_stage_vocab_size = self.class_cond_dim
        elif self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key
            self.cond_stage_model = SOSProvider(self.sos_token)
            self.cond_stage_vocab_size = 0
        else:
            ValueError('conditional model %s is not implementated'%self.cond_stage_key)

    def forward(self, x, c, cbox=None):
        is_image = x.ndim == 4
        # one step to produce the logits
        z_indices = x
        z_indices[:,512:768] += 5078
        z_indices[:,768:2816] += (5078 + 5403)
        z_indices[:,2816:] += (5078 + 5403 + 1872)
        _, c_indices = self.encode_to_c(c, is_image)
        # import pdb; pdb.set_trace()
        if self.starts_with_sos:
            _, sos = self.sos_provider.encode(c)
            _, ss = self.ss_provider.encode(c)
            _, sn = self.sn_provider.encode(c)
            _, ts = self.ts_provider.encode(c)
            _, tn = self.tn_provider.encode(c)
            _, sms = self.sms_provider.encode(c)
            _, smn = self.smn_provider.encode(c)
            _, tms = self.tms_provider.encode(c)
            _, tmn = self.tmn_provider.encode(c)

            c_indices = c_indices + 9
            z_indices = z_indices + self.cond_stage_vocab_size + 9
            z_indices = torch.cat((ss,z_indices[:,:512], smn, sms, z_indices[:,512:768],sn,ts,z_indices[:,768:2816],tms,tmn,z_indices[:,2816:],tn), dim=1)
        else:
            _, ss = self.ss_provider.encode(c)
            _, sn = self.sn_provider.encode(c)
            _, ts = self.ts_provider.encode(c)
            _, tn = self.tn_provider.encode(c)
            _, sms = self.sms_provider.encode(c)
            _, smn = self.smn_provider.encode(c)
            _, tms = self.tms_provider.encode(c)
            _, tmn = self.tmn_provider.encode(c)
            z_indices = z_indices + self.cond_stage_vocab_size + 8
            z_indices = torch.cat((ss,z_indices[:,:512], smn, sms, z_indices[:,512:768],sn,ts,z_indices[:,768:2816],tms,tmn,z_indices[:,2816:],tn), dim=1)
        
        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        # print(c_indices)
        if self.starts_with_sos:
            if self.p_drop_cond is not None:
                if random.random() > self.p_drop_cond:
                    if self.class_first:
                        cz_indices = torch.cat((c_indices, sos, a_indices), dim=1)
                    else:
                        cz_indices = torch.cat((sos, c_indices, a_indices), dim=1)
                    prefix_len = 1+c_indices.shape[1]-1
                else:
                    cz_indices = torch.cat((c_indices, a_indices), dim=1)
                    prefix_len = c_indices.shape[1]-1
            else:
                if self.class_first:
                    cz_indices = torch.cat((c_indices, sos, a_indices), dim=1)
                else:
                    cz_indices = torch.cat((sos, c_indices, a_indices), dim=1)
                
                prefix_len = 1+c_indices.shape[1]-1
        
        else:
            cz_indices = torch.cat((c_indices, a_indices), dim=1)
            prefix_len = c_indices.shape[1]-1
        
        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1], cbox=cbox)
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, prefix_len:]
        
        assert logits.shape[1] == target.shape[1]
        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        x = torch.cat((c,x),dim=1).long()
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape)==2
            noise_shape = (x.shape[0], steps-1)
            #noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
            x = torch.cat((x,noise),dim=1)
            logits, _ = self.transformer(x)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1],shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0],shape[1],shape[2])
                ix = ix.reshape(shape[0],shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix[:, c.shape[1]-1:]
        else:
            for k in range(steps):
                if callback is not None:
                    callback(k)
                assert x.size(1) <= block_size # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                logits, _ = self.transformer(x_cond)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            # cut off conditioning
            x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def encode_to_z(self, x, is_image):
        if self.vtokens:
            targets = x.reshape(x.shape[0], -1)
        else:
            x, targets = self.first_stage_model.encode(x, is_image, include_embeddings=True)
            if self.sample_every_n_latent_frames > 0:
                x = x[:, :, ::self.sample_every_n_latent_frames]
                targets = targets[:, ::self.sample_every_n_latent_frames]
            x = shift_dim(x, 1, -1)
            targets = targets.reshape(targets.shape[0], -1)
        return x, targets

    @torch.no_grad()
    def encode_to_c(self, c, is_image):
        if isinstance(self.cond_stage_model, Labelator) or isinstance(self.cond_stage_model, SOSProvider):
            quant_c, indices = self.cond_stage_model.encode(c)
        else:
            quant_c, indices = self.cond_stage_model.encode(c, is_image, include_embeddings=True)
        
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    def get_input(self, key, batch):
        x = batch[key]
        # if x.dtype == torch.double:
            # x = x.float()
        return x

    def get_xc(self, batch, N=None):
        """x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]"""
        if isinstance(batch, dict):
            x = batch[self.first_stage_key]
            c = batch[self.cond_stage_key]
        
        else:
            assert isinstance(batch, list) and len(batch) == 1
            x = batch[0][self.first_stage_key]
            c = batch[0][self.cond_stage_key]

        if N is not None:
            x = x[:N]
            c = c[:N]
        
        return x, c

    def shared_step(self, batch, batch_idx):
        if not self.vtokens:
            self.first_stage_model.eval()
        x, c = self.get_xc(batch)
        if self.args.vtokens_pos:
            cbox = batch['cbox']
        else:
            cbox = None
        
        logits, target = self(x, c, cbox)
        # print(logits.shape, target.shape)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        acc1, acc5 = accuracy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), topk=(1, 5))
        return loss, acc1, acc5

    def training_step(self, batch, batch_idx):
        sch = self.lr_schedulers()
        opt = self.optimizers()

        loss, acc1, acc5 = self.shared_step(batch, batch_idx)
        # print(batch_idx, loss)

        self.manual_backward(loss)

        cur_global_step = self.global_step
        if (cur_global_step + 1) % self.grad_accumulates == 0:
            if self.grad_clip_val is not None:
                self.clip_gradients(opt, gradient_clip_val=self.grad_clip_val)
                
            opt.step()
            
            sch.step(cur_global_step)
            opt.zero_grad()

        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/acc1', acc1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/acc5', acc5, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc1, acc5 = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/acc1', acc1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/acc5', acc5, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        if self.args.vtokens_pos:
            no_decay.add('vtokens_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))

        lr_min = self.args.lr_min
        train_iters = self.args.max_steps
        warmup_steps = self.args.warmup_steps
        warmup_lr_init = self.args.warmup_lr_init

       
        scheduler = CosineLRScheduler(
            optimizer,
            lr_min = lr_min,
            t_initial = train_iters,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_steps,
            cycle_mul = 1.,
            cycle_limit=1,
            t_in_epochs=True,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    

    def log_images(self, batch, **kwargs):
        log = dict()
        if isinstance(batch, list):
            batch = batch[0]
        
        x = batch[self.first_stage_key]
        c = batch[self.cond_stage_key]

        logits, _ = self(x, c)
        probs = F.softmax(logits, dim=-1)
        _, ix = torch.topk(probs, k=1, dim=-1)
        
        index = torch.clamp(ix-self.cond_stage_vocab_size, min=0, max=self.first_stage_vocab_size-1).squeeze(-1)
        x_rec = self.first_stage_model.decode(index, is_image=(x.ndim==4))
        
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log

    def log_videos(self, batch, **kwargs):
        log = dict()
        if isinstance(batch, list):
            batch = batch[0]
        
        x = batch[self.first_stage_key]
        c = batch[self.cond_stage_key]
        
        logits, _ = self(x, c)
        probs = F.softmax(logits, dim=-1)
        _, ix = torch.topk(probs, k=1, dim=-1)
        
        index = torch.clamp(ix-self.cond_stage_vocab_size, min=0, max=self.first_stage_vocab_size-1).squeeze(-1)

        x_rec = self.first_stage_model.decode(index, is_image=(x.ndim==4))


        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


class Net2NetTransformer_scb(BaseModel):
    def __init__(self,
                 config,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="video",
                 cond_stage_key="label",
                 pkeep=1.0,
                 sos_token=0,
                 ):
        super().__init__()
        self.config = config.model.generator
        self.class_cond_dim = self.config.condition_num_classes
        self.be_unconditional = self.config.unconditional
        self.sos_token = sos_token
        self.spatial_start = 1
        self.spatial_end = 2
        self.temporal_start = 3
        self.temporal_end = 4


        self.first_stage_key = self.config.first_stage_key
        self.cond_stage_key = cond_stage_key
        self.vtokens = self.config.vtokens
        self.sample_every_n_latent_frames = getattr(self.config, 'sample_every_n_latent_frames', 0)
        
        # self.init_first_stage_from_ckpt(self.config)
        self.first_stage_vocab_size = 10481 + 11139
        self.init_cond_stage_from_ckpt(self.config)

        # if not hasattr(self.config, "starts_with_sos"):
        #     self.config.starts_with_sos = False
        
        # if not hasattr(self.config, "p_drop_cond"):
        #     self.config.p_drop_cond = False

        # if not hasattr(self.config, "class_first"):
        #     self.config.class_first = False
        
        self.starts_with_sos = self.config.starts_with_sos
        self.sos_provider = SOSProvider(self.sos_token)
        self.ss_provider = SOSProvider(self.spatial_start)
        self.sn_provider = SOSProvider(self.spatial_end)
        self.ts_provider = SOSProvider(self.temporal_start)
        self.tn_provider = SOSProvider(self.temporal_end)

        self.p_drop_cond = self.config.p_drop_cond
        self.class_first = self.config.class_first

        if self.be_unconditional:
            self.starts_with_sos = False

        gpt_vocab_size = self.first_stage_vocab_size + self.cond_stage_vocab_size
        if self.starts_with_sos:
            gpt_vocab_size += 5
        


        self.config.transformer_dropout = 0.
        
        self.transformer = GPT(self.config, gpt_vocab_size, 1285, n_layer=24, n_head=16, 
                                n_embd=1536, vtokens_pos=self.config.vtokens_pos, n_unmasked=0. , embd_pdrop=self.config.transformer_dropout, resid_pdrop=self.config.transformer_dropout, attn_pdrop=self.config.transformer_dropout)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.pkeep = pkeep
        # self.save_hyperparameters()
        
        self.automatic_optimization = False
        self.grad_accumulates = 1
        self.grad_clip_val = 1.0
    
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

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_cond_stage_from_ckpt(self, args):
        if self.cond_stage_key=='label' and not self.be_unconditional:
            model = Labelator(n_classes=args.condition_num_classes)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model
            self.cond_stage_vocab_size = self.class_cond_dim
        elif self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key
            self.cond_stage_model = SOSProvider(self.sos_token)
            self.cond_stage_vocab_size = 0
        else:
            ValueError('conditional model %s is not implementated'%self.cond_stage_key)

    def forward(self, x, c, cbox=None):
        is_image = x.ndim == 4
        # one step to produce the logits
        z_indices = x
        z_indices[:,256:] += 10481
        _, c_indices = self.encode_to_c(c, is_image)
        # import pdb; pdb.set_trace()
        if self.starts_with_sos:
            _, sos = self.sos_provider.encode(c)
            _, ss = self.ss_provider.encode(c)
            _, sn = self.sn_provider.encode(c)
            _, ts = self.ts_provider.encode(c)
            _, tn = self.tn_provider.encode(c)


            c_indices = c_indices + 5
            z_indices = z_indices + self.cond_stage_vocab_size + 5
            z_indices = torch.cat((ss,z_indices[:,:256],sn,ts,z_indices[:,256:],tn), dim=1)
        else:
            z_indices = z_indices + self.cond_stage_vocab_size
        
        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        # print(c_indices)
        if self.starts_with_sos:
            if self.p_drop_cond is not None:
                if random.random() > self.p_drop_cond:
                    if self.class_first:
                        cz_indices = torch.cat((c_indices, sos, a_indices), dim=1)
                    else:
                        cz_indices = torch.cat((sos, c_indices, a_indices), dim=1)
                    prefix_len = 1+c_indices.shape[1]-1
                else:
                    cz_indices = torch.cat((c_indices, a_indices), dim=1)
                    prefix_len = c_indices.shape[1]-1
            else:
                if self.class_first:
                    cz_indices = torch.cat((c_indices, sos, a_indices), dim=1)
                else:
                    cz_indices = torch.cat((sos, c_indices, a_indices), dim=1)
                
                prefix_len = 1+c_indices.shape[1]-1
        
        else:
            cz_indices = torch.cat((c_indices, a_indices), dim=1)
            prefix_len = c_indices.shape[1]-1
        
        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1], cbox=cbox)
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, prefix_len:]
        
        assert logits.shape[1] == target.shape[1]
        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        x = torch.cat((c,x),dim=1).long()
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape)==2
            noise_shape = (x.shape[0], steps-1)
            #noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
            x = torch.cat((x,noise),dim=1)
            logits, _ = self.transformer(x)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1],shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0],shape[1],shape[2])
                ix = ix.reshape(shape[0],shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix[:, c.shape[1]-1:]
        else:
            for k in range(steps):
                if callback is not None:
                    callback(k)
                assert x.size(1) <= block_size # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                logits, _ = self.transformer(x_cond)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            # cut off conditioning
            x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def encode_to_z(self, x, is_image):
        if self.vtokens:
            targets = x.reshape(x.shape[0], -1)
        else:
            x, targets = self.first_stage_model.encode(x, is_image, include_embeddings=True)
            if self.sample_every_n_latent_frames > 0:
                x = x[:, :, ::self.sample_every_n_latent_frames]
                targets = targets[:, ::self.sample_every_n_latent_frames]
            x = shift_dim(x, 1, -1)
            targets = targets.reshape(targets.shape[0], -1)
        return x, targets

    @torch.no_grad()
    def encode_to_c(self, c, is_image):
        if isinstance(self.cond_stage_model, Labelator) or isinstance(self.cond_stage_model, SOSProvider):
            quant_c, indices = self.cond_stage_model.encode(c)
        else:
            quant_c, indices = self.cond_stage_model.encode(c, is_image, include_embeddings=True)
        
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    def get_input(self, key, batch):
        x = batch[key]
        # if x.dtype == torch.double:
            # x = x.float()
        return x

    def get_xc(self, batch, N=None):
        """x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]"""
        if isinstance(batch, dict):
            x = batch[self.first_stage_key]
            c = batch[self.cond_stage_key]
        
        else:
            assert isinstance(batch, list) and len(batch) == 1
            x = batch[0][self.first_stage_key]
            c = batch[0][self.cond_stage_key]

        if N is not None:
            x = x[:N]
            c = c[:N]
        
        return x, c

    def shared_step(self, batch, batch_idx):
        if not self.vtokens:
            self.first_stage_model.eval()
        x, c = self.get_xc(batch)
        if self.args.vtokens_pos:
            cbox = batch['cbox']
        else:
            cbox = None
        
        logits, target = self(x, c, cbox)
        # print(logits.shape, target.shape)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        acc1, acc5 = accuracy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), topk=(1, 5))
        return loss, acc1, acc5

    def training_step(self, batch, batch_idx):
        sch = self.lr_schedulers()
        opt = self.optimizers()

        loss, acc1, acc5 = self.shared_step(batch, batch_idx)
        # print(batch_idx, loss)

        self.manual_backward(loss)

        cur_global_step = self.global_step
        if (cur_global_step + 1) % self.grad_accumulates == 0:
            if self.grad_clip_val is not None:
                self.clip_gradients(opt, gradient_clip_val=self.grad_clip_val)
                
            opt.step()
            
            sch.step(cur_global_step)
            opt.zero_grad()

        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/acc1', acc1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/acc5', acc5, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc1, acc5 = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/acc1', acc1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val/acc5', acc5, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        if self.args.vtokens_pos:
            no_decay.add('vtokens_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))

        lr_min = self.args.lr_min
        train_iters = self.args.max_steps
        warmup_steps = self.args.warmup_steps
        warmup_lr_init = self.args.warmup_lr_init

       
        scheduler = CosineLRScheduler(
            optimizer,
            lr_min = lr_min,
            t_initial = train_iters,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_steps,
            cycle_mul = 1.,
            cycle_limit=1,
            t_in_epochs=True,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    

    def log_images(self, batch, **kwargs):
        log = dict()
        if isinstance(batch, list):
            batch = batch[0]
        
        x = batch[self.first_stage_key]
        c = batch[self.cond_stage_key]

        logits, _ = self(x, c)
        probs = F.softmax(logits, dim=-1)
        _, ix = torch.topk(probs, k=1, dim=-1)
        
        index = torch.clamp(ix-self.cond_stage_vocab_size, min=0, max=self.first_stage_vocab_size-1).squeeze(-1)
        x_rec = self.first_stage_model.decode(index, is_image=(x.ndim==4))
        
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log

    def log_videos(self, batch, **kwargs):
        log = dict()
        if isinstance(batch, list):
            batch = batch[0]
        
        x = batch[self.first_stage_key]
        c = batch[self.cond_stage_key]
        
        logits, _ = self(x, c)
        probs = F.softmax(logits, dim=-1)
        _, ix = torch.topk(probs, k=1, dim=-1)
        
        index = torch.clamp(ix-self.cond_stage_vocab_size, min=0, max=self.first_stage_vocab_size-1).squeeze(-1)

        x_rec = self.first_stage_model.decode(index, is_image=(x.ndim==4))


        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log

