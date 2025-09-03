"""Vector quantizer.

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

Reference: 
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
    https://github.com/google-research/magvit/blob/main/videogvt/models/vqvae.py
"""
from typing import Mapping, Text, Tuple
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
from torch.cuda.amp import autocast

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv,GATConv

class GCN(torch.nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

class VectorQuantizer(torch.nn.Module):
    def __init__(self,
                 codebook_size: int = 1024,
                 token_size: int = 256,
                 commitment_cost: float = 0.25,
                 use_l2_norm: bool = False,
                 ):
        super().__init__()
        self.commitment_cost = commitment_cost

        self.embedding = torch.nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.use_l2_norm = use_l2_norm

    # Ensure quantization is performed using f32
    @autocast(enabled=False)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        z = z.float()
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = rearrange(z, 'b h w c -> (b h w) c')

        if self.use_l2_norm:
            z_flattened = torch.nn.functional.normalize(z_flattened, dim=-1)
            embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)
        else:
            embedding = self.embedding.weight
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, embedding.T)

        min_encoding_indices = torch.argmin(d, dim=1) # num_ele
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

        if self.use_l2_norm:
            z = torch.nn.functional.normalize(z, dim=-1)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        codebook_loss = torch.mean((z_quantized - z.detach()) **2)

        loss = commitment_loss + codebook_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            min_encoding_indices=min_encoding_indices.view(z_quantized.shape[0], z_quantized.shape[2], z_quantized.shape[3])
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        if len(indices.shape) == 1:
            z_quantized = self.embedding(indices)
        elif len(indices.shape) == 2:
            z_quantized = torch.einsum('bd,dn->bn', indices, self.embedding.weight)
        else:
            raise NotImplementedError
        if self.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        return z_quantized


class MLC_quantizer_noun(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, topk, remap=None, unknown_index="random",
                 sane_index_shape=True, legacy=True, use_idmap=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.topk = topk
        self.legacy = legacy
        self.use_idmap = use_idmap
        # nlp prior
        with open('codebook_priors/adj_noun_word_knowledge_qwen.pkl', 'rb') as handle:
            graph = pickle.load(handle)
        
        code_book = []
        adj_code_book_ = []
        noun_code_book_ = []
        for k, v in graph['adj_code_book_num'].items():
            if v > 10:
                adj_code_book_.append(k)
                code_book.append(k)

        for k, v in graph['noun_code_book_num'].items():
            if v > 10 and k not in graph['adj_code_book_num'].keys():
                noun_code_book_.append(k)
                code_book.append(k)

        code2index = {v: i for i, v in enumerate(code_book)}
        self.adj_index2code = {i:v for i,v in enumerate(adj_code_book_)}
        self.noun_index2code = {i:v for i,v in enumerate(noun_code_book_)}

        
        n = len(code_book)
        edges = graph['adj_noun_edges']
        edges_ = []
        for k, v in edges:
            if k in code2index.keys() and v in code2index.keys():
                edges_.append((code2index[k], code2index[v]))

        edges_ = edges_ + [(v, u) for (u, v) in edges_]
        #edges_ = edges_ + [(u, u) for u in range(n)]
        edges_ = torch.tensor(np.array(edges_))
        adj_vectors = []
        for name in adj_code_book_:
            if name in graph['adj_code_book_vec'].keys():
                adj_vectors.append(graph['adj_code_book_vec'][name].squeeze().squeeze())

        noun_vectors = []
        for name in noun_code_book_:
            if name in graph['noun_code_book_vec'].keys():
                noun_vectors.append(graph['noun_code_book_vec'][name].squeeze().squeeze())

        adj_vectors = torch.stack(adj_vectors, dim=0).type(torch.float32)

        noun_vectors = torch.stack(noun_vectors, dim=0).type(torch.float32)

        # self.code_book_mapping = nn.Sequential(
        #     nn.Linear(in_features=noun_vectors.shape[-1], out_features=self.e_dim),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(in_features=self.e_dim, out_features=self.e_dim),
        # )
        self.gcn = GCN(in_dim=noun_vectors.shape[-1],hid_dim=self.e_dim,out_dim=self.e_dim)
        # self.mlp = nn.Linear(256,512)

        self.adj_vectors = adj_vectors
        self.adj_len = adj_vectors.shape[0]
        self.noun_vectors = noun_vectors
        self.code = torch.concat((adj_vectors,noun_vectors),dim=0)        
        self.data = Data(edge_index=edges_.t().contiguous(), x=self.code)
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        total_embedding_weight = self.gcn(self.data.to(z.device))
        # total_embedding_weight = self.mlp(self.code)
        
        b = z.shape[0]
        ################################# adj embedding #########################################
        z_adj = z[:, :self.e_dim, :, :]
        b,c,h,w = z[:, :self.e_dim, :, :].shape
        z_adj = rearrange(z_adj, 'b c h w -> b h w c').contiguous()
        z_flattened = z_adj.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 
        embedding_weight = total_embedding_weight[:self.adj_len].type_as(z)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        _,min_encoding_indices1 = d.topk(k=self.topk, dim=1, largest=False)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_adj_q = F.embedding(min_encoding_indices1, embedding_weight)
        z_adj_q = z_adj_q.mean(dim=1)
        z_adj_q = z_adj_q.view(z_adj.shape)

        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_adj_q.detach()-z_adj)**2) + \
                   torch.mean((z_adj_q - z_adj.detach()) ** 2)
        else:
            loss = torch.mean((z_adj_q.detach()-z_adj)**2) + self.beta * \
                   torch.mean((z_adj_q - z_adj.detach()) ** 2)
            commit_loss = self.beta * \
                   torch.mean((z_adj_q - z_adj.detach()) ** 2)
            quan_loss = torch.mean((z_adj_q.detach()-z_adj)**2)

        # preserve gradients
        z_adj_q = z_adj + (z_adj_q - z_adj).detach()
        # reshape back to match original input shape
        z_adj_q = rearrange(z_adj_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices1 = min_encoding_indices1.reshape(z_adj.shape[0],-1) # add batch axis
            min_encoding_indices1 = self.remap_to_used(min_encoding_indices1)
            min_encoding_indices1 = min_encoding_indices1.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices1 = rearrange(min_encoding_indices1, '(b c) d -> b (c d)', c = 256)

        ################################# noun embedding #########################################
        z_noun = z[:, self.e_dim:, :, :]
        z_noun = rearrange(z_noun, 'b c h w -> b h w c').contiguous()
        z_flattened = z_noun.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 
        embedding_weight = total_embedding_weight[self.adj_len:].type_as(z)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        min_encoding_indices2 = torch.argmin(d, dim=1)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_noun_q = F.embedding(min_encoding_indices2, embedding_weight).view(z_noun.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = loss + self.beta * torch.mean((z_noun_q.detach()-z_noun)**2) + \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)
        else:
            loss = loss + torch.mean((z_noun_q.detach()-z_noun)**2) + self.beta * \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)
            commit_loss += self.beta * \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)
            quan_loss += torch.mean((z_noun_q.detach()-z_noun)**2)

        # preserve gradients
        z_noun_q = z_noun + (z_noun_q - z_noun).detach()

        # reshape back to match original input shape
        z_noun_q = rearrange(z_noun_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices2 = min_encoding_indices2.reshape(z_noun.shape[0],-1) # add batch axis
            min_encoding_indices2 = self.remap_to_used(min_encoding_indices2)
            min_encoding_indices2 = min_encoding_indices2.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices2 = rearrange(min_encoding_indices2, '(b c)-> b c', c = 256)

        result_dict = dict(
            quantizer_loss = loss,
            commitment_loss=commit_loss,
            codebook_loss=quan_loss,
            min_encoding_indices=torch.cat([min_encoding_indices1,min_encoding_indices2],dim=1)
        )

        return torch.cat([z_adj_q, z_noun_q], dim=1), result_dict

    

    def get_codebook_entry(self, indices):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again
        # shape specifying (batch, height, width, channel)   
        embedding_weight = self.gcn(self.data.to(indices.device))
        embedding_adj = embedding_weight[:self.adj_len]
        embedding_noun = embedding_weight[self.adj_len:]
        
        z_q = F.embedding(indices[:512], embedding_adj)

        z_q = z_q.reshape(1, 2, -1, z_q.shape[-1])
        z_q = z_q.mean(1).squeeze(0)

        z_q_n = F.embedding(indices[512:], embedding_noun)

        z_q_n = z_q_n.reshape(1, -1, z_q_n.shape[-1])
        z_q_n = z_q_n.squeeze(0)
        # z_q = torch.cat([z_q[:,0,: ,:], z_q[:,1,:,:]], dim=-1)

        # if shape is not None:
        #     z_q = z_q.view(shape)
        #     # reshape back to match original input shape
        #     z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return torch.cat([z_q, z_q_n], dim=1)


class MLC_quantizer_verb(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, topk, remap=None, unknown_index="random",
                 sane_index_shape=True, legacy=True, use_idmap=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.topk = topk
        self.legacy = legacy
        self.use_idmap = use_idmap
        # nlp prior
        with open('codebook_priors/big_nlp_word_knowledge_clip_video_qwen.pkl', 'rb') as handle:
            graph = pickle.load(handle)
        
        code_book = []
        adj_code_book_ = []
        noun_code_book_ = []
        for k, v in graph['adverb_code_book_num'].items():
            adj_code_book_.append(k)
            code_book.append(k)

        for k, v in graph['verb_code_book_num'].items():
            noun_code_book_.append(k)
            code_book.append(k)

        code2index = {v: i for i, v in enumerate(code_book)}
        self.adj_index2code = {i:v for i,v in enumerate(adj_code_book_)}
        self.noun_index2code = {i:v for i,v in enumerate(noun_code_book_)}


        n = len(code_book)
        edges = graph['verb_adverb_edges']
        edges_ = []
        for k, v in edges:
            if k in code2index.keys() and v in code2index.keys():
                edges_.append((code2index[k], code2index[v]))

        edges_ = edges_ + [(v, u) for (u, v) in edges_]
        #edges_ = edges_ + [(u, u) for u in range(n)]
        edges_ = torch.tensor(np.array(edges_))
        adj_vectors = []
        for name in adj_code_book_:
            if name in graph['adverb_vec'].keys():
                adj_vectors.append(graph['adverb_vec'][name])

        noun_vectors = []
        for name in noun_code_book_:
            if name in graph['verb_vec'].keys():
                noun_vectors.append(graph['verb_vec'][name])

        adj_vectors = torch.stack(adj_vectors, dim=0).type(torch.float32)

        noun_vectors = torch.stack(noun_vectors, dim=0).type(torch.float32)

        self.code_book_mapping = nn.Sequential(
            nn.Linear(in_features=noun_vectors.shape[-1], out_features=self.e_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=self.e_dim, out_features=self.e_dim),
        )
        self.gcn = GCN(in_dim=noun_vectors.shape[-1],hid_dim=self.e_dim,out_dim=self.e_dim)
        # self.mlp = nn.Linear(256,512)

        self.adj_vectors = adj_vectors
        self.adj_len = adj_vectors.shape[0]
        self.noun_vectors = noun_vectors
        self.code = torch.concat((adj_vectors,noun_vectors),dim=0)        
        self.data = Data(edge_index=edges_.t().contiguous(), x=self.code)
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        total_embedding_weight = self.gcn(self.data.to(z.device))
        # total_embedding_weight = self.mlp(self.code)
        
        b = z.shape[0]
        ################################# adj embedding #########################################
        z_adj = z[:, :self.e_dim, :, :]
        b,c,h,w = z[:, :self.e_dim, :, :].shape
        z_adj = rearrange(z_adj, 'b c h w -> b h w c').contiguous()
        z_flattened = z_adj.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 
        embedding_weight = total_embedding_weight[:self.adj_len].type_as(z)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        _,min_encoding_indices1 = d.topk(k=self.topk, dim=1, largest=False)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_adj_q = F.embedding(min_encoding_indices1, embedding_weight)
        z_adj_q = z_adj_q.mean(dim=1)
        z_adj_q = z_adj_q.view(z_adj.shape)

        perplexity = None
        min_encodings = None
        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_adj_q.detach()-z_adj)**2) + \
                   torch.mean((z_adj_q - z_adj.detach()) ** 2)
        else:
            loss = torch.mean((z_adj_q.detach()-z_adj)**2) + self.beta * \
                   torch.mean((z_adj_q - z_adj.detach()) ** 2)
            commit_loss = self.beta * \
                   torch.mean((z_adj_q - z_adj.detach()) ** 2)
            quan_loss = torch.mean((z_adj_q.detach()-z_adj)**2)

        # preserve gradients
        z_adj_q = z_adj + (z_adj_q - z_adj).detach()

        # reshape back to match original input shape
        z_adj_q = rearrange(z_adj_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices1 = min_encoding_indices1.reshape(z_adj.shape[0],-1) # add batch axis
            min_encoding_indices1 = self.remap_to_used(min_encoding_indices1)
            min_encoding_indices1 = min_encoding_indices1.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices1 = rearrange(min_encoding_indices1, '(b c) d -> b (c d)', c = 1024)

        ################################# noun embedding #########################################
        z_noun = z[:, self.e_dim:, :, :]
        z_noun = rearrange(z_noun, 'b c h w -> b h w c').contiguous()
        z_flattened = z_noun.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # embedding 
        embedding_weight = total_embedding_weight[self.adj_len:].type_as(z)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding_weight, 'n d -> d n'))

        min_encoding_indices2 = torch.argmin(d, dim=1)
        # z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_noun_q = F.embedding(min_encoding_indices2, embedding_weight).view(z_noun.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = loss + self.beta * torch.mean((z_noun_q.detach()-z_noun)**2) + \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)
        else:
            loss = loss + torch.mean((z_noun_q.detach()-z_noun)**2) + self.beta * \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)
            commit_loss += self.beta * \
                   torch.mean((z_noun_q - z_noun.detach()) ** 2)
            quan_loss += torch.mean((z_noun_q.detach()-z_noun)**2)

        # preserve gradients
        z_noun_q = z_noun + (z_noun_q - z_noun).detach()

        # reshape back to match original input shape
        z_noun_q = rearrange(z_noun_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices2 = min_encoding_indices2.reshape(z_noun.shape[0],-1) # add batch axis
            min_encoding_indices2 = self.remap_to_used(min_encoding_indices2)
            min_encoding_indices2 = min_encoding_indices2.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices2 = rearrange(min_encoding_indices2, '(b c) -> b c ', c = 1024)

        result_dict = dict(
            quantizer_loss = loss,
            commitment_loss=commit_loss,
            codebook_loss=quan_loss,
            min_encoding_indices=torch.cat([min_encoding_indices1,min_encoding_indices2],dim=1)
        )

        return torch.cat([z_adj_q, z_noun_q], dim=1), result_dict
    

    def get_codebook_entry(self, indices):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again
        # shape specifying (batch, height, width, channel)   
        embedding_weight = self.gcn(self.data.to(indices.device))
        embedding_adj = embedding_weight[:self.adj_len]
        embedding_noun = embedding_weight[self.adj_len:]
        
        z_q = F.embedding(indices[:2048], embedding_adj)

        z_q = z_q.reshape(1, 2, -1, z_q.shape[-1])
        z_q = z_q.mean(1).squeeze(0)

        z_q_n = F.embedding(indices[2048:], embedding_noun)

        z_q_n = z_q_n.reshape(1, -1, z_q_n.shape[-1])
        z_q_n = z_q_n.squeeze(0)
        # z_q = torch.cat([z_q[:,0,: ,:], z_q[:,1,:,:]], dim=-1)

        # if shape is not None:
        #     z_q = z_q.view(shape)
        #     # reshape back to match original input shape
        #     z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return torch.cat([z_q, z_q_n], dim=1)


