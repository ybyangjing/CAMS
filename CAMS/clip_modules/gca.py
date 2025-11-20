
import torch
import torch.nn as nn
from collections import OrderedDict
from utils import LayerNorm,QuickGELU,Adapter,CrossAttentionLayer,Transformer as Trs


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            attn_mask: torch.Tensor = None,
            pos=0,
            args=None
    ):
        super().__init__()
        self.pos = pos
        self.attn = nn.MultiheadAttention(
            d_model, n_head
        )
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.m_layers = args.m_layers    
        
        # Fine-tune the visual encoder using LoRA. See "Parameter-Efficient Transfer Learning for NLP"
        if self.pos >= 24 - self.m_layers:
            self.adapter_attn = Adapter(d_model,args.adapter_dim,0.1)
            self.adapter_fnn = Adapter(d_model,args.adapter_dim,0.1)

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def std(self,x):
        adapt_x = self.adapter_attn(x,add_residual=False)
        x = x + self.attention(self.ln_1(x)) # Clip attention
        x = x + adapt_x
        adapt_x = self.adapter_fnn(x,add_residual=False)
        x = x + self.mlp(self.ln_2(x)) # Clip mlp
        x = x + adapt_x
        return x
    
    def forward(self, x):
        x,latent_queries,cross_attention = x
        # Extract semantic features using Latent Queries.
        if self.pos >= 24 - self.m_layers:
            x = self.std(x) # Loading LoRA
            latent_queries = cross_attention(latent_queries,x)  
        else:
            x = x + self.attention(self.ln_1(x)) # Clip attention
            x = x + self.mlp(self.ln_2(x)) # Clip mlp
        return x,latent_queries,cross_attention

class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            attn_mask: torch.Tensor = None,
            args=None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        res_layers = []
        for layer in range(layers):
            res_layers.append(
                ResidualAttentionBlock(
                    width,
                    heads,
                    attn_mask=attn_mask,
                    pos=layer,
                    args=args
                )
            )
        self.resblocks = nn.Sequential(*res_layers)

    def forward(self, x):
        return self.resblocks(x)

class VisionTransformer(nn.Module):
    def __init__(
            self,
            input_resolution: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            output_dim: int,
            args
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(
            width,
            layers,
            heads,
            args=args
        )
        
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
 
        self.proj_att = nn.Parameter(scale * torch.randn(width, output_dim))
        self.proj_obj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.proj_com = nn.Parameter(scale * torch.randn(width, output_dim))

        self.latent_queries = nn.Parameter(torch.randn(args.num_queries,width))

        self.drop = nn.Dropout(0.5)
        self.pre_norm = LayerNorm(width)
        self.ln_norm = LayerNorm(width)
        
        self.tr_a = Trs(width=1024,layers=1,heads=8,drop=0.1,batch_first=True)
        self.tr_o = Trs(width=1024,layers=1,heads=8,drop=0.1,batch_first=True)
        self.tr_c = Trs(width=1024,layers=1,heads=8,drop=0.1,batch_first=True)
        self.cross_attention = CrossAttentionLayer(1024,16,0.1)

    def before(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype)+ torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),x,],dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        return x

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = self.before(x)

        # Introduce latent_queries for extracting features
        latent_queries = self.drop(self.latent_queries)
        latent_queries = latent_queries.unsqueeze(0).repeat(batch_size,1,1)
        
        x,latent_queries = self.transformer([x,latent_queries,self.cross_attention])[:2]
        
        # Multi-Space Disentanglement
        att_queries = self.tr_a(latent_queries)
        obj_queries = self.tr_o(latent_queries)
        com_queries = self.tr_c(latent_queries)
        
        att_queries = self.ln_norm(att_queries.mean(1))
        obj_queries = self.ln_norm(obj_queries.mean(1))
        com_queries = self.ln_norm(com_queries.mean(1))

        att_queries = att_queries @ self.proj_att
        obj_queries = obj_queries @ self.proj_obj
        com_queries = com_queries @ self.proj_com

        att = self.drop(att_queries)
        obj = self.drop(obj_queries)
        com = self.drop(com_queries)
        ################################
        
        # glb representation
        x = x.permute(1, 0, 2)     
        x = self.ln_post(x[:,0,:])            
        glb = x @ self.proj 

        return att,obj,com,glb 