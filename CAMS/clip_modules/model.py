import torch
import numpy as np
from torch import nn
from collections import OrderedDict
from clip_modules.gca import VisionTransformer

"""
    CLIP Architecture
"""

class Transformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None,args=None,drop=0.,batch_first=False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[AttentionBlock(width, heads, attn_mask,drop,batch_first,args) for _ in range(layers)])
    def forward(self, x):
        for _, layer in enumerate(self.resblocks.children()):
            x = layer(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, d_model, heads, attn_mask,drop,batch_first,args):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model,heads,batch_first=batch_first)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ('c_fc', nn.Linear(d_model, d_model * 4)),
                    ('gelu', nn.GELU()),
                    ('c_proj', nn.Linear(d_model * 4, d_model))
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask
        
    def attention(self, x):
        attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, attn_mask=attn_mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
def build_model(state_dict, context_length,args):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("visual.")
                   and k.endswith(".attn.in_proj_weight")
            ]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5
        )
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2]
                    for k in state_dict
                    if k.startswith(f"visual.layer{b}")
                )
            )
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1)
            ** 0.5
        )
        vision_patch_size = None
        assert (
                output_width ** 2 + 1
                == state_dict["visual.attnpool.positional_embedding"].shape[0]
        )
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]

    if context_length != 77:
        if context_length > 77:
            zeros = torch.zeros((context_length - 77, embed_dim))
            state_dict["positional_embedding"] = torch.cat(
                (state_dict["positional_embedding"], zeros), dim=0
            )

        else:
            state_dict["positional_embedding"] = state_dict[
                                                     "positional_embedding"
                                                 ][:context_length, :]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2]
            for k in state_dict
            if k.startswith(f"transformer.resblocks")
        )
    )
    model = CLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        args
    ) 
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    return model.eval()

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim, 
                 image_resolution, 
                 vision_layers, 
                 vision_width,
                 vision_patch_size,  
                 # transformer of text
                 context_length, 
                 vocab_size,  
                 transformer_width,
                 transformer_heads,
                 transformer_layers,
                 args):
        super().__init__()
        self.context_length = context_length
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            args=args
        )
        self.initialize_parameters()

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            args=args
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = nn.LayerNorm(transformer_width)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2) 
        x = self.transformer(x)
        x = x[1].permute(1, 0, 2) 
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1) 
        return mask

    def initialize_parameters(self):
        nn.init.normal_(self.visual.positional_embedding, std=0.01)
        proj_std = (self.visual.transformer.width ** -0.5) * (
                (2 * self.visual.transformer.layers) ** -0.5
        )
        attn_std = self.visual.transformer.width ** -0.5
        fc_std = (2 * self.visual.transformer.width) ** -0.5
        for block in self.visual.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        nn.init.normal_(self.visual.proj, std=self.visual.transformer.width ** -0.5)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
