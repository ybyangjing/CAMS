import torch
from clip_modules import clip

class CustomTextEncoder(torch.nn.Module):
    def __init__(self, clip_model, dtype=torch.float16):
        super().__init__()
        self.dtype = dtype
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding        

    def tokenize(self, text):
        return torch.cat([clip.tokenize(tok) for tok in text])

    def encode_text(self, text, enable_pos_emb=True):
        token_ids = self.tokenize(text)
        text_features = self.forward(token_ids, None, enable_pos_emb)
        return text_features

    def forward(self, token_ids=None, token_tensors=None, enable_pos_emb=False):
        if token_tensors is not None:
            text_features = token_tensors
        else:
            text_features = self.token_embedding(token_ids)

        text_features = text_features
        x = (
            text_features + self.positional_embedding
            if enable_pos_emb
            else text_features
        )

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)     
        x = self.ln_final(x)
        
        tf = (                   
                x[
                    torch.arange(x.shape[0]), token_ids.argmax(dim=-1)
                ]  # POS of <EOS>
                @ self.text_projection
            )  
        return tf
