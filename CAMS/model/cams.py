import torch
from torch import nn
from clip_modules import clip
import torch.nn.functional as F
from model.csp import CSPInterface
from clip_modules.text_encoder import CustomTextEncoder

""""
    CAMS: Towards Compositional Zero-Shot Learning via Gated Cross-Attention and Multi-Space Disentanglement 
"""

class CAMS(nn.Module):

    def __init__(self,args,dataset):
        super(CAMS, self).__init__()
        self.args = args
        self.dset = dataset
        
        self.clip_model = clip.load_clip(self.args) # Load the pre-trained CLIP. See "Learning Transferable Visual Models From Natural Language Supervision".
        self.image_encoder = self.clip_model.encode_image
        self.text_encoder = CustomTextEncoder(self.clip_model)
        
        self.csp = CSPInterface(clip.tokenize, self.clip_model, self.dset,self.args) # Use the soft prompt method. See "Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot Learning".
        attr2idx = self.dset.attr2idx
        obj2idx = self.dset.obj2idx
        self.train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                         for attr, obj in self.dset.train_pairs])        

        self.temp_logit = nn.Parameter(self.clip_model.logit_scale.exp().detach()) # Temperature
        
    def forward(self, data):
        img, att_id, obj_id , pair_id = data[0] , data[1], data[2] , data[3] 

        # Obtain the semantic representations of the four branches.
        att,obj,com,glb = self.image_encoder(img)
        att_semantic_reps = F.normalize(att, p=2, dim=-1) 
        obj_semantic_reps = F.normalize(obj, p=2, dim=-1) 
        com_semantic_reps = F.normalize(com, p=2, dim=-1) 
        glb_semantic_reps = F.normalize(glb, p=2, dim=-1)
        
        # Obtain the prompt representations.
        token_com,token_att,token_obj = self.csp.construct_token_tensors(self.train_pairs)
        com_prompt_reps,att_prompt_reps,obj_prompt_reps = self.get_text_features(token_com,token_att,token_obj)
        
        
        # Align the four branches. 
        att_branch = self.temp_logit * att_semantic_reps @ att_prompt_reps.T
        obj_branch = self.temp_logit * obj_semantic_reps @ obj_prompt_reps.T
        com_branch = self.temp_logit * com_semantic_reps @ com_prompt_reps.T
        glb_branch = self.temp_logit * glb_semantic_reps @ com_prompt_reps.T
        
        att_loss = F.cross_entropy(att_branch , att_id)
        obj_loss = F.cross_entropy(obj_branch , obj_id)
        com_loss = F.cross_entropy(com_branch , pair_id)
        glb_loss = F.cross_entropy(glb_branch , pair_id)
                     
        total_loss = self.args.att_loss_weight * att_loss + \
                     self.args.obj_loss_weight * obj_loss + \
                     self.args.com_loss_weight * com_loss + \
                     self.args.glb_loss_weight * glb_loss

        return total_loss
    
    def get_text_features(self,token_com,token_att,token_obj):
        text_features_com = self.text_encoder(
            self.csp.token_ids[0],
            token_com,
            enable_pos_emb=True
        )  
        text_features_att = self.text_encoder(
            self.csp.token_ids[1],
            token_att,
            enable_pos_emb=True
        )  
        text_features_obj = self.text_encoder(
            self.csp.token_ids[2],
            token_obj,
            enable_pos_emb=True
        )  
        text_features_com = F.normalize(text_features_com, p=2, dim=-1)  
        text_features_att = F.normalize(text_features_att, p=2, dim=-1)  
        text_features_obj = F.normalize(text_features_obj, p=2, dim=-1)  
        return text_features_com,text_features_att,text_features_obj
    
