import torch
from torch import nn

class CSPInterface(nn.Module):
    def __init__(self, tokenize, clip_model, dset,args):
        super().__init__()
        self.tokenize = tokenize
        self.config = args
        self.dset = dset
        self.clip_model = clip_model
        self.attr_dropout = nn.Dropout(0.3)
        self.token_ids, soft_att_obj, com_ctx_vectors, att_ctx_vectors, obj_ctx_vectors = self.construct_soft_prompt()
        self.com_ctx_vectors = nn.Parameter(com_ctx_vectors)
        self.att_ctx_vectors = nn.Parameter(att_ctx_vectors)
        self.obj_ctx_vectors = nn.Parameter(obj_ctx_vectors)
        self.soft_att_obj = nn.Parameter(soft_att_obj)

    def construct_soft_prompt(self):
        classes = [cla.replace(".", " ").lower() for cla in self.dset.objs]
        attributes = [attr.replace(".", " ").lower() for attr in self.dset.attrs]
        
        self.num_att = len(attributes)
        self.num_cls = len(classes)

        
        token_ids = self.tokenize(self.config.prompt_template,
                              context_length=self.config.context_length).cuda()

        tokenized = torch.cat(
            [
                self.tokenize(tok, context_length=16)
                for tok in attributes + classes
            ]
        )
        orig_token_embedding = self.clip_model.token_embedding(tokenized.cuda())
        soft_att_obj = torch.zeros(
            (len(attributes) + len(classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = self.config.ctx_init
        assert isinstance(ctx_init, list)
        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenize(ctx_init,
                            context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(prompt)

        comp_ctx_vectors = embedding[0, 1 : 1 + n_ctx[0], :].to(self.clip_model.dtype)
        attr_ctx_vectors = embedding[1, 1 : 1 + n_ctx[1], :].to(self.clip_model.dtype)
        obj_ctx_vectors = embedding[2, 1 : 1 + n_ctx[2], :].to(self.clip_model.dtype)
        
        return token_ids, soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors


    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        token_tensor, num_elements = list(), [len(pair_idx), self.num_att, self.num_cls]
        for i_element in range(self.token_ids.shape[0]):
            class_token_ids = self.token_ids[i_element].repeat(num_elements[i_element], 1)
            token_tensor.append(self.clip_model.token_embedding(
                class_token_ids.cuda()
            ).type(self.clip_model.dtype))

        eos_idx = [int(self.token_ids[i_element].argmax()) for i_element in range(self.token_ids.shape[0])]
        soft_att_obj = self.attr_dropout(self.soft_att_obj)
        # com
        token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.clip_model.dtype)
        token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[
            obj_idx + self.num_att
        ].type(self.clip_model.dtype)
        token_tensor[0][
            :, 1 : len(self.com_ctx_vectors) + 1, :
        ] = self.com_ctx_vectors.type(self.clip_model.dtype)
        # attr
        token_tensor[1][:, eos_idx[1] - 2, :] = soft_att_obj[
            :self.num_att
        ].type(self.clip_model.dtype)
        token_tensor[1][
            :, 1 : len(self.att_ctx_vectors) + 1, :
        ] = self.att_ctx_vectors.type(self.clip_model.dtype)
        # obj
        token_tensor[2][:, eos_idx[2] - 1, :] = soft_att_obj[
            self.num_att:
        ].type(self.clip_model.dtype)
        token_tensor[2][
            :, 1 : len(self.obj_ctx_vectors) + 1, :
        ] = self.obj_ctx_vectors.type(self.clip_model.dtype)

        return token_tensor