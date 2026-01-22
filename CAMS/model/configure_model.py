import torch
from model.cams import CAMS

def configure_model(args,dataset):
    
    cams = CAMS(args,dataset)
    keywords = ['csp','norm','tr_','adapter','proj_att','proj_obj','proj_com','temp_logit','cross_attention','latent_units']
    
    # Freeze the pre-trained CLIP parameters.
    for name, param in cams.named_parameters():
        param.requires_grad = False
        if any(keyword in name for keyword in keywords):
            param.requires_grad = True
    
    optimizer = torch.optim.Adam(
        cams.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    return cams, optimizer

