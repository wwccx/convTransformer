from torch import optim as optim
from ATTCG import ATTCG

def build_optimizer(config, model):

    _optim = config.optim
    if _optim.lower() == 'adamw':
        skip_keywords = {}
        if hasattr(model, 'no_weight_decay_keywords'):
            skip_keywords = model.no_weight_decay_keywords()
        parameters = set_weight_decay(model, skip_keywords)

        optimizer = optim.AdamW(parameters, eps=1e-8, betas=(0.9, 0.999),
                            lr=1.25e-4, weight_decay=0.05)  # 1.25e-4
    elif _optim.lower() == 'attcg':
        optimizer = ATTCG(model.parameters(), lr=1.25e-4) 
    return optimizer


def set_weight_decay(model, skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
