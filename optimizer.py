from torch import optim as optim
from ATTCG import ATTCG


def build_optimizer(config, model):
    _optim = config.TRAIN.OPTIMIZER.TYPE
    if _optim.lower() == 'adamw':
        skip_keywords = {}
        if hasattr(model, 'no_weight_decay_keywords'):
            skip_keywords = model.no_weight_decay_keywords()
        parameters = set_weight_decay(model, skip_keywords)

        optimizer = optim.AdamW(parameters,
                                eps=config.TRAIN.OPTIMIZER.EPS,
                                betas=config.TRAIN.OPTIMIZER.BETA,
                                lr=config.TRAIN.BASE_LR,
                                weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)  # 1.25e-4

    elif _optim.lower() == 'attcg':
        skip_keywords = {}
        if hasattr(model, 'no_weight_decay_keywords'):
            skip_keywords = model.no_weight_decay_keywords()
        parameters = set_weight_decay(model, skip_keywords)
        
        optimizer = ATTCG(parameters, lr=config.TRAIN.BASE_LR,
                          weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    else:
        raise NotImplementedError(f"Optimizer {_optim} is not implemented")

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
