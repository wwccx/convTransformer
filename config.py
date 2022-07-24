import os
import yaml
from yacs.config import CfgNode as CN
from torch import nn

_C = CN()

_C.DATA = CN()
_C.DATA.BATCH_SIZE = 512
_C.DATA.DATASET = 'grasp'
_C.DATA.MIXUP_ON = False
_C.DATA.IMG_SIZE = (1, 96, 96)
_C.DATA.NOISE = True
_C.DATA.LOSS_WEIGHT = [1., 1.]

_C.DATA.MIXUP = CN()
_C.DATA.MIXUP.MIXUP_ALPHA = 0.8
_C.DATA.MIXUP.CUTMIX_ALPHA = 1.0
_C.DATA.MIXUP.CUTMIX_MINMAX = None
_C.DATA.MIXUP.PROB = 1.0
_C.DATA.MIXUP.SWITCH_PROB = 0.5
_C.DATA.MIXUP.MODE = 'batch'
_C.DATA.MIXUP.LABEL_SMOOTHING = 0.1
_C.DATA.MIXUP.NUM_CLASSES = 2

_C.MODEL = CN()
_C.MODEL.ARCH = 'convTrans'
_C.MODEL.NUM_CLASSES = 32
_C.MODEL.IN_CHANNELS = 1
_C.MODEL.EMBED_DIM = 128
_C.MODEL.PATCH_EMBED_SIZE = (8, 8)
_C.MODEL.PATCH_MERGE_SIZE = (2, 2)
_C.MODEL.FULLY_CONV_FOR_GRASP = True
_C.MODEL.WINDOW_SIZE = (3, 3)
_C.MODEL.NUM_HEADS = (8, )
_C.MODEL.DEPTHS = (8, )
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.NORM_LAYER = nn.BatchNorm2d

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 100
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.MIN_LR = 5e-6
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.AMP_LEVEL = 'O1'
_C.TRAIN.CLIP_GRAD = 0

_C.TRAIN.LR_SCHEDULE = CN()
_C.TRAIN.LR_SCHEDULE.TYPE = 'cosine'

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = 'adamw'
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 0.05
_C.TRAIN.OPTIMIZER.BETA = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.EPS = 1e-8


def update_config(opt):
    cfg = _C.clone()
    cfg.TRAIN.EPOCHS = opt.n_epochs
    cfg.DATA.BATCH_SIZE = opt.batch_size
    cfg.MODEL.ARCH = opt.model
    cfg.DATA.DATASET = opt.dataset
    cfg.TRAIN.AMP_LEVEL = opt.amp_level
    cfg.DATA.MIXUP_ON = opt.mixup
    cfg.freeze()
    return cfg
