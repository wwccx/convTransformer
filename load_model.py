from convTrans import convTransformer
from compare_networks import ResNet
from torch import nn
from gqcnn import GQCNN 

def build_model(config):
    try:
        x = config.MODEL.DYNAMIC
    except AttributeError:
        config.defrost()
        config.MODEL.DYNAMIC = False
        print('Loading Model, config has not DYNAMIC, set it to False')
        config.freeze()
    if config.MODEL.ARCH == 'convTrans':
        model = convTransformer(
            in_chans=config.MODEL.IN_CHANNELS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.EMBED_DIM,
            depths=config.MODEL.DEPTHS,
            num_heads=config.MODEL.NUM_HEADS,
            patch_embedding_size=config.MODEL.PATCH_EMBED_SIZE,
            patch_merging_size=config.MODEL.PATCH_MERGE_SIZE,
            window_size=config.MODEL.WINDOW_SIZE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            fully_conv_for_grasp=config.MODEL.FULLY_CONV_FOR_GRASP,
            norm_layer=nn.BatchNorm2d if config.MODEL.NORM_LAYER == 'BN' else nn.LayerNorm,
            dynamic=config.MODEL.DYNAMIC,
        )
    elif config.MODEL.ARCH == 'res':
        model = ResNet([2, 2, 2, 2], inChannel=config.MODEL.IN_CHANNELS, dynamic=config.MODEL.DYNAMIC)
    
    elif config.MODEL.ARCH == 'gqcnn':
        model = GQCNN(inChannel=config.MODEL.IN_CHANNELS, dynamic=config.MODEL.DYNAMIC)

    else:
        raise NotImplementedError(f"Unkown model: {config.MODEL.ARCH}")
    return model
    # if 'convTrans' in opt.model:
    #     if 'vib' in opt.dataset:
    #         self.network = convTransformer(num_classes=2, in_chans=1, window_size=(1, 3),
    #                                        patch_embedding_size=(1, 32), patch_merging_size=(1, 2),
    #                                        ).to(self.device)
    #     elif 'grasp' in opt.dataset:
    #         self.network = convTransformer(in_chans=1, num_classes=32,
    #                                        embed_dim=96, depths=(2, 6),
    #                                        num_heads=(3, 12),
    #                                        patch_embedding_size=(4, 4),
    #                                        fully_conv_for_grasp=True).to(self.device)
    #
    #     else:


    return model
