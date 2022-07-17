from torchvision import datasets
import torch.utils.data as D
import torchvision.transforms as dataTransforms
import os
import torch


def build_dataset(config, data_path='./data', transform=None):
    name = config.DATA.DATASET
    batch_size = config.DATA.BATCH_SIZE
    dataset_dict = {
        'root': data_path,
        'train': True,
        'download': True}

    if transform is None:
        dataset_dict['transform'] = dataTransforms.Compose(
            [dataTransforms.Resize((224, 224)),
             dataTransforms.ToTensor(),
             dataTransforms.Normalize(mean=[0.5], std=[0.5])]
        )

    else:
        dataset_dict['transform'] = transform

    if 'cifar' in name.lower():
        t = D.DataLoader(
            datasets.CIFAR10(**dataset_dict),
            batch_size=batch_size,
            shuffle=True
        )
        dataset_dict['train'] = False
        v = D.DataLoader(
            datasets.CIFAR10(**dataset_dict),
            batch_size=batch_size,
            shuffle=True
        )
        return t, v

    elif 'minst' in name.lower():
        t = D.DataLoader(
            datasets.CIFAR10(**dataset_dict),
            batch_size=batch_size,
            shuffle=True
        )
        dataset_dict['train'] = False
        v = D.DataLoader(
            datasets.CIFAR10(**dataset_dict),
            batch_size=batch_size,
            shuffle=True
        )
        return t, v
    elif 'imagenet' in name.lower():
        dataset_dict['root'] = './data/imagenet/ILSVRC/Data/CLS-LOC'
        train_data =datasets.ImageFolder(
                root=os.path.abspath(dataset_dict['root'])+'/train',
                transform=dataset_dict['transform'])
        t = D.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                num_workers=12)
        val_data = datasets.ImageFolder(
                root=os.path.abspath(dataset_dict['root'])+'/val',
                transform=dataset_dict['transform'])
        v = D.DataLoader(val_data, batch_size=batch_size, shuffle=True,
                num_workers=12)
        return t, v
    elif 'vibration' in name.lower():
        from viDataloader import VibrationDataset
        seed = 42
        torch.manual_seed(seed)  # 设置 cpu 的随机数种子
        torch.cuda.manual_seed(seed)  # 对于单张显卡，设置 gpu 的随机数种子
        # negative_channels = torch.randperm(301)
        negative_channels = torch.arange(301)
        positive_channels = torch.randperm(140)
        train_data = VibrationDataset((positive_channels[0:119], negative_channels[31:]))
        val_data = VibrationDataset((positive_channels[119:], negative_channels[:31]))
        t = D.DataLoader(batch_size=batch_size, dataset=train_data, num_workers=12)
        v = D.DataLoader(batch_size=batch_size, dataset=val_data, num_workers=12)
        return t, v
    elif 'grasp' in name.lower():
        from graspDataset import MixupGraspDataset as GraspDataset
        # tdataset = GraspDataset('/home/server/grasp1/virtual_grasp/fine_tune')
        # vdataset = GraspDataset('/home/server/grasp1/virtual_grasp/fine_tune', pattern='validation')
        tdataset = GraspDataset(data_path, batch_size=64, add_noise=config.DATA.NOISE)
        vdataset = GraspDataset(data_path, pattern='val', batch_size=8, add_noise=config.DATA.NOISE)
        # print(batch_size)
        t = D.DataLoader(tdataset, batch_size=batch_size // 64, shuffle=True, num_workers=12)
        v = D.DataLoader(vdataset, batch_size=batch_size // 8, shuffle=True, num_workers=12)
        return t, v

    else:
        raise NotImplementedError('Only support MINST and CIFAR10')


