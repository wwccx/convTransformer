from torchvision import datasets
import torch.utils.data as D
import torchvision.transforms as dataTransforms
import os

def build_dataset(name, batch_size, data_path, transform=None):
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

    else:
        raise NotImplementedError('Only support MINST and CIFAR10')


