from torchvision import datasets
import torch.utils.data as D
import torchvision.transforms as dataTransforms


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

    if 'CIFAR' in name:
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

    elif 'MINST' in name:
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
    else:
        raise NotImplementedError('Only support MINST and CIFAR10')