from torchvision import datasets
import torch.utils.data as D
import torchvision.transforms as dataTransforms
import csv
import os
import shutil
dataset_dict = {
         'root': './data/imagenet/ILSVRC/Data/CLS-LOC/train',
         #'train': True,
         }
dataset_dict['transform'] = dataTransforms.Compose(
         [dataTransforms.Resize((224, 224)),
          dataTransforms.ToTensor(),
          dataTransforms.Normalize(mean=[0.5], std=[0.5])]
         )

# d = datasets.ImageFolder(**dataset_dict)
# print(d.class_to_idx)
# loader = D.DataLoader(d, batch_size=1)
path = '/home/server/convTransformer/data/imagenet/ILSVRC/Data/CLS-LOC'
save_path = os.path.join(path, 'val_sort')
os.makedirs(save_path, exist_ok=True)

with open('/home/server/convTransformer/data/imagenet/LOC_val_solution.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    
    for row in spamreader:
        if 'ImageId' == row[0]:
            continue
        # print(row, type(row))
        # print(row[0], row[1])
        # continue
        file_name = row[0] + '.JPEG'
        dict_name = row[1][0:9]
        # print(file_name, dict_name)
        os.makedirs(os.path.join(save_path, dict_name), exist_ok=True)
        p = shutil.copy(os.path.join(path, 'val', file_name),
                os.path.join(save_path, dict_name, file_name)
                )
        print(p)
        

