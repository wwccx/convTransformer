import numpy as np
import torch
from convTrans import convTransformer
import torch.utils.data as D
import datetime
import os
import logging
import time
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as dataTransforms
from optimizer import build_optimizer
import argparse
logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=60, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
opt = parser.parse_args()


class gqTrain:
    def __init__(self, dataDir='', saveDir='./train'):

        self.device = torch.device("cuda:0")

        self.dataDir = dataDir
        self.saveDir = os.path.join(saveDir,
                                    datetime.datetime.now().strftime('convTrans%y_%m_%d_%H:%M'))
        logging.info('save path:'+self.saveDir)

        os.makedirs(self.saveDir, exist_ok=True)

        self.trainDataLoader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data/CIFAR10',
                             train=True,
                             download=True,
                             transform=dataTransforms.Compose(
                                 [dataTransforms.Resize((224, 224)),
                                  dataTransforms.ToTensor(),
                                  dataTransforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                             )
                             ),
            batch_size=opt.batch_size,
            shuffle=True
        )
        self.valDataLoader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data/CIFAR10',
                             train=False,
                             download=True,
                             transform=dataTransforms.Compose(
                                 [dataTransforms.Resize((224, 224)),
                                  dataTransforms.ToTensor(),
                                  dataTransforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                             )
                             ),
            batch_size=opt.batch_size,
            shuffle=True
        )

        logging.info('data set loaded')
        self.network = convTransformer().to(self.device)
        self.optimizer = build_optimizer(self.network)
        self.currentEpoch = 0
        self.loss_value = np.array(0)
        self.acc_value = np.array(0)
        
        self.lossFun = torch.nn.CrossEntropyLoss()

    def train(self, log_frequency=100):
        self.network.train()
        batchIdx = 0
        t0 = time.time()
        for img, target in self.trainDataLoader:
            target = target.to(self.device)
            img = img.to(self.device)
            target_pre = self.network(img)

            loss = self.lossFun(target_pre, target)
            
            # self.loss_value = np.append(self.loss_value, loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batchIdx % log_frequency == 0:
                self.loss_value = np.append(self.loss_value, loss.item())
                logging.info('Epoch:{}--{:.4f}%, loss:{:.6e}, batch time:{:.3f}s, epoch remain:{:.2f}min'
                             .format(self.currentEpoch, 100*batchIdx/len(self.trainDataLoader),
                                     loss.item(),
                                     (time.time()-t0)/log_frequency,
                                     (time.time()-t0)/log_frequency*(len(self.trainDataLoader) - batchIdx)/60
                                     )
                             )
                t0 = time.time()
            batchIdx += 1

    def validate(self):
        self.network.eval()
        accuracy = 0
        valBatchIdx = 0
        success_pre = torch.tensor(0.1).cuda()
        total_pre = torch.tensor(0.1).cuda()

        for img, target in self.valDataLoader:
            target = target.to(self.device)
            img = img.to(self.device)
            target_pre = self.network(img)
            target_pre = torch.argmax(target_pre, dim=1)
            judge_tensor = (target_pre == target)
            total_pre += len(target)
            success_pre += torch.sum(judge_tensor)

            if valBatchIdx % 20 == 0:
                logging.info('evalutating:{:.2f}%, success pre{:.3f}%'
                             .format(100*valBatchIdx/len(self.valDataLoader),
                                     success_pre/total_pre)
                             )
            valBatchIdx += 1

        return success_pre/total_pre

    def save(self, epoch, accuracy):
        dt = datetime.datetime.now().strftime('%H%M')
        save_path = os.path.join(self.saveDir, dt + 'convTrans_epoch{}_acc{:.4f}.pth'.format(epoch, accuracy))
        torch.save(self.network.state_dict(), save_path)
        logging.info('save to '+save_path)

    def run(self, epoch=60, start=0):
        loss_value = np.array(0)
        acc_value = np.array(0)

        for i in range(start, epoch):
            self.currentEpoch = i
            self.train()
            if i % 1 == 0:
                accuracy = self.validate()
                self.acc_value = np.append(self.acc_value, accuracy.cpu().detach().numpy())
                self.network.train()
                self.save(self.currentEpoch, accuracy)
                # self.lrScheduler.step()
                for p in self.optimizer.param_groups:
                    logging.info('current lr:{}'.format(p['lr']))
                np.save(os.path.join(self.saveDir, 'loss_value.npy'), self.loss_value)
                np.save(os.path.join(self.saveDir, 'acc_value.npy'), self.acc_value)
            self.currentEpoch += 1


if __name__ == '__main__':
    gqTrain = gqTrain()
    # gqTrain.network.load_state_dict(
    #     torch.load('/home/server/grasp/6t9only_attcg_shuffle_gqcnn22_02_23_20:12/19116t9only_simp_attcg_epoch34_pacc0.7045_nacc0.9925.pth')
    # )
    gqTrain.run(epoch=opt.n_epochs)






