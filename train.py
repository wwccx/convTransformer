import numpy as np
import torch
from convTrans import convTransformer
import datetime
import os
import logging
import time
from optimizer import build_optimizer
import argparse
from torchsummary import summary
from torchvision import models as models
from load_dataset import build_dataset
from lr_scheduler import build_scheduler
from swin_transformer import SwinTransformer
from optimizer_swin import build_optimizer as swin_optim
logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=350, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--log_frequency", type=int, default=10, help="log info per batches")
parser.add_argument("--model", type=str, default='convTrans', help="the trained model")
parser.add_argument("--dataset", type=str, default='CIFAR', help='the dataset')
parser.add_argument("--check_point", type=str, default='')
opt = parser.parse_args()


class gqTrain:
    def __init__(self, dataDir='', saveDir='./train', check_point=opt.check_point):

        self.device = torch.device("cuda:0")

        self.dataDir = dataDir

        self.trainDataLoader, self.valDataLoader = build_dataset(
            name=opt.dataset,
            batch_size=opt.batch_size,
            data_path='./data'
        )

        logging.info('data set loaded')
        if 'convTrans' in opt.model:
            self.network = convTransformer(B=opt.batch_size).to(self.device)
            self.optimizer = build_optimizer(self.network)
        else:
            # self.network = models.resnet50(num_classes=10).to(self.device)
            self.network = SwinTransformer(num_classer=10).to(self.device)
            self.optimizer = swin_optim(self.network)
        summary(self.network, (3, 224, 224), batch_size=opt.batch_size)
        # self.optimizer = build_optimizer(self.network)
        self.num_step_per_epoch = len(self.trainDataLoader)
        self.lr_scheduler = build_scheduler(opt, optimier=self.optimizer, n_iter_per_epoch=self.num_step_per_epoch)
        self.currentEpoch = 0
        self.loss_value = np.array(0)
        self.acc_value = np.array(0)
        self.maxAcc = 0
        self.lossFun = torch.nn.CrossEntropyLoss()

        if check_point:
            self.saveDir = check_point
            logging.info('resuming path:' + self.saveDir)
            self.load_check_point(check_point)
        else:
            self.saveDir = os.path.join(saveDir,
                                        datetime.datetime.now().strftime(opt.model + '%y_%m_%d_%H:%M'))
            logging.info('save path:'+self.saveDir)
            os.makedirs(self.saveDir, exist_ok=True)

    def train(self, log_frequency=opt.log_frequency):
        self.network.train()
        batchIdx = 0
        t0 = time.time()
        for img, target in self.trainDataLoader:
            target = target.to(self.device)
            img = img.to(self.device)
            target_pre = self.network(img)

            loss = self.lossFun(target_pre, target)
            self.loss_value = np.append(self.loss_value, loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (batchIdx + 1) % log_frequency == 0:
                self.loss_value = np.append(self.loss_value, loss.item())
                logging.info('Epoch:{}--{:.4f}%, loss:{:.6e}, batch time:{:.3f}s, epoch remain:{:.2f}min'
                             .format(self.currentEpoch, 100*batchIdx/self.num_step_per_epoch,
                                     loss.item(),
                                     (time.time()-t0)/log_frequency,
                                     (time.time()-t0)/log_frequency*(self.num_step_per_epoch - batchIdx)/60
                                     )
                             )
                t0 = time.time()
                np.save(os.path.join(self.saveDir, 'loss_value.npy'), self.loss_value)
            batchIdx += 1
            self.lr_scheduler.step_update(self.currentEpoch * self.num_step_per_epoch + batchIdx)

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
            judge_tensor = torch.tensor(target_pre == target)
            total_pre += len(target)
            success_pre += torch.sum(judge_tensor)

            if (valBatchIdx + 1) % opt.log_frequency == 0:
                logging.info('evalutating:{:.2f}%, success pre{:.3f}%'
                             .format(100*valBatchIdx/len(self.valDataLoader),
                                     100*success_pre/total_pre)
                             )
            valBatchIdx += 1

        return success_pre/total_pre

    def save(self, epoch, accuracy):
        dt = datetime.datetime.now().strftime('%H%M')
        save_state = {
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': self.currentEpoch,
            'accuracy': accuracy
        }
        save_path = os.path.join(self.saveDir,
                                 opt.model + opt.dataset + dt + 'state_epoch{}_acc{:.4f}.pth'.format(epoch, accuracy))
        torch.save(save_state, save_path)
        logging.info('save to ' + save_path)

    def run(self, epoch=300):

        for i in range(self.currentEpoch, epoch):
            self.train()
            if i % 5 == 0:
                accuracy = self.validate()
                if accuracy > self.maxAcc or (i + 1) % 10 == 0:
                    self.maxAcc = max(accuracy, self.maxAcc)
                    self.save(self.currentEpoch, accuracy)
                self.acc_value = np.append(self.acc_value, accuracy.cpu().detach().numpy())
                self.network.train()
                for p in self.optimizer.param_groups:
                    logging.info('current lr:{}'.format(p['lr']))
                np.save(os.path.join(self.saveDir, 'acc_value.npy'), self.acc_value)
            self.currentEpoch += 1

    def load_check_point(self, save_path):
        logging.info('searching in ' + save_path)
        check_points = os.listdir(save_path)
        check_points = [ckpt for ckpt in check_points if ckpt.endswith('.pth')]
        if len(check_points) > 0:
            latest_ckpt = max([os.path.join(save_path, d) for d in check_points], key=os.path.getmtime)
            logging.info('resuming from ' + latest_ckpt)
        else:
            latest_ckpt = None
        if latest_ckpt:
            check_point = torch.load(latest_ckpt)
            self.network.load_state_dict(check_point['model'])
            self.optimizer.load_state_dict(check_point['optimizer'])
            self.lr_scheduler.load_state_dict(check_point['lr_scheduler'])
            self.currentEpoch = check_point['epoch'] + 1
            self.maxAcc = max(self.maxAcc, check_point['accuracy'])

            self.acc_value = np.load(os.path.join(save_path, 'acc_value.npy'))
            self.loss_value = np.load(os.path.join(save_path, 'loss_value.npy'))
        else:
            raise FileNotFoundError('No check points in the path!')

    def test_train(self):
        log_frequency = 10
        self.network.train()
        batchIdx = 0
        t0 = time.time()

        for img, target in self.trainDataLoader:
            # with torchprof.Profile(self.network, use_cuda=True) as prof:
            #     target = target.to(self.device)
            #     img = img.to(self.device)
            #     t = time.time()
            #     target_pre = self.network(img)
            #     # torch.cuda.synchronize()
            #     print(time.time() - t)
            #     loss = self.lossFun(target_pre, target)
            #     # self.loss_value = np.append(self.loss_value, loss.item())
            #     self.optimizer.zero_grad()
            #     loss.backward()
            #     self.optimizer.step()
            # print(prof.display(show_events=True))
            # break
            with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
                target = target.to(self.device)
                img = img.to(self.device)
                t = time.time()
                target_pre = self.network(img)
                torch.cuda.synchronize()
                print(time.time() - t)

                loss = self.lossFun(target_pre, target)
                # self.loss_value = np.append(self.loss_value, loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
            break


if __name__ == '__main__':
    gqTrain = gqTrain()
    gqTrain.run(epoch=opt.n_epochs)
    # gqTrain.test_train()





