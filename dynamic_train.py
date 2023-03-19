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
from load_model import build_model
from swin_transformer import SwinTransformer
from optimizer_swin import build_optimizer as swin_optim
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from apex import amp
from virtual_grasp.virtualGraspDataset import VirtualGraspDataset
from torchmetrics import AveragePrecision as AP
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from config import update_config

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--log_frequency", type=int, default=10, help="log info per batches")
parser.add_argument("--model", type=str, default='convTrans', help="the trained model")
parser.add_argument("--dataset", type=str, default='imagenet', help='the dataset')
parser.add_argument("--check_point", type=str, default='')
parser.add_argument("--optim", type=str, default='adamw')
parser.add_argument('--amp_level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument("--mixup", type=bool, default=False)
parser.add_argument("--finetune", type=str, default='')
parser.add_argument("--tag", type=str, default='')
opt = parser.parse_args()
config = update_config(opt)
logging.info(config)


class gqTrain:
    def __init__(self, dataDir='', saveDir='./train', check_point=opt.check_point):

        self.device = torch.device("cuda:0")

        self.dataDir = dataDir

        self.trainDataLoader, self.valDataLoader = build_dataset(config)

        logging.info('data set loaded')
        self.network = build_model(config).to(self.device)
        logging.info('model loaded')
        self.optimizer = build_optimizer(config, self.network)

        if opt.amp_level != 'O0':
            self.network, self.optimizer = amp.initialize(self.network, self.optimizer, opt_level=opt.amp_level)

        summary(self.network, (config.MODEL.IN_CHANNELS, 96, 96), batch_size=opt.batch_size)
        # self.optimizer = build_optimizer(self.network)

        self.currentEpoch = 0
        self.loss_value = np.array([])
        self.acc_value = np.array([])
        self.lr = np.array([])
        self.grad = np.array([])
        self.finetune_acc = np.array([])
        self.maxAcc = 0
        if not config.DATA.MIXUP_ON:
            if config.DATA.DATASET == 'grasp':
                from graspDataset import GraspLossFunction
                self.lossFun = GraspLossFunction(config)
            elif config.DATA.DATASET == 'dynamic':
                from DynamicGraspDataset import DynamicGraspLossFunction
                self.lossFun = DynamicGraspLossFunction(config)
            else:
                self.lossFun = torch.nn.CrossEntropyLoss()
        else:
            if config.DATA.DATASET == 'grasp':
                from graspDataset import GraspLossFunction
                self.lossFun = GraspLossFunction(SoftTargetCrossEntropy())
            elif config.DATA.DATASET == 'dynamic':
                from DynamicGraspDataset import DynamicGraspLossFunction
                self.lossFun = DynamicGraspLossFunction(SoftTargetCrossEntropy())
            else:
                self.lossFun = SoftTargetCrossEntropy()

        self.gradNormVal = config.TRAIN.CLIP_GRAD
        self.mixup = Mixup(mixup_alpha=config.DATA.MIXUP.MIXUP_ALPHA,
                           cutmix_alpha=config.DATA.MIXUP.CUTMIX_ALPHA,
                           cutmix_minmax=config.DATA.MIXUP.CUTMIX_MINMAX,
                           prob=config.DATA.MIXUP.PROB,
                           switch_prob=config.DATA.MIXUP.SWITCH_PROB,
                           mode=config.DATA.MIXUP.MODE,
                           label_smoothing=config.DATA.MIXUP.LABEL_SMOOTHING,
                           num_classes=config.DATA.MIXUP.NUM_CLASSES
                           )
        self.saveDir = ''
        if opt.finetune != '' and opt.finetune.endswith('.pth'):
            self.network.load_state_dict(
                torch.load(opt.finetune)['model']
            )
            # self.trainDataLoader = VirtualGraspDataset(self.network)
            if check_point:
                self.saveDir = check_point
            else:
                self.saveDir = os.path.join(saveDir,
                                            'finetune' + datetime.datetime.now().strftime(opt.model + '%y_%m_%d_%H_%M' + '_' + opt.tag))
        if self.saveDir == '':
            self.saveDir = os.path.join(saveDir,
                    datetime.datetime.now().strftime(opt.model + '%y_%m_%d_%H_%M' + '_' + opt.tag))
        os.makedirs(self.saveDir, exist_ok=True)
        torch.save({'config': config}, os.path.join(self.saveDir, 'config.pth'))
        self.num_step_per_epoch = len(self.trainDataLoader)
        self.lr_scheduler = build_scheduler(config, optimier=self.optimizer, n_iter_per_epoch=self.num_step_per_epoch)

        if check_point:
            self.saveDir = check_point
            logging.info('resuming path:' + self.saveDir)
            self.load_check_point(check_point)

        logging.info('save path:' + self.saveDir)

    def train(self, log_frequency=opt.log_frequency, p=None):
        self.network.train()
        batchIdx = 0
        avg_loss = 0
        batch_time = 0
        avg_grad = 0
        t0 = time.time()
        tid = p.add_task(f'Epoch{self.currentEpoch}', loss=0, avg_loss=0,
                         batch_time=0, lr=0, avg_grad=0)
        p.update(tid, total=self.num_step_per_epoch)

        # for img, target in self.trainDataLoader:
        for img, pose, target, mask, target_pos in self.trainDataLoader:
            target = target.to(self.device).flatten(0, 1)
            img = img.to(self.device).flatten(0, 1)
            mask = mask.to(self.device).flatten(0, 1)
            pose = pose.to(self.device).flatten(0, 1)
            target_pos = target_pos.to(self.device).flatten(0, 1)
            if opt.mixup:
                img, target = self.mixup(img, target)
            # _input = [img, pose]
            target_class_pre, target_pos_pre = self.network(*[img, pose])
            # print(target_pre.shape)
            # print(target.shape)
            loss = self.lossFun(target_class_pre, target_pos_pre, target, mask, target_pos)
            self.loss_value = np.append(self.loss_value, loss.item())
            self.lr = np.append(self.lr, self.optimizer.param_groups[0]['lr'])

            self.optimizer.zero_grad()
            if opt.amp_level != "O0":
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if self.gradNormVal:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradNormVal)
                self.grad = np.append(self.grad, grad_norm.item())
                avg_grad = (batchIdx * avg_grad + grad_norm.item()) / (batchIdx + 1)
            else:
                grad_norm = self.get_grad_norm(self.network.parameters())
                if not np.isinf(grad_norm):
                    self.grad = np.append(self.grad, grad_norm)
                    avg_grad = (batchIdx * avg_grad + grad_norm) / (batchIdx + 1)
            self.optimizer.step()
            avg_loss = (batchIdx * avg_loss + loss.item()) / (batchIdx + 1)
            if (batchIdx + 1) % log_frequency == 0:
                batch_time = (time.time() - t0) / log_frequency
                t0 = time.time()
                np.save(os.path.join(self.saveDir, 'loss_value.npy'),
                        self.loss_value)
                np.save(os.path.join(self.saveDir, 'lr_value.npy'),
                        self.lr)
                np.save(os.path.join(self.saveDir, 'grad_value.npy'),
                        self.grad)
            batchIdx += 1
            p.update(tid, advance=1, loss=loss.item(), avg_loss=avg_loss,
                     batch_time=batch_time, lr=self.lr[-1], avg_grad=avg_grad)
            self.lr_scheduler.step_update(self.currentEpoch * self.num_step_per_epoch + batchIdx)

    @torch.no_grad()
    def validate(self, p=None):
        self.network.eval()
        accuracy = 0
        valBatchIdx = 0
        success_pre = torch.tensor(0.1).cuda()
        total_pre = torch.tensor(0.1).cuda()
        tid = p.add_task(f'Validation:{self.currentEpoch}', accuracy=0,
                         positive_accuracy=0, negative_accuracy=0, loss=0, rloss=0)
        p.update(tid, total=len(self.valDataLoader))
        positive_pre = torch.tensor(0.1).cuda()
        negative_pre = torch.tensor(0.1).cuda()
        total_positive_pre = torch.tensor(0.1).cuda()
        total_negative_pre = torch.tensor(0.1).cuda()
        average_loss = 0
        average_rloss = 0
        ap = AP()

        for img, pose, target, mask, target_pos in self.valDataLoader:
            target = target.to(self.device).flatten(0, 1)
            img = img.to(self.device).flatten(0, 1)
            mask = mask.to(self.device).flatten(0, 1)
            pose = pose.to(self.device).flatten(0, 1)
            target_pos = target_pos.to(self.device).flatten(0, 1)
            target_class_pre, target_pos_pre = self.network(*[img, pose])
            loss = self.lossFun(target_class_pre, target_pos_pre, target, mask, target_pos)
            loss_pos = torch.nn.functional.l1_loss(target_pos_pre.squeeze(), target_pos)
            average_loss = (valBatchIdx * average_loss + loss.item()) / (valBatchIdx + 1)
            average_rloss = (valBatchIdx * average_rloss + loss_pos.item()) / (valBatchIdx + 1)

            target_class_pre = target_class_pre.squeeze()[torch.where(mask > 0)].view(-1, 2)
            target_class_pre = torch.nn.functional.softmax(target_class_pre, dim=1)
            ap.update(target_class_pre[:, 1], target)
            target_class_pre = torch.argmax(target_class_pre, dim=1)
            # print(target_pre.shape, target.shape)
            # print(target_pre, target)
            judge_tensor = (target_class_pre == target)
            total_pre += len(target)
            success_pre += torch.sum(judge_tensor)
            # print(target_pre)
            positive_judge = judge_tensor[torch.where(target > 0)]
            total_positive_pre += len(positive_judge)
            positive_pre += torch.sum(positive_judge)

            negative_judge = judge_tensor[torch.where(target == 0)]
            total_negative_pre += len(negative_judge)
            negative_pre += torch.sum(negative_judge)

            # if (valBatchIdx + 1) % opt.log_frequency == 0:
            #     logging.info('evalutating:{:.2f}%, success pre{:.3f}%'
            #                  .format(100*valBatchIdx/len(self.valDataLoader),
            #                          100*success_pre/total_pre)
            #                  )
            valBatchIdx += 1
            p.update(tid, advance=1, accuracy=100 * success_pre / total_pre,
                     loss=average_loss, rloss=average_rloss,
                     positive_accuracy=100 * positive_pre / total_positive_pre,
                     negative_accuracy=100 * negative_pre / total_negative_pre)
        # print(success_pre, total_pre, positive_pre, total_positive_pre,
        #         negative_pre, total_negative_pre)
        map = ap.compute()
        return map

    def save(self, epoch, accuracy):
        dt = datetime.datetime.now().strftime('%H%M')
        save_state = {
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': self.currentEpoch,
            'accuracy': accuracy,
            'loss_hist': self.loss_value,
            'accuracy_hist': self.acc_value,
            'lr_hist': self.lr,
            'grid_hist': self.grad
        }
        if accuracy < 0:
            save_path = os.path.join(self.saveDir, 'temp_epoch_train.pth')
            torch.save(save_state, save_path)
            return
        save_path = os.path.join(self.saveDir,
                                 opt.model + opt.dataset + 'state_epoch{}_acc{:.4f}.pth'.format(epoch, accuracy))
        torch.save(save_state, save_path)
        logging.info('save to ' + save_path)

    def run(self, epoch=300):

        for i in range(self.currentEpoch, epoch):
            p = self.make_progress('train')
            with p as x:

                # with torch.autograd.set_detect_anomaly(True):
                self.train(p=x)
                # pass
            if i % 1 == 0:
                p = self.make_progress('val')
                with p:
                    accuracy = self.validate(p=p)
                # if accuracy > self.maxAcc or (i + 1) % 10 == 0:
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
            # print(self.optimizer.defaults)
            self.lr_scheduler.load_state_dict(check_point['lr_scheduler'])
            self.currentEpoch = check_point['epoch'] + 1
            self.maxAcc = max(self.maxAcc, check_point['accuracy'])
            try:
                self.acc_value = check_point['accuracy_hist']
                self.loss_value = check_point['loss_hist']
                self.lr = check_point['lr_hist']
                self.grad = check_point['grid_hist']
                self.finetune_acc = np.load(os.path.join(save_path, 'acc_value.npy'))
                # self.acc_value = check_point['acc']
                # self.loss_value = check_point['loss']
                # self.lr = check_point['lr']
                # self.grad = check_point['grid']
            except:
                # self.acc_value = np.load(os.path.join(save_path, 'acc_value.npy'))
                # self.loss_value = np.load(os.path.join(save_path, 'loss_value.npy'))
                # self.lr = np.load(os.path.join(save_path, 'lr_value.npy'))
                self.grad = np.array([])
                self.finetune_acc = np.array([])
                pass
        else:
            raise FileNotFoundError('No check points in the path!')

        # p = self.make_progress('val')
        # with p:
        #     accuracy = self.validate(p=p)

    def test_train(self):
        log_frequency = 10
        self.network.train()
        batchIdx = 0
        t0 = time.time()

        for img, target in self.trainDataLoader:
            # with torchprof.Profile(self.network, use_cuda=True) as prof:
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
            # print(prof.display(show_events=True))
            # break
            batchIdx += 1
            if batchIdx > 10:
                with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True) as prof:
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
                print(prof.key_averages().table(sort_by="self_cuda_time_total"))
                break

    @staticmethod
    def make_progress(partten='train'):
        if partten == 'train':

            progress = Progress(
                TextColumn("[bold blue]{task.description}", justify="right"),
                "[progress.percentage]{task.percentage:>.3f}%",
                BarColumn(bar_width=None),
                "•",
                TextColumn("lr:{task.fields[lr]:>5.4e}"),
                "•",
                TextColumn("L:{task.fields[loss]:>6.5e}"),
                "•",
                TextColumn("avgL:{task.fields[avg_loss]:>6.5e}"),
                "•",
                TextColumn("Grad:{task.fields[avg_grad]:>5.4e}"),
                "•",
                TextColumn("bTime:{task.fields[batch_time]:>.3f}s"),
                "•",
                TimeRemainingColumn(),
            )
        elif partten == 'finetune':
            progress = Progress(
                TextColumn("[bold blue]{task.description}", justify="right"),
                "[progress.percentage]{task.percentage:>.3f}%",
                BarColumn(bar_width=None),
                "•",
                TextColumn("lr:{task.fields[lr]:>5.4e}"),
                "•",
                TextColumn("L:{task.fields[loss]:>6.5e}"),
                "•",
                TextColumn("avgL:{task.fields[avg_loss]:>6.5e}"),
                "•",
                TextColumn("avgAcc:{task.fields[avg_acc]:>5.4e}"),
                "•",
                TextColumn("Grad:{task.fields[avg_grad]:>5.4e}"),
                "•",
                TextColumn("bTime:{task.fields[batch_time]:>.3f}s"),
                "•",
                TimeRemainingColumn(),
            )
        else:
            progress = Progress(
                TextColumn("[yellow]{task.description}", justify="right"),
                "[progress.percentage]{task.percentage:>.3f}%",
                BarColumn(bar_width=None),
                "•",
                TextColumn("Loss:{task.fields[loss]:>.3f}"),
                "•",
                TextColumn("RLoss:{task.fields[rloss]:>.4e}"),
                "•",
                TextColumn("Accuracy:{task.fields[accuracy]:>.2f}"),
                "•",
                TextColumn("positiveAccuracy:{task.fields[positive_accuracy]:>.2f}"),
                "•",
                TextColumn("negativeAccuracy:{task.fields[negative_accuracy]:>.2f}"),
                "•",
                TimeRemainingColumn(),
            )

        return progress

    @staticmethod
    def get_grad_norm(parameters, norm_type=2):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        norm_type = float(norm_type)
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type if not torch.isnan(param_norm) else 0
        total_norm = total_norm ** (1. / norm_type)
        return total_norm


if __name__ == '__main__':
    gqTrain = gqTrain()
    gqTrain.run(epoch=opt.n_epochs)

    #  gqTrain.test_train()
