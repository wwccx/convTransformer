from numpy import zeros_like
import torch.nn as nn
import torch.nn.functional as F
# from graspDataLoader import GraspDataset
import torch
'''

'''

class GQCNN(nn.Module):
    def __init__(self, inChannel, dynamic):
        super(GQCNN, self).__init__()
        self.inChannel = inChannel
        self.layers = self.structure()
        # self.lossWeight = torch.ones(32).cuda()
        # self.lossWeight[1::2] = 1200
        self.sf = nn.Softmax(dim=1)
        self.ceLoss = nn.CrossEntropyLoss(weight=torch.tensor([1., 25.]))
        self.dynamic = dynamic
        if self.dynamic:
            self.v_head = nn.Conv2d(32, 2, kernel_size=1, padding=0)
        self.h_head = nn.Conv2d(32, 32, kernel_size=1, padding=0)

    def forward(self, img, pose=None):
        if pose is None: pose = torch.zeros(img.shape[0]).cuda()

        img -= pose.squeeze().view(pose.shape[0], 1, 1, 1)
        img = self.layers(img)
        if self.dynamic:
            return self.h_head(img), self.v_head(img)
        else:
            return self.h_head(img)
        
    def structure(self):
        layers = nn.Sequential(
            nn.Conv2d(self.inChannel, 32, kernel_size=9, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=9, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 128, kernel_size=14, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # nn.Softmax(dim=1)
            nn.Conv2d(128, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32, 32, kernel_size=1, padding=0),
            # nn.ReLU(inplace=True),
            # nn.Sigmoid()
        )

        return layers

    def compute_loss(self, input_tensor, metric):
        # label_pred = self(input_tensor)
        # label_pred = label_pred.squeeze(2).squeeze(2)
        #
        # # ceLoss = nn.CrossEntropyLoss(weight=self.lossWeight, reduction='sum')
        # # loss = ceLoss(label_pred, label)
        # bceLoss = nn.BCEWithLogitsLoss(pos_weight=self.lossWeight)
        # loss = bceLoss(label_pred, label.float())

        label_pred = self(input_tensor)
        label_pred = label_pred.squeeze(2).squeeze(2)
        loss = self.ceLoss(label_pred, metric)

        return loss

    # pos_weight version
    # def get_label(self, input_tensor):
    #     label_pred_sf = self(input_tensor).squeeze(2).squeeze(2)[:, 1::2]
    #     label_pred_th = (label_pred_sf > 1e-2).clone().detach()
    #     label_pred = torch.zeros_like(label_pred_sf)
    #     s = torch.sum(label_pred_th, dim=1)
    #     index = torch.argmax(label_pred_sf, dim=1)
    #     for i in range(len(label_pred_sf)):
    #         if s[i] > 0:
    #             label_pred[i, index[i]] = 1
    #
    #     return label_pred

    # cross_entropy version
    def get_label(self, input_tensor, mask):
        output_tensor = self(input_tensor).squeeze(2).squeeze(2)
        label_pred = torch.where(self.sf(output_tensor) > 0.5)[1]

        return label_pred


if __name__ == '__main__':
    from torchsummary import summary
    cnn = GQCNN().cuda()
    dataset = GraspDataset('/home/wangchuxuan/library/dataset/parallel_jaw')
    import torch.utils.data as D
    import time
    # train_data = D.DataLoader(dataset, batch_size=64, num_workers=6)
    evlData = D.DataLoader(dataset, batch_size=1)
    device = torch.device("cuda:0")  #
    # for img, depth, label, metric in evlData:
    #
    #     img = img.cuda()
    #     label = label.cuda()
    #     pre = cnn(img)
    #     # print(pre.shape)
    #     pre = pre.squeeze(2).squeeze(2)
    #     print(pre)
    #     label_pred = torch.zeros_like(pre)
    #     print(torch.argmax(pre, dim=1))
    #     label_pred[:, torch.argmax(pre, dim=1)] = 1
    #     print(label_pred)
    #     break

    cnn.load_state_dict(torch.load(
        '/home/wangchuxuan/PycharmProjects/grasp/mygqcnn/21_03_17_19:06/0439_19_5.206595089868435e-07.pth'
        )
    )
    cnn.eval()
    # sf = nn.Softmax(dim=1)
    for img, depth, label, metric in evlData:
        # img0 = img - depth.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        img0 = img.to(device)
        label0 = label.to(device)
        label_pred = cnn(img0)
        if metric:
            print(label_pred.squeeze(2).squeeze(2), label0)







