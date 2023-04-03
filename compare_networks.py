import torch.nn as nn
import torch
import math
from torchvision.transforms.functional import resize


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1 conv
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3x3 conv
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1x1 conv
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, layers, inChannel=1, block=Bottleneck, dynamic=False):
        self.inplanes = 64
        self.inChannel = inChannel
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.dynamic = dynamic
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 // 2, layers[0])
        self.layer2 = self._make_layer(block, 128 // 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256 // 2, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 256 // 2, layers[3], stride=2)

        self.norm_img = nn.BatchNorm2d(256*2)
        self.norm_pose = nn.BatchNorm2d(256*2)
        self.out = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=8, padding=0),
        )
        if dynamic:
            self.out_pos = nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=3, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=3, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 2, kernel_size=8, padding=0),
            )
        else:
            self.out_pos = nn.Identity()
        # kaiming weight normal after default init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.sf = nn.Softmax(dim=1)
        self.ceLoss = nn.CrossEntropyLoss()

        self.layers = self.structure()

    # construct layer/stage conv2_x,conv3_x,conv4_x,conv5_x
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # when to need downsample
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        # inplanes expand for next block
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, *x, shape=None):
        if len(x) == 1:
            pose = torch.zeros(x[0].shape[0]).cuda()
            x = [x[0], pose]
        x, pose = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.norm_img(x)
        if shape is not None:
            x = resize(x, list(shape))

        pos_bias = self.out_pos(x)

        if x.shape[0] == 1:
            x = x.repeat(pose.shape[0], 1, 1, 1)
        pose = self.norm_pose(pose.squeeze().view(pose.shape[0], 1, 1, 1).expand_as(x))
        x -= pose

        x = self.out(x)
        if self.dynamic:
            return x, pos_bias
        else:
            return x

        # return x if not self.dynamic else x, pos_bias

    # def forward(self, x):
    #     return self.layers(x)

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
            nn.Conv2d(32, 2, kernel_size=1, padding=0),
            # nn.ReLU(inplace=True),
            # nn.Sigmoid()
        )

        return layers


if __name__ == '__main__':
    from torchsummary import summary
    cnn = ResNet([2, 2, 2]).cuda()
    # cnn.load_state_dict(torch.load(
    #     '/home/wangchuxuan/PycharmProjects/grasp/coGQcnnRgb.pth'
    #     # '/home/wangchuxuan/PycharmProjects/grasp/cogqcnn.pth'
    #     )
    # )
    summary(cnn, (1, 96, 96), batch_size=16)
