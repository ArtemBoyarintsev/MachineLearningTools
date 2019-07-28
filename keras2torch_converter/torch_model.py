import torch.nn as nn
import torch


def conv3x3(in_planes, out_planes, stride):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes,eps=1e-3, momentum=0.01)  # batch_norm issue 
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride=2)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.downsample = nn.Conv2d(inplanes, planes,
                                    kernel_size=1, stride=2, bias=True)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu1(self.bn1(x))
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv2(out)
        out += residual

        return out


class Flatten(nn.Module):  # Flatten issue 
    def forward(self, input):
        batch_size = input.shape[0]
        temp = input.permute(0, 2, 3, 1)  # Cause we need the same sequence of dimension as in Keras
        return temp.contiguous().view(batch_size, -1) 


class ResNet8(nn.Module):
    def __init__(self, fine_tune_path=None):
        super(ResNet8, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=0, bias=True)
        self.mxpl1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.block1 = BasicBlock(32, 32)
        self.block2 = BasicBlock(32, 64)
        self.block3 = BasicBlock(64, 128)

        self.flatten = nn.Sequential(
            Flatten(),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.steer = nn.Linear(6272, 1)
        self.coll = nn.Linear(6272, 1)

        if fine_tune_path is not None:
            print(f"    Fine tune model from path: {fine_tune_path}")
            self.load_state_dict(torch.load(fine_tune_path))

    def forward(self, x):
        padder = torch.nn.ZeroPad2d((1, 2, 1, 2))  # padding issue 
        x_conv1 = self.conv1(padder(x))
        x = self.mxpl1(x_conv1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        
        steer = self.steer(x)
        coll = self.coll(x)
        return steer, coll
