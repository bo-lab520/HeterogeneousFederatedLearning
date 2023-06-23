import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)  # 6 x 28 x 28
        out = F.max_pool2d(out, 2)  # 6 x 14 x 14
        out = F.relu(self.conv2(out), inplace=True)  # 16 x 7 x 7
        out = F.max_pool2d(out, 2)   # 16 x 5 x 5
        out = out.view(out.size(0), -1)  # 16 x 5 x 5
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)

        return F.log_softmax(out, dim=1)


def get_model(name="vgg16", pretrained=True):
    if name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
    elif name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
    elif name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
    elif name == "vgg19":
        model = models.vgg19(pretrained=pretrained)
    elif name == "inception_v3":
        model = models.inception_v3(pretrained=pretrained)
    elif name == "googlenet":
        model = models.googlenet(pretrained=pretrained)
    elif name == "LeNet":
        model = LeNet()

    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model
