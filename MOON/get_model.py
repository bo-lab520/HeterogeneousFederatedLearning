import torch
from torch import nn
import torch.nn.functional as F

from models.ResNetv1 import resnet18, resnet34, resnet50, resnet101, resnet152
from models.ResNetv2 import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, \
    resnet110, resnet116, resnet8x4, resnet32x4
from models.CNNMnist import cnnmnist


def get_model(name):
    if name == "resnet18":
        model = resnet18()
    elif name == "resnet34":
        model = resnet34()
    elif name == "resnet50":
        model = resnet50()
    elif name == "resnet101":
        model = resnet101()
    elif name == "resnet152":
        model = resnet152()
    elif name == "resnet8":
        model = resnet8()
    elif name == "resnet14":
        model = resnet14()
    elif name == "resnet20":
        model = resnet20()
    elif name == "resnet32":
        model = resnet32()
    elif name == "resnet44":
        model = resnet44()
    elif name == "resnet56":
        model = resnet56()
    elif name == "resnet110":
        model = resnet110()
    elif name == "resnet116":
        model = resnet116()
    elif name == "resnet8x4":
        model = resnet8x4()
    elif name == "resnet32x4":
        model = resnet32x4()
    elif name == "cnnmnist":
        model = cnnmnist()

    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model


class MOONModel(nn.Module):

    def __init__(self, base_model, out_dim, n_classes):
        super(MOONModel, self).__init__()

        basemodel = get_model(base_model)
        self.features = nn.Sequential(*list(basemodel.children())[:-1])
        num_ftrs = basemodel.fc.in_features

        # projection MLP 投影器
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer 输出层
        self.l3 = nn.Linear(out_dim, n_classes)

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        # 投影器
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        return h, x, y
