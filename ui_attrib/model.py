from torchvision.models import resnet50
import torch.nn as nn
import torch

class RegressionResNet(nn.Module):
    def __init__(self, out_channels=6):
        super(RegressionResNet, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.fc = nn.Linear(1000, out_channels)

    def forward(self, x):
        x = self.resnet(x)
        # x = torch.cat((x, w.unsqueeze(1), h.unsqueeze(1)), dim=1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x