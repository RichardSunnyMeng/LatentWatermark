import torch
import torch.nn as nn
from torchvision.models import resnet18


class NaiveDetector(nn.Module):
    def __init__(self, pretrained=False):
        super(NaiveDetector, self).__init__()

        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, 2)
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, 1.0)
        torch.nn.init.constant_(self.model.fc.bias.data, 0)

    def forward(self, batch, others=None):
        return {"logits": self.model(batch["img"])}