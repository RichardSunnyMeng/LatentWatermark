import torch
import torch.nn as nn

from torch import Tensor
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        padding_mode="reflect"
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError("BasicBlock only supports groups=1")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class NoBiasBasicBlock(BasicBlock):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(inplanes, planes, stride, downsample, groups, dilation, norm_layer)
        self.relu = nn.Tanh()


class NaiveExtractor(nn.Module):
    def __init__(
        self,
        layers: List[int],
        groups: int = 1,
        block=BasicBlock,
        mode="decouple"
    ) -> None:
        super().__init__()
        self._norm_layer = nn.BatchNorm2d

        self.dilation = 1
        self.inplanes = 3
        replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.conv1 = nn.Conv2d(3, 3, kernel_size=7, stride=1, padding=3, bias=False, padding_mode="reflect")
        self.bn1 = self._norm_layer(3)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 3, layers[0])
        self.layer2 = self._make_layer(block, 3, layers[1], stride=1, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 3, layers[2], stride=1, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 3, layers[3], stride=1, dilate=replace_stride_with_dilation[2])

        self.mode = mode

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
    
    def couple_artifact(self, batch, model_out):
        B, C, W, H = batch['img'].shape
        fake_img, real_img = batch['img'][:B//2], batch['img'][B//2:]
        artifacts = model_out[:B//2]
        artifacts_real = model_out[B//2:]

        fake0 = fake_img
        fake1 = real_img + artifacts
        real0 = real_img
        real1 = fake_img - artifacts

        fake2 = fake0 - artifacts_real
        fake3 = fake1 - artifacts_real
        real2 = real0 + artifacts_real
        real3 = real1 + artifacts_real

        batch['img'] = torch.cat([fake0, fake1, real0, real1, fake2, fake3, real2, real3], dim=0)
        batch["match_indices"] = [(1, 2), (0, 3), (0, 4), (1, 5), (2, 6), (3, 7)]
        batch['label'] = torch.cat([
            batch['label'][:B//2].clone(), 
            batch['label'][:B//2].clone(),
            batch['label'][B//2:].clone(),
            batch['label'][B//2:].clone(),
            batch['label'][:B//2].clone(), 
            batch['label'][:B//2].clone(),
            batch['label'][B//2:].clone(),
            batch['label'][B//2:].clone()
        ], dim=0)

        batch["B"] = B // 2

    def predict_artifact(self, batch, model_out):
        # artifact预测
        B, C, W, H = batch['img'].shape
        fake_img, real_img = batch['img'][:B//2], batch['img'][B//2:]
        pred_fake, pred_real = model_out[:B//2], model_out[B//2:]

        fake0 = fake_img
        fake1 = real_img + pred_real
        real0 = real_img
        real1 = fake_img - pred_fake

        batch['img'] = torch.cat([fake0, fake1, real0, real1], dim=0)
        batch["match_indices"] = [(0, 3), (1, 2)]
        batch['label'] = torch.cat([
            batch['label'][:B//2].clone(), 
            batch['label'][:B//2].clone(),
            batch['label'][B//2:].clone(),
            batch['label'][B//2:].clone(),
        ], dim=0)

        batch["B"] = B // 2

    def forward(self, x: Tensor, others=None) -> Tensor:
        batch = x
        # See note [TorchScript super()]
        x = x['img']
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        artifacts = self.layer4(x)

        couple = False
        if self.training:
            if self.mode == "predict":
                self.predict_artifact(batch, artifacts)
            elif self.mode == "decouple":
                self.couple_artifact(batch, artifacts)
            else:
                raise NotImplementedError
            couple = True

        return {"artifacts": artifacts, "couple": couple}

