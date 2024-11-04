import torch
import torchvision
import torch.nn as nn


class DeeplabV3(nn.Module):
    def __init__(self, configs):
        super(DeeplabV3, self).__init__()
        self.in_channels = configs['in_channels']
        self.out_channels = configs['out_channels']

        self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1', weights_backbone='IMAGENET1K_V1')
        self.model.backbone.conv1 = torch.nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        torch.nn.init.kaiming_normal_(self.model.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.model.classifier[4] = torch.nn.Conv2d(256, self.out_channels, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)['out']