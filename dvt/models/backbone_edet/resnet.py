import torch
import torch.nn as nn
import torchvision.models as models
from dvt.models.backbone_edet.efficientnet_utils import (round_filters, round_repeats, drop_connect,
                                                                  get_same_padding_conv2d, get_model_params,
                                                                  efficientnet_params, load_pretrained_weights,
                                                                  Swish, MemoryEfficientSwish)

from dvt.models.backbone_edet.efficientdet import BiFPN
from dvt.models.backbone_edet.efficientdet_utils import Anchors, load_pretrained_weights


class ResNetBackbone(nn.Module):

    def __init__(self, compound_coef=3):
        super(ResNetBackbone, self).__init__()
        self.compound_coef = compound_coef
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.resnet = models.resnet152(pretrained=True)

        in_channels = 3
        out_channels = 64

        # self.num_classes = num_classes
        # self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
        #                            num_layers=self.box_class_repeats[self.compound_coef],
        #                            pyramid_levels=self.pyramid_levels[self.compound_coef])
        # self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
        #                              num_classes=num_classes,
        #                              num_layers=self.box_class_repeats[self.compound_coef],
        #                              pyramid_levels=self.pyramid_levels[self.compound_coef])

        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
                               pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist())

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=160, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=160, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=160, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=2048, out_channels=160, kernel_size=3, stride=2, padding=1, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, inputs):
        max_size = inputs.shape[-1]

        features = self.conv1(inputs)
        features = self._bn0(features)

        p2 = self.resnet.layer1(features)
        p3 = self.resnet.layer2(p2)
        p4 = self.resnet.layer3(p3)
        p5 = self.resnet.layer4(p4)

        p2 = self.conv2(p2)
        p3 = self.conv3(p3)
        p4 = self.conv4(p4)
        p5 = self.conv5(p5)

        features = (p2, p3, p4, p5)

        #features = self.bifpn(features)
        # features = self.bifpn(features)

        # regression = self.regressor(features)
        # classification = self.classifier(features)
        # anchors = self.anchors(inputs, inputs.dtype)

        return features  # , regression, classification, anchors
