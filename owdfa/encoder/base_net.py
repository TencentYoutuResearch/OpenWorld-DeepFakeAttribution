import warnings
import numpy as np
import timm
from loguru import logger

import torch
import torch.nn as nn


__all__ = ['BinaryClassifier']

MODEL_DICTS = {}
MODEL_DICTS.update(timm.models.__dict__)

warnings.filterwarnings("ignore")


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class AdaptiveAvgPool2dCustom(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dCustom, self).__init__()
        self.output_size = np.array(output_size)

    def forward(self, x: torch.Tensor):
        stride_size = np.floor(
            np.array(x.shape[-2:]) / self.output_size).astype(np.int32)
        kernel_size = np.array(x.shape[-2:]) - \
            (self.output_size - 1) * stride_size
        avg = nn.AvgPool2d(kernel_size=list(kernel_size),
                           stride=list(stride_size))
        x = avg(x)
        return x


class BaseClassifier(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes=20,
                 drop_rate=0.2,
                 is_feat=False,
                 neck='no',
                 pretrained=False,
                 **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.is_feat = is_feat
        self.neck = neck

        self.encoder = MODEL_DICTS[encoder](pretrained=pretrained, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

        if hasattr(self.encoder, 'get_classifier'):
            self.num_features = self.encoder.get_classifier().in_features
        else:
            self.num_features = self.encoder.last_channel

        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.pool = AdaptiveAvgPool2dCustom((1, 1))

        self.dropout = nn.Dropout(drop_rate)

        if self.neck == 'bnneck':
            logger.info('Using BNNeck')
            self.bottleneck = nn.BatchNorm1d(self.num_features)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.fc2 = nn.Linear(
                self.num_features, self.num_classes, bias=False)
            self.bottleneck.apply(weights_init_kaiming)
            self.fc2.apply(weights_init_classifier)
        else:
            self.fc2 = nn.Linear(self.num_features, self.num_classes)

    def forward_featuremaps(self, x):
        featuremap = self.encoder.forward_features(x)
        return featuremap

    def forward_features(self, x):
        featuremap = self.encoder.forward_features(x)
        feature = self.pool(featuremap).flatten(1)

        if self.neck == 'bnneck':
            feature = self.bottleneck(feature)

        return feature

    def forward(self, x, label=None):
        feature = self.forward_features(x)

        x = self.dropout(feature)
        method = self.fc2(x)

        y = method

        if self.is_feat:
            return y, feature

        return y


class CPLClassifier(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes=20,
                 num_patch=3,
                 drop_rate=0.2,
                 is_feat=False,
                 neck='bnneck',
                 pretrained=False,
                 **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_patch = num_patch
        self.drop_rate = drop_rate
        self.is_feat = is_feat
        self.neck = neck

        self.encoder = MODEL_DICTS[encoder](pretrained=pretrained, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

        if hasattr(self.encoder, 'get_classifier'):
            self.num_features = self.encoder.get_classifier().in_features
        else:
            self.num_features = self.encoder.last_channel

        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.part = nn.AdaptiveAvgPool2d((self.num_patch, self.num_patch))

        self.pool = AdaptiveAvgPool2dCustom((1, 1))
        self.part = AdaptiveAvgPool2dCustom((self.num_patch, self.num_patch))

        self.dropout = nn.Dropout(drop_rate)

        self.bottleneck = nn.BatchNorm1d(self.num_features)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.fc2 = nn.Linear(self.num_features, self.num_classes, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.fc2.apply(weights_init_classifier)

        for i in range(self.num_patch ** 2):
            name = 'bnneck' + str(i)
            setattr(self, name, nn.BatchNorm1d(self.num_features))
            getattr(self, name).bias.requires_grad_(False)
            getattr(self, name).apply(weights_init_kaiming)

    def forward_featuremaps(self, x):
        featuremap = self.encoder.forward_features(x)
        return featuremap

    def forward_features(self, x):
        featuremap = self.encoder.forward_features(x)
        f_g = self.pool(featuremap).flatten(1)
        f_p = self.part(featuremap)
        f_p = f_p.view(f_p.size(0), f_p.size(1), -1)

        fs_p = []
        for i in range(self.num_patch ** 2):
            f_p_i = f_p[:, :, i]
            f_p_i = getattr(self, 'bnneck' + str(i))(f_p_i)
            fs_p.append(f_p_i)
        fs_p = torch.stack(fs_p, dim=-1)

        f_g = self.bottleneck(f_g)

        return (f_g, fs_p)

    def forward(self, x, label=None):
        feature = self.forward_features(x)
        (f_g, _) = feature

        x = self.dropout(f_g)
        method = self.fc2(x)

        y = method

        if self.is_feat:
            return y, feature

        return y
