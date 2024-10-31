# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import torch.nn as nn
import torch.nn.functional as F

# Local libs
from src.classifier.base import BaseClassifier, get_global_pooling


class TCNClassifier(BaseClassifier):
    """Class for TCN classifier"""

    def __init__(self, configs, name="tcn", verbose=True, **kwargs):
        super().__init__(name, configs.num_classes, verbose)

        self.configs = configs
        self.path_to_pretrained = configs.path_to_pretrained

        # TCN classifier
        self.init_classifier()

    def init_variables(self):
        pass

    def init_classifier(self):
        """Create the TCN classifier"""
        if self.configs.model_name == "tcn":
            self.model = TCNModel(
                in_ch=self.configs.in_ch,
                num_f_maps=self.configs.num_f_maps,
                num_layers=self.configs.num_layers,
                num_classes=self.configs.num_classes,
                max_len=self.configs.max_len,
                g_pool=self.configs.g_pool,
            ).to(self.configs.device)
        else:
            raise NotImplementedError(f"invalid model_name: {self.configs.model_name}")


# -------------------------------------------------------
class TCNModel(nn.Module):
    def __init__(
        self, in_ch, num_f_maps, num_layers, max_len=31, g_pool="max_pool", num_classes=3, **kwargs
    ):
        super().__init__()

        self.conv_1x1 = nn.Conv1d(in_ch, num_f_maps, 1)
        # self.layers = nn.ModuleList([copy.deepcopy(
        #     DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)
        #     ) for i in range(num_layers)])
        self.layers = [
            DilatedResidualLayer(2**i, num_f_maps, num_f_maps) for i in range(num_layers)
        ]
        self.model = nn.Sequential(*self.layers)

        # Global Pool
        self.global_pool = get_global_pooling(g_pool)
        if self.global_pool is None:
            num_f_maps = num_f_maps * max_len

        # Head
        self.head = nn.Linear(num_f_maps, num_classes)

    def forward(self, x, emb_only=False):
        emb = self.conv_1x1(x)
        emb = self.model(emb)
        # for layer in self.layers:
        #     emb = layer(emb)

        if self.global_pool is not None:
            emb = self.global_pool(emb).squeeze(2)
            emb = emb.view(-1, emb.shape[1])
        else:
            emb = emb.view(emb.shape[0], -1)

        if emb_only:
            return emb

        if self.head is not None:
            x = self.head(emb)

        return x, emb


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


# class TemporalBlock(nn.Module):
#     def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
#         super(TemporalBlock, self).__init__()
#         self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
#                                            stride=stride, padding=0, dilation=dilation))
#         self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
#                                            stride=stride, padding=0, dilation=dilation))
#         self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout,
#                                  self.pad, self.conv2, self.relu, self.dropout)
#         self.downsample = nn.Conv1d(
#             n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
#         self.relu = nn.ReLU()
#         self.init_weights()

#     def init_weights(self):
#         self.conv1.weight.data.normal_(0, 0.01)
#         self.conv2.weight.data.normal_(0, 0.01)
#         if self.downsample is not None:
#             self.downsample.weight.data.normal_(0, 0.01)

#     def forward(self, x):
#         out = self.net(x.unsqueeze(2)).squeeze(2)
#         res = x if self.downsample is None else self.downsample(x)
#         return self.relu(out + res)


# class TemporalConvNet(nn.Module):
#     def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
#         super(TemporalConvNet, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = num_inputs if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
#                                      padding=(kernel_size-1) * dilation_size, dropout=dropout)]

#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.network(x)
