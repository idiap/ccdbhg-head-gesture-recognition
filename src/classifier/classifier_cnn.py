# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import torch.nn as nn

# Local libs
from src.classifier.base import BaseClassifier, get_global_pooling


class CNNClassifier(BaseClassifier):
    """Class for CNN classifier"""

    def __init__(self, configs, name="cnn", verbose=True, **kwargs):
        super().__init__(name, configs.num_classes, verbose)

        self.configs = configs
        self.path_to_pretrained = configs.path_to_pretrained

        # CNN classifier
        self.init_classifier()

    def init_variables(self):
        pass

    def init_classifier(self):
        """Create the CNN classifier"""
        if self.configs.model_name == "cnn":
            self.model = CNNModel(
                in_ch=self.configs.in_ch,
                kernel_size=self.configs.kernel_size,
                stride=self.configs.stride,
                dropout=self.configs.dropout,
                feat_dim=self.configs.feat_dim,
                num_classes=self.configs.num_classes,
                bias=self.configs.bias,
                max_len=self.configs.max_len,
                g_pool=self.configs.g_pool,
                batch_norm=self.configs.batch_norm,
            ).to(self.configs.device)
        else:
            raise NotImplementedError(f"invalid model_name: {self.configs.model_name}")


class CNNModel(nn.Module):
    def __init__(
        self,
        in_ch=5,
        kernel_size=[8, 8, 8],
        stride=[2, 1, 1],
        dropout=[0.3, 0.0, 0.0],
        feat_dim=[32, 64, 64],
        batch_norm=True,
        num_classes=3,
        max_len=31,
        g_pool="max_pool",
        bias=False,
        **kwargs,
    ):
        super().__init__()

        encoder = []
        for ix in range(len(feat_dim)):
            encoder += [
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=feat_dim[ix],
                    kernel_size=kernel_size[ix],
                    stride=stride[ix],
                    bias=bias,
                    padding=(kernel_size[ix] // 2),
                ),
            ]
            if batch_norm:
                encoder += [nn.BatchNorm1d(feat_dim[ix])]

            encoder += [nn.ReLU(), nn.Dropout(dropout[ix])]

            in_ch = feat_dim[ix]

        self.out_ch = feat_dim[ix]

        # Encoder
        self.model = nn.Sequential(*encoder)

        # Global Pool
        self.global_pool = get_global_pooling(g_pool)
        if self.global_pool is None:
            self.out_ch = self.out_ch * max_len

        # Classifier head
        self.head = nn.Linear(self.out_ch, num_classes)

    def forward(self, x, emb_only=False):
        emb = self.model(x)

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
