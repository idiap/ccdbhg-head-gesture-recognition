# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import torch.nn as nn

# Local libs
from src.classifier.base import BaseClassifier


class MLPClassifier(BaseClassifier):
    """Class for MLP classifier"""

    def __init__(self, configs, name="mlp", verbose=True, **kwargs):
        super().__init__(name, configs.num_classes, verbose)

        self.configs = configs

        # MLP classifier
        self.init_classifier()

    def init_variables(self):
        pass

    def init_classifier(self):
        """Create the MLP classifier"""
        if self.configs.model_name == "mlp":
            self.model = MLPModel(
                in_ch=self.configs.in_ch,
                hidden_dims=self.configs.hidden_dims,
                num_classes=self.configs.num_classes,
            ).to(self.configs.device)
        else:
            raise NotImplementedError(f"invalid mlp model {self.configs.model_name}")


class MLPModel(nn.Module):
    def __init__(self, in_ch=5, hidden_dims=[30, 30, 30, 30], num_classes=3, **kwargs):
        super().__init__()

        model = []
        in_hd = in_ch
        # self.dropout_in = nn.Dropout(0.1)

        for ix, out_hd in enumerate(hidden_dims):
            model += [nn.Linear(in_hd, out_hd)]
            # model += [nn.BatchNorm1d(out_hd)]
            model += [nn.ReLU()]
            # model += [nn.Dropout(dropout[ix])]
            in_hd = out_hd

        self.model = nn.Sequential(*model)

        # Classifier head
        self.head = nn.Linear(out_hd, num_classes)

    def forward(self, x):
        emb = x.reshape(x.shape[0], -1)
        emb = self.model(x)
        x = self.head(emb)
        return x, emb
