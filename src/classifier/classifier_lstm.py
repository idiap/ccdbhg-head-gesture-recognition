# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import torch.nn as nn

# Local libs
from src.classifier.base import BaseClassifier, get_global_pooling


class LSTMClassifier(BaseClassifier):
    """Class for LSTM classifier"""

    def __init__(self, configs, name="lstm", verbose=True, **kwargs):
        super().__init__(name, configs.num_classes, verbose)

        self.configs = configs
        self.path_to_pretrained = configs.path_to_pretrained

        # CNN classifier
        self.init_classifier()

    def init_variables(self):
        pass

    def init_classifier(self):
        """Create the LSTM classifier"""
        if self.configs.model_name == "lstm":
            self.model = LSTMModel(
                encoder="lstm",
                g_pool=self.configs.g_pool,
                in_ch=self.configs.in_ch,
                dropout=self.configs.dropout,
                num_classes=self.configs.num_classes,
                num_layers=self.configs.num_layers,
                hidden_size=self.configs.hidden_size,
                bidirectional=self.configs.bidirectional,
            ).to(self.configs.device)
        elif self.configs.model_name == "gru":
            self.model = LSTMModel(
                encoder="gru",
                g_pool=self.configs.g_pool,
                in_ch=self.configs.in_ch,
                dropout=self.configs.dropout,
                num_classes=self.configs.num_classes,
                num_layers=self.configs.num_layers,
                hidden_size=self.configs.hidden_size,
                bidirectional=self.configs.bidirectional,
            ).to(self.configs.device)
        else:
            raise NotImplementedError(f"invalid model_name: {self.configs.model_name}")


class LSTMModel(nn.Module):
    def __init__(
        self,
        encoder="lstm",
        g_pool="avg_pool",
        num_layers=2,
        hidden_size=64,
        in_ch=1,
        num_classes=1,
        dropout=0.1,
        bidirectional=False,
        **kwargs,
    ):
        super().__init__()

        # Encoder
        if encoder == "lstm":
            self.encoder = nn.LSTM(
                input_size=in_ch,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )
        elif encoder == "gru":
            self.encoder = nn.GRU(
                input_size=in_ch,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )
        else:
            raise NotImplementedError(f"invalid encoder: {encoder}")

        # Global Pool
        self.global_pool = get_global_pooling(g_pool)

        self.hidden_size = hidden_size
        self.bi_dim = 2 if bidirectional else 1

        # Classifier head
        self.head = nn.Linear(hidden_size * self.bi_dim, num_classes)

    def forward(self, x, emb_only=False):
        # usally x.shape = (batch_size, in_ch, seq_len) for CNN
        # but here we use lstm thus x.shape = (batch_size, seq_len, in_ch)
        x = x.transpose(1, 2)

        emb, _ = self.encoder(x)

        if emb_only:
            return emb[:, -1, :]

        # emb is (NxLxH) where N is batch size, L is seq_len, H is hidden_size
        if self.global_pool is not None:
            out = emb.transpose(1, 2)
            out = self.global_pool(out).squeeze(2)
        else:
            out = emb[:, -1, :]

        assert out.shape[1] == self.hidden_size * self.bi_dim
        out = self.head(out)

        return out, emb[:, -1, :]
