# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import os
import pickle
import random

import numpy as np
import sklearn
import torch
import torch.nn as nn


class BaseClassifier:
    """Base class for different classifiers"""

    def __init__(self, name, num_classes, verbose=True):
        self.name = name  # Classifier name
        self.verbose = verbose  # Verbosity
        self.num_classes = num_classes  # Number of classes

        # Classifier model
        self.model = None

        # self.fix_random_seed()
        self.init_variables()

    def init_variables(self):
        pass

    def init_classifier(self):
        """Initialize/create classifier"""
        pass

    def fix_random_seed(self, seed=33):
        """Fixing all random seeds"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        sklearn.utils.check_random_state(seed)

    def train(self, x_data, y_data):
        """
        Train the classifier with input data (x_data) and its corresponding
        labels (y_data)
        """
        pass

    def predict(self, x_data):
        """Predict the class for the input data (x_data)"""
        pass

    def save(self, out_path):
        """Save the classifier model as pickle file"""
        pickle.dump(self.model, open(out_path, "wb"))

    def run_classifier(self, train_loader, val_loader, test_loader):
        """Run the classifier"""
        pass

    def save_classifier(self, path):
        state_dict = {
            "model": self.model.state_dict(),
        }
        file_name = "ckpt.pth.tar"
        save_path = os.path.join(path, file_name)
        torch.save(state_dict, save_path)

    def predict_unlabeled(self, test_loader):
        assert self.model is not None, "Classifier model is None"
        return self.trainer.predict_unlabeled(test_loader)

    def load_model(self, net, path_to_exp):
        """
        load the pre-trained weights & (freeze if needed ...)
        """
        if path_to_exp is not None:
            load_path = os.path.join(path_to_exp, "ckpt.pth.tar")
            state_dict = torch.load(load_path)
            state_dict["model"] = {
                k.replace("model.", ""): v for k, v in state_dict["model"].items()
            }
            net.load_state_dict(state_dict["model"], strict=False)

        return net

    def load_classifier(self, path_to_exp):
        """
        load the pre-trained weights & (freeze if needed ...)
        """
        load_path = os.path.join(path_to_exp, "ckpt.pth.tar")
        state_dict = torch.load(load_path, map_location=torch.device("cpu"))
        # state_dict['model'] = {k.replace('model.', ''):v for k,v in state_dict['model'].items()}
        self.model.load_state_dict(state_dict["model"], strict=True)

        return self.model


def get_global_pooling(g_pool):
    # Global Pool
    if g_pool == "max_pool":
        return nn.AdaptiveMaxPool1d(1, return_indices=False)
    elif g_pool == "avg_pool":
        return nn.AdaptiveAvgPool1d(1)
    elif g_pool == "none":
        return None
    else:
        raise NotImplementedError(f"Invalid global pooling: {g_pool}")
