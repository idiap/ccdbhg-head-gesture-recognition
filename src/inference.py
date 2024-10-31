# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.classifier import CNNClassifier, LSTMClassifier, MLPClassifier, TCNClassifier
from src.utils import TS_Normalizer, get_landmarks, normalize_landmarks
from src.utils_dataset import (
    compute_relative_invariant,
    compute_relative_torch,
    is_input,
)

CLASSIFIERS = {
    "mlp": MLPClassifier,
    "cnn": CNNClassifier,
    "tcn": TCNClassifier,
    "lstm": LSTMClassifier,
    "gru": LSTMClassifier,
}

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Class for head gesture inference
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#


class HeadGestureInference(object):
    def __init__(self, exp_folder, device="cpu"):
        self.exp_folder = exp_folder
        self.device = device
        self.label_name = ["none", "nod", "shake", "tilt", "turn", "up/down"]

        # load config
        self.init_variable()
        # load model
        self.load_classifier()

    @staticmethod
    def load_config(path_exp):
        with open(os.path.join(path_exp, "config.yaml"), "r") as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        return configs

    def init_variable(self):
        """
        Load experiment variables
        """
        # load config
        configs = self.load_config(self.exp_folder)

        self.config = argparse.Namespace(**configs)

        # load normalizer
        is_rel = self.config.transform_abs_rel == "rel"
        mean_method = self.config.mean_method if not is_rel else "none"  # NOTE: hard coded
        self.normalizer = TS_Normalizer(
            mean_method=mean_method,
            std_method=self.config.std_method,
            channel_wise=self.config.norm_channel_wise,
            load_path=self.exp_folder,
        )
        # parameter
        self.relative = self.config.transform_relative
        self.input_domain = self.config.input_domain
        self.transform_abs_rel = self.config.transform_abs_rel
        self.relative_invariant = self.config.relative_invariant
        self.relative_method = self.config.relative_method
        self.gaze_type = self.config.gaze_type
        self.config.device = self.device

    def load_classifier(self):
        classifier = CLASSIFIERS[self.config.classifier_name](self.config)
        path = os.path.join(self.exp_folder, self.config.classifier_name)
        classifier.load_classifier(path)
        self.model = classifier.model
        self.model.eval()

    def feature_selection(self, x_data: dict):
        """
        Select the features for the model choice
        match with the features name from x_data

        Args:
            x_data (_type_): data extracted should contain at least landmark, head_pose
            or gaze in the name of the key

        Returns:
            feature_names: list of features name
            matching_key: dict of matching key
        """
        feature_names = self.input_domain.copy()
        feature_names.sort()

        matching_key = {k: None for k in x_data.keys()}
        for k in x_data.keys():
            for feature in feature_names:
                if feature in k:
                    matching_key[k] = feature
                    break

        # check if all features are present
        unused_keys = [k for k, v in matching_key.items() if v is None]
        # delete unused keys
        for k in unused_keys:
            matching_key.pop(k)

        matching_key = {v: k for k, v in matching_key.items()}

        return feature_names, matching_key

    def input_adapter(self, x_data: dict):
        """
        Input adapter from extracted data to model input
        x_data: dict of extracted data
        """

        feature_names, matching_key = self.feature_selection(x_data)

        x_tmp = []
        for feature in feature_names:
            # check input features
            if not is_input(feature):
                continue

            if "landmark" in feature:
                if len(x_data[matching_key[feature]][0]) != 15:
                    x_data[matching_key[feature]] = get_landmarks(x_data[matching_key[feature]])
                x_data[matching_key[feature]] = normalize_landmarks(x_data[matching_key[feature]])

            # invariance
            if self.relative_invariant:
                x_tmp.append(
                    compute_relative_invariant(
                        x_data[matching_key[feature]],
                        self.relative,
                        x_data[matching_key["head_pose"]],
                        feature,
                        self.gaze_type,
                        self.relative_method,
                    )
                )
            # raw features
            else:
                x_tmp.append(x_data[matching_key[feature]])

        # add sample with all features to all data
        x_data_out = np.concatenate(x_tmp, axis=1)
        x_data_out = x_data_out.reshape(1, -1, x_data_out.shape[-1])

        # n_x, window_size, feature_size = x_data_out.shape
        # print('x_data_out', x_data_out.shape)

        # if relative variant but not invariant
        if (self.transform_abs_rel == "rel") and (not self.relative_invariant):
            x_data_out = compute_relative_torch(x_data_out, self.relative)

        # normalization
        if self.normalizer is not None:
            x_data_out = self.normalizer.normalize(x_data_out)

        return x_data_out

    def preprocess(self, image, info, data):
        """
        Preprocess input data
        """
        process_data = self.input_adapter(data)

        # convert to tensor
        x = torch.from_numpy(process_data).float()
        x = x.permute(0, 2, 1)

        return x

    def inference(self, x_tensor: torch.Tensor):
        """
        Model inference
        """
        with torch.no_grad():
            y_pred, _ = self.model(x_tensor.to(self.device))
            probs = F.softmax(y_pred, dim=1)

        if x_tensor.shape[0] == 1:
            return {"pred": y_pred.detach().cpu()[0], "prob": probs.detach().cpu()[0]}
        else:
            return {"pred": y_pred.detach().cpu(), "prob": probs.detach().cpu()}

    def get_data(self, out):
        """
        Return post-processed prediction in the correct format
        """
        # get label
        idx_pred = out["prob"].argmax()
        # print(out["prob"],idx_pred)
        return {
            "head_gesture": self.label_name[idx_pred],
            "head_geture_idx": idx_pred,
            "head_gesture_prob": out["prob"],
            "confidence": out["prob"][idx_pred],
        }

    def process(self, image, info, data):
        """
        Run the all pipeline
        """
        if data is None:
            return self.get_data({"prob": np.array([1.0, 0, 0, 0, 0, 0])})
        input = self.preprocess(image, info, data)
        out = self.inference(input)
        out = self.get_data(out)

        return out
