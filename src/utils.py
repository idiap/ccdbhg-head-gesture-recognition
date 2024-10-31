# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import math
import os
import pickle

import numpy as np
from scipy import interpolate
from scipy.spatial.transform import Rotation as R_

# landmarks index per facial feature
left_eye = [130, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25]
right_eye = [463, 414, 286, 258, 257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341]
nose = [51, 5, 281, 45, 4, 275, 220, 134, 236, 3, 195, 248, 456, 363, 440]
ear_l = [234, 93, 227, 137]
ear_r = [454, 323, 447, 366]
bottom_lip = [14, 15, 16, 17, 87, 86, 85, 84, 317, 316, 315, 314]
top_lip = [0, 11, 12, 13, 37, 72, 38, 82, 267, 302, 268, 312]
left_eyelash = [161, 160, 159, 158, 157]
right_eyelash = [388, 387, 386, 385, 384]
chin_tips = [140, 208, 199, 428, 269, 400, 296, 377, 175, 152, 171, 148, 176]


def get_landmarks(lmks):
    """Selected landmarks from mediapipe
    Args:
        lmks (array): 468x3 landmarks of mediapipe
    Returns:
        selected_landmarks (array): 5x3 landmarks
    """
    assert lmks.shape[1] == 3, f"Landmarks shape is {lmks.shape} instead of (468, 3)"
    left_eye_avg = lmks[left_eye].mean(axis=0, keepdims=True)
    right_eye_avg = lmks[right_eye].mean(axis=0, keepdims=True)
    nose_avg = lmks[nose].mean(axis=0, keepdims=True)
    ear_l_avg = lmks[ear_l].mean(axis=0, keepdims=True)
    ear_r_avg = lmks[ear_r].mean(axis=0, keepdims=True)
    selected_landmarks = np.concatenate(
        [left_eye_avg, right_eye_avg, nose_avg, ear_l_avg, ear_r_avg], axis=0
    )
    return selected_landmarks.reshape(-1)


def get_head_pose(R):
    re = R.reshape(4, 4)
    rotation_x_pi = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

    re[1:3, :] = -re[1:3, :]
    rot_matrix = re[:3, :3] @ rotation_x_pi

    # Head pose angle (Euler)
    euler_face_geometry = R_.from_matrix(rot_matrix).as_euler("xyz", degrees=False)

    pitch = euler_face_geometry[0]
    roll = euler_face_geometry[2]
    yaw = euler_face_geometry[1]
    return np.array([yaw, roll, pitch])


def normalize_landmarks(x, window_size=31):
    """
    Normalize landmarks with the head size
    distance between the ears
    x : one landmark sequence
    """

    if isinstance(x, list):
        x = np.array(x)

    landmarks_ = x.reshape(window_size, 5, 3)
    head_size = np.linalg.norm(landmarks_[:, 3, :] - landmarks_[:, 4, :], axis=1)
    head_size = np.mean(head_size) / 100
    norm_landmarks = x / head_size
    return norm_landmarks


class TS_Normalizer(object):
    """
    Normalizes a multivariate time series NumPy array with mean per sample and std per channel.
    NOTE: if per modality normalization needed, please do it outside here with seperate
          normalizers and concat (respecting order)
    """

    def __init__(
        self, mean_method, std_method, channel_wise=True, mean=None, std=None, load_path=None
    ):
        self.mean_method = mean_method
        self.std_method = std_method
        self.channel_wise = channel_wise
        self.mean = mean
        self.std = std
        self.load_path = load_path

        # load stats if exists
        try:
            self.mean, self.std = self.load_stat(load_path)
        except:
            raise ValueError("No stats found at %s" % load_path)

    @staticmethod
    def load_stat(load_path):
        path = os.path.join(load_path, "train_stats.pkl")
        with open(path, "rb") as pickle_file:
            dt = pickle.load(pickle_file)
        return dt["mean"], dt["std"]

    def broadcast(self, stat, shape, method):
        """
        Broadcast statistics.
        NOTE: do NOT merge this function with compute_stats
        """
        B, T, C = shape
        if method == "none":
            return None
        # per-sample stat
        elif method == "per_sample":
            b, t, c = (1, T, 1) if self.channel_wise else (1, T, C)
        # per-data stat
        elif method == "per_data":
            b, t, c = (B, T, 1) if self.channel_wise else (B, T, C)
        else:
            raise NotImplementedError(f"{method} stat method  is not available")
        # Broadcast the means and std to the original shape of the array
        stat_broad = np.tile(stat, (b, t, c))

        return stat_broad

    def normalize(self, arr):
        """
        Args:
            arr: Input NumPy array of shape (B, T, C)

        Returns:
            normalized_arr: Normalized NumPy array of the same shape
        """

        # Broadcast the means and std to the original shape of the array
        std_broadcasted = self.broadcast(self.std, arr.shape, self.std_method)
        if self.mean is None:
            # Apply the normalization
            normalized_arr = arr / (std_broadcasted + np.finfo(float).eps)
            # print('Norm: divided by std')
        else:
            means_broadcasted = self.broadcast(self.mean, arr.shape, self.mean_method)
            # Apply the normalization
            normalized_arr = (arr - means_broadcasted) / (std_broadcasted + np.finfo(float).eps)

        return normalized_arr


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


def time_interpolation(time_stamp, y, kind="linear"):
    """
    Interpolate a time series data for
    a fixes interval given original time_stamp

    30 fps => 33.333 ms
    31 windows => 1033.33 ms
    Args:
        time_stamp: list of time stamp (L)
        y: list of data to interpolate (Lxd)
    """
    window_size = 31
    time_of_window = 33.333 * (window_size - 1)

    assert len(time_stamp) == len(y), "len(time_stamp) != len(y)"

    end_time = time_stamp[-1]
    start_idx = None
    # check beginning
    for i, t in enumerate(time_stamp[::-1]):
        time_diff = end_time - t
        if time_diff > time_of_window:
            start_idx = len(time_stamp) - 1 - i
            break

    # interpolate
    if start_idx is not None:
        y = y[start_idx:]
        time_stamp = time_stamp[start_idx:]
    else:
        raise ValueError("Segement is too short")

    middle_frame_timestamp = time_stamp[len(time_stamp) // 2]
    time_centered = [time_of_window - (time_stamp[-1] - t) for t in time_stamp]
    # print(len(time_centered))
    # print(len(y))
    # print(time_centered)
    # interpolate
    f = interpolate.interp1d(time_centered, y, kind=kind, axis=0)
    fixed_time_stamp = np.linspace(0.0, time_centered[-1], 31, endpoint=True)
    fixed_y = f(fixed_time_stamp)

    return middle_frame_timestamp, fixed_y
