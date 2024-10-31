# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np
import torch
from einops import rearrange
from scipy.spatial.transform import Rotation as R_

"""
Code for the data processing
"""


def compute_relative_torch(x, m):
    x = torch.from_numpy(x)
    x = x.permute(0, 2, 1)
    first_value = x[:, :, 0:1]
    # Create new values by repeating the first value k times
    new_values = first_value.expand(-1, -1, m)
    # Concatenate the new values with the original input tensor
    x = torch.cat((new_values, x), dim=2)
    x = x[:, :, m:] - x[:, :, :-m]
    x = x.permute(0, 2, 1).numpy()

    return x


def rotation_maxtrix_from_euler_angle_xyz(euler_angle):
    """
    Gives the rotation matrix from a euler angle XYZ representation.
    """
    return R_.from_euler("xyz", euler_angle).as_matrix()


def euler_angle_xyz_from_rotation_matrix(rotation_matrix, degrees=False):
    """
    Gives the euler angle as a XYZ representation of the rotation matrix.
    """
    return R_.from_matrix(rotation_matrix).as_euler("xyz", degrees=degrees)


def euler_hp_to_rotation_matrix(x):
    """x is yaw, roll, pitch"""
    assert x.shape[1] == 3, "Invalid shape"
    matrix = x.copy()
    x_tmp = matrix[:, [2, 0, 1]]
    rot = rotation_maxtrix_from_euler_angle_xyz(x_tmp)
    assert rot.shape[0] == x.shape[0]
    assert rot.shape[1] == rot.shape[2] == 3
    return rot


def rotation_matrix_to_euler_hp(x):
    """output is yaw, roll, pitch"""
    assert x.ndim == 3
    assert x.shape[1] == x.shape[2] == 3
    rot = x.copy()
    euler = euler_angle_xyz_from_rotation_matrix(rot)
    assert euler.shape[0] == x.shape[0]
    assert euler.shape[1] == 3

    return euler[:, [1, 2, 0]]


def pitchyaw_to_rotation(a):
    assert a.shape[1] == 2, "Invalid shape"

    cos = np.cos(a)
    sin = np.sin(a)
    ones = np.ones_like(cos[:, 0])
    zeros = np.zeros_like(cos[:, 0])

    matrices_1 = np.array(
        [[ones, zeros, zeros], [zeros, cos[:, 0], sin[:, 0]], [zeros, -sin[:, 0], cos[:, 0]]]
    ).transpose(2, 0, 1)

    matrices_2 = np.array(
        [[cos[:, 1], zeros, sin[:, 1]], [zeros, ones, zeros], [-sin[:, 1], zeros, cos[:, 1]]]
    ).transpose(2, 0, 1)

    matrices = np.matmul(matrices_2, matrices_1)
    return matrices


def vector_to_pitchyaw(a):
    if a.shape[1] == 2:
        return a
    elif a.shape[1] == 3:
        a = a.reshape(-1, 3)
        norm_a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-7)
        return np.stack(
            [
                np.arcsin(norm_a[:, 1]),
                np.arctan2(norm_a[:, 0], norm_a[:, 2]),
            ],
            axis=1,
        )
    else:
        raise ValueError("Do not know how to convert array of shape %s" % str(a.shape))


def pitchyaw_to_vector(pitchyaws):
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])

    return out


def rotation_to_vector(a):
    assert a.ndim == 3
    assert a.shape[1] == a.shape[2] == 3

    frontal_vector = np.concatenate(
        [
            np.zeros_like(a[:, :2, 0]).reshape(-1, 2, 1),
            np.ones_like(a[:, 2, 0]).reshape(-1, 1, 1),
        ],
        axis=1,
    )
    return np.matmul(a, frontal_vector)


def rotation_to_pitchyaw(a):
    return vector_to_pitchyaw(rotation_to_vector(a))


def compute_relative_rotation_matrix_head_pose(x, m):
    # x head pose is yaw, roll, pitch
    assert x.shape[1] == 3, "Invalid shape"
    matrix = x.copy()
    for i in range(x.shape[0]):
        t_old = max(i - m, 0)
        t = i
        rot_t_old = rotation_maxtrix_from_euler_angle_xyz(x[t_old, [2, 0, 1]])
        rot_t = rotation_maxtrix_from_euler_angle_xyz(x[t, [2, 0, 1]])
        # rot = (rot_t.T)@rot_t_old
        rot = (rot_t_old.T) @ rot_t
        euler = euler_angle_xyz_from_rotation_matrix(rot)
        matrix[t] = np.array([euler[1], euler[2], euler[0]])
    return matrix


def compute_relative_rotation_matrix(x, m):
    first_matrix = x[:1]  # (1,3,3)
    prepend = np.repeat(first_matrix, m, axis=0)  # (m,3,3)
    assert prepend.shape == (m, 3, 3)
    X_tmp = np.concatenate((prepend, x), axis=0)
    X_transpose = np.transpose(X_tmp, (0, 2, 1))
    return np.matmul(X_transpose[m:], X_tmp[:-m])


def compute_relative(X, m):
    """compute the relative position of the time series"""
    if m == 0:
        return X
    first_col = X[:1]
    prepend = np.repeat(first_col, m, axis=0)
    X_tmp = np.concatenate((prepend, X), axis=0)
    return X_tmp[m:] - X_tmp[:-m]


def compute_relative_landmark_invariant(x, m, R, x_hp=None, method=None):
    x_reshape = rearrange(
        x, "n (n_landmarks d_landmarks) -> n n_landmarks d_landmarks", d_landmarks=3
    )

    landmarks_method = method  # 'without_head_center'
    # print('landmarks option : ' + landmarks_method)
    if landmarks_method == "with_head_center":
        # first option with head pose center
        head_center = np.mean(x_reshape, axis=1, keepdims=True)
        head_center = np.repeat(head_center, x_reshape.shape[1], axis=1)
        landmarks_center = (x_reshape - head_center).transpose(0, 2, 1)
        landmarks_center = np.matmul(np.repeat(R.T[None], x.shape[0], axis=0), landmarks_center)
        landmarks_center = landmarks_center.transpose(0, 2, 1)
        landmarks_center = landmarks_center + head_center

    elif landmarks_method == "with_head_center_no_add":
        # first option with head pose center
        head_center = np.mean(x_reshape, axis=1, keepdims=True)
        head_center = np.repeat(head_center, x_reshape.shape[1], axis=1)
        landmarks_center = (x_reshape - head_center).transpose(0, 2, 1)
        landmarks_center = np.matmul(np.repeat(R.T[None], x.shape[0], axis=0), landmarks_center)
        landmarks_center = landmarks_center.transpose(0, 2, 1)

    elif landmarks_method == "without_head_center":
        # second option wihtout head pose center
        landmarks_center = x_reshape.transpose(0, 2, 1)
        landmarks_center = np.matmul(np.repeat(R.T[None], x.shape[0], axis=0), landmarks_center)
        landmarks_center = landmarks_center.transpose(0, 2, 1)

    elif landmarks_method == "wrt_prev_head":
        # Third option
        assert x_hp.shape[1] == 3, "Invalid shape"
        landmarks_center = x_reshape.copy()
        for i in range(x_reshape.shape[0]):
            t_old = max(i - m, 0)
            t = i
            # center_head = np.mean(x_reshape[t_old],axis=0)
            # center_head_t = np.mean(x_reshape[t],axis=0)
            # ldmk_center = x_reshape[t] - center_head #5x3
            rot_t_old = rotation_maxtrix_from_euler_angle_xyz(x_hp[t_old, [2, 0, 1]])
            landmarks_center[t] = np.matmul(rot_t_old.T, (x_reshape[t] - x_reshape[t_old]).T).T

    elif landmarks_method == "wrt_prev_head_with_center":
        # Third option
        assert x_hp.shape[1] == 3, "Invalid shape"
        landmarks_center = x_reshape.copy()
        for i in range(x_reshape.shape[0]):
            t_old = max(i - m, 0)
            t = i
            center_head_old = np.mean(x_reshape[t_old], axis=0)
            center_head_t = np.mean(x_reshape[t], axis=0)
            # ldmk_center = x_reshape[t] - center_head #5x3
            rot_t_old = rotation_maxtrix_from_euler_angle_xyz(x_hp[t_old, [2, 0, 1]])
            landmarks_center[t] = np.matmul(
                rot_t_old.T,
                ((x_reshape[t] - center_head_t) - (x_reshape[t_old] - center_head_old)).T,
            ).T

    elif landmarks_method == "wrt_prev_head_with_center_add":
        # fourth option
        assert x_hp.shape[1] == 3, "Invalid shape"
        landmarks_center = x_reshape.copy()
        for i in range(x_reshape.shape[0]):
            t_old = max(i - m, 0)
            t = i
            center_head_old = np.mean(x_reshape[t_old], axis=0)
            center_head_t = np.mean(x_reshape[t], axis=0)
            # ldmk_center = x_reshape[t] - center_head #5x3
            rot_t_old = rotation_maxtrix_from_euler_angle_xyz(x_hp[t_old, [2, 0, 1]])
            landmarks_center[t] = np.matmul(
                rot_t_old.T,
                ((x_reshape[t] - center_head_t) - (x_reshape[t_old] - center_head_old)).T,
            ).T - (center_head_t - center_head_old)

    else:
        print("landmarks option : " + landmarks_method + " is not a valid option")
        raise NotImplementedError

    out = rearrange(
        landmarks_center, "n n_landmarks d_landmarks -> n (n_landmarks d_landmarks)", d_landmarks=3
    )

    # Finally compute the relative based on the new landmarks
    if (
        landmarks_method == "with_respect_to_previous_head"
        or landmarks_method == "with_respect_to_average_head"
    ):
        return out
    else:
        out = compute_relative(out, m)
        return out


def compute_avg_rotation_matrix(x):
    assert x.shape[1] == 3, "Invalid shape"
    hp_mean = np.mean(x, axis=0, keepdims=True)
    avg_rotation = euler_hp_to_rotation_matrix(hp_mean)[0]
    assert avg_rotation.shape[0] == avg_rotation.shape[1] == 3, "Invalid shape"
    return avg_rotation


def compute_relative_invariant(x, m, x_hp, modality, gaze_type, method=None):
    """
    compute the relative position of the time series
    here specifically for rotation matrix
    """
    if m == 0:
        return x
    # TODO parallelize this for loop
    if modality == "head_pose":
        # x = euleur_to_rotation_matrix_head_pose(x,m)
        hp_rotation = euler_hp_to_rotation_matrix(np.array(x))
        hp_rotation_relative = compute_relative_rotation_matrix(hp_rotation, m)
        out = rotation_matrix_to_euler_hp(hp_rotation_relative)
    elif modality == "gaze":
        if gaze_type == "gaze_camera":
            gaze_rotation = pitchyaw_to_rotation(np.array(x))
            assert gaze_rotation.shape[1] == 3, "Invalid shape"
            assert gaze_rotation.shape[2] == 3, "Invalid shape"
            gaze_rotation_relative = compute_relative_rotation_matrix(gaze_rotation, m)
            out = rotation_to_pitchyaw(gaze_rotation_relative)
        elif gaze_type == "gaze_head":
            # since gaze is in head coordinate system
            out = compute_relative(np.array(x), m)
        else:
            raise NotImplementedError(f"Wrong gaze_type in Relative: {gaze_type}")

    elif modality == "landmarks":
        R_avg = compute_avg_rotation_matrix(np.array(x_hp))
        out = compute_relative_landmark_invariant(np.array(x), m, R_avg, np.array(x_hp), method)
    else:
        raise NotImplementedError(f"Invalid modality in invariance {modality}")

    return out


def is_input(k):
    if "head_pose" in k or "landmarks" in k or "gaze" in k:
        return True
    return False
