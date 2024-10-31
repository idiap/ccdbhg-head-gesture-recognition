# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import os
import pickle
from typing import Dict

import cv2
import mediapipe as mp
import numpy as np
import torch
from matplotlib import pyplot as plt
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in face_landmarks
            ]
        )

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [
        face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes
    ]
    face_blendshapes_scores = [
        face_blendshapes_category.score for face_blendshapes_category in face_blendshapes
    ]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(
        face_blendshapes_ranks,
        face_blendshapes_scores,
        label=[str(x) for x in face_blendshapes_ranks],
    )
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel("Score")
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


############################################################################################################
# DRAWING UTILS HEAD GESTURE
############################################################################################################


def draw_dash_line(frame, start, end, color, thickness=1):
    """
    Draw a dashed line
    """
    dir_vec = np.array(end) - np.array(start)
    norm_vec = np.linalg.norm(dir_vec)
    dir_vec = dir_vec / norm_vec
    interval_nb = 14  # odd number
    interval_size = norm_vec / interval_nb
    points = np.linspace(0, norm_vec, interval_nb)
    points = points[::2]

    for point in points:
        cv2.line(
            frame,
            (int(start[0] + point * dir_vec[0]), int(start[1] + point * dir_vec[1])),
            (
                int(start[0] + (point + interval_size) * dir_vec[0]),
                int(start[1] + (point + interval_size) * dir_vec[1]),
            ),
            color,
            thickness,
        )

    return frame


class DrawHG:
    def __init__(self, clip_database_path=None):
        self.to_plot = ["pred", "proba", "gt"]
        self.label_to_name = {0: "None", 1: "Nod", 2: "Shake", 3: "Tilt", 4: "Turn", 5: "Up/Down"}
        self.label_color = {
            "Nod": np.array([0.0, 255.0, 0.0]),
            "Up/Down": np.array([255.0, 0.0, 255.0]),
            "Shake": np.array([0.0, 127.5, 255.0]),
            "Turn": np.array([255.0, 127.5, 0.0]),
            "Tilt": np.array([127.5, 191.25, 127.5]),
        }
        self.window_to_plot = 121
        self.thickness_line = 8
        self.thickness_text = 4
        self.fontscale = 2

        # to be set for each drawing
        self.image_height = None
        self.image_width = None
        self.plot_size_h = None

        # self.video_writer = VideoWriterIO( fps = 30 )

        # load the image and clip database

        with open(clip_database_path, "rb") as f:
            self.clip_database = pickle.load(f)

    def set_video_reader(self, path_video):
        # self.video_reader = VideoReader(path_video)
        pass

    def set_image_size(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height
        self.plot_size_h = int(0.35 * self.image_height)

    def reset_draw(self):
        self.image_width = None
        self.image_height = None
        self.plot_size_h = None
        self.video_reader = None

    def load_image(self, frame):
        return self.video_reader[frame - 1].asnumpy()

    def write_video(self, image_stream, path_output):
        print(image_stream.shape)
        # write the video
        self.video_writer(image_stream, path_output)

    def draw_head_gesture(self, hg_pred: Dict, output_path: str):
        # load the orginal image :
        # TODO

        image_stream = []
        # for each image draw the prediction
        frames_id = hg_pred["frame_id"]
        clip_key = f"clip_{hg_pred['video_id'][0]:08d}"
        clip_info = self.clip_database[clip_key]
        # print(clip_key)
        # print(clip_info)
        self.set_image_size(clip_info["other"]["video_width"], clip_info["other"]["video_height"])
        self.set_video_reader(clip_info["clip_path"])

        sequence_to_plot = {
            k: self.output_to_dict_plot(hg_pred[k], k) for k in self.to_plot if k in hg_pred
        }

        for i, frame_id in enumerate(frames_id):
            image = self.load_image(frame_id)  # todo load the image based on the id
            image = self.draw_legend(image)
            for plot in self.to_plot:
                if plot in hg_pred:
                    sub_sequence_to_plot = self.select_sequence(sequence_to_plot[plot], i)
                    image = self.draw_prediction(image, sub_sequence_to_plot)
            image_stream.append(image)

        image_stream = np.stack(image_stream, axis=0)
        print(image_stream.shape)
        self.write_video(image_stream, output_path)
        self.reset_draw()

    def output_to_dict_plot(self, output, mode):
        output_tensor = torch.tensor(output)
        output_pad = torch.nn.functional.pad(
            output_tensor, (self.window_to_plot // 2, self.window_to_plot // 2), value=0
        )
        one_hot = torch.nn.functional.one_hot(
            output_pad, num_classes=len(self.label_to_name)
        ).numpy()

        # create a dict to plot
        if mode == "pred":
            dict_to_plot = {
                self.label_to_name[i]: one_hot[:, i] * 0.5 for i in range(len(self.label_to_name))
            }
        else:
            dict_to_plot = {
                self.label_to_name[i]: one_hot[:, i] for i in range(len(self.label_to_name))
            }
        # drop the none label
        dict_to_plot.pop("None")
        return dict_to_plot

    def select_sequence(self, sequence_to_plot, index):
        # select the sequence to plot
        return {
            k: v[
                (index + self.window_to_plot // 2)
                - self.window_to_plot // 2 : (index + self.window_to_plot // 2)
                + self.window_to_plot // 2
                + 1
            ]
            for k, v in sequence_to_plot.items()
        }

    def draw_legend(self, image):
        # draw the legend on the image
        # TODO
        space = 55
        draw_text = lambda frame, text, pos, color: cv2.putText(
            frame,
            text,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.fontscale,
            color,
            self.thickness_text,
            cv2.LINE_AA,
        )
        for j, (key, color) in enumerate(self.label_color.items()):
            draw_text(image, key, (10, 5 + space * (j + 1)), color)

        frames_to_pix_width = 15 * self.image_width // (self.window_to_plot)
        draw_dash_line(
            image,
            (self.image_width // 2, self.image_height),
            (self.image_width // 2, int(self.image_height - self.plot_size_h)),
            (255, 255, 255),
            2,
        )
        draw_dash_line(
            image,
            (self.image_width // 2 - frames_to_pix_width, self.image_height),
            (
                self.image_width // 2 - frames_to_pix_width,
                int(self.image_height - self.plot_size_h),
            ),
            (255, 255, 255),
            2,
        )
        draw_dash_line(
            image,
            (self.image_width // 2 + frames_to_pix_width, self.image_height),
            (
                self.image_width // 2 + frames_to_pix_width,
                int(self.image_height - self.plot_size_h),
            ),
            (255, 255, 255),
            2,
        )
        return image

    def draw_prediction(self, image, sequence_to_plot):
        for k, v in sequence_to_plot.items():
            # value_to_plot = (value_to_plot - self.dict_to_plot_info[key]['lower_bound'])/(self.dict_to_plot_info[key]['upper_bound'] - self.dict_to_plot_info[key]['lower_bound'])
            value_to_plot = (
                self.image_height - v * self.plot_size_h
            )  # the y axes is reversed in opencv fix min value to bottom of the images
            value_to_plot_x = np.linspace(0, self.image_width, len(value_to_plot), dtype=int)
            value_to_plot_y = value_to_plot.astype(int)

            for i in range(len(value_to_plot_x) - 1):
                cv2.line(
                    image,
                    (value_to_plot_x[i], value_to_plot_y[i]),
                    (value_to_plot_x[i + 1], value_to_plot_y[i + 1]),
                    self.label_color[k],
                    self.thickness_line,
                )

        return image
