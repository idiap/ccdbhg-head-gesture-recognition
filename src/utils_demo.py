# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import hashlib
import os
from dataclasses import dataclass, field
from typing import Sequence
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# tracking
from motpy import Detection, MultiObjectTracker, NpImage

from src.classifier.yunet import YuNet
from src.inference import HeadGestureInference
from src.utils import get_head_pose, get_landmarks, time_interpolation
from src.utils_metrics import smooth_pred


def get_events(y, marge=7, background_label=0):
    """
    Get events from a given frame-wise prediction excluding background label."""
    events = []
    current_event = {"start": None, "end": None, "label": None}

    for i in range(len(y)):
        if y[i] != background_label and current_event["start"] is None:
            current_event["start"] = i
            current_event["label"] = y[i]
        elif y[i] != current_event["label"] and current_event["start"] is not None:
            current_event["end"] = i - 1
            events.append(current_event.copy())
            if y[i] != background_label:
                current_event["start"] = i
                current_event["label"] = y[i]
            else:
                current_event = {"start": None, "end": None, "label": None}

    if current_event["start"] is not None:
        current_event["end"] = len(y) - 1
        events.append(current_event.copy())

    # remove small events
    if (marge is not None) and (marge > 0):
        for e in events:
            if (e["end"] - e["start"] + 1) <= marge:
                events.remove(e)

    return events


@dataclass
class Location:
    """Data class for face location.

    Attributes:
        x1 (int): x1 coordinate
        x2 (int): x2 coordinate
        y1 (int): y1 coordinate
        y2 (int): y2 coordinate
    """

    x1: int = field(default=0)
    x2: int = field(default=0)
    y1: int = field(default=0)
    y2: int = field(default=0)

    def form_square(self) -> None:
        """Form a square from the location.

        Returns:
            None
        """
        height = self.y2 - self.y1
        width = self.x2 - self.x1

        if height > width:
            diff = height - width
            self.x1 = self.x1 - int(diff / 2)
            self.x2 = self.x2 + int(diff / 2)
        elif height < width:
            diff = width - height
            self.y1 = self.y1 - int(diff / 2)
            self.y2 = self.y2 + int(diff / 2)
        else:
            pass

    def expand(self, amount: float) -> None:
        """Expand the location while keeping the center.

        Args:
            amount (float): Amount to expand the location by in multiples of the original size.


        Returns:
            None
        """
        assert amount >= 0, "Amount must be greater than or equal to 0."
        # if amount != 0:
        #     self.x1 = self.x1 - amount
        #     self.y1 = self.y1 - amount
        #     self.x2 = self.x2 + amount
        #     self.y2 = self.y2 + amount
        if amount != 0.0:
            self.x1 = self.x1 - int((self.x2 - self.x1) / 2 * amount)
            self.y1 = self.y1 - int((self.y2 - self.y1) / 2 * amount)
            self.x2 = self.x2 + int((self.x2 - self.x1) / 2 * amount)
            self.y2 = self.y2 + int((self.y2 - self.y1) / 2 * amount)

    def cast_to_int(self):
        self.x1 = int(self.x1)
        self.x2 = int(self.x2)
        self.y1 = int(self.y1)
        self.y2 = int(self.y2)


@dataclass
class Event:
    start: int
    end: int
    label: str
    confidence: float
    track_id: str

    def __post_init__(self):
        # generate a unique id for the event random
        self.idx = f"{self.end}_{self.label}_{self.track_id}"


@dataclass
class Face:
    """Data class for face attributes.

    Attributes:
        indx (int): Index of the face.
        loc (Location): Location of the face in the image.
        tensor (torch.Tensor): Face tensor.
        ratio (float): Ratio of the face area to the image area.
        preds (Dict[str, Prediction]): Predictions of the face given by predictor set.
    """

    indx: str
    loc: Location
    conf: float
    timestamp: int
    prediction: dict = field(default_factory=dict)
    gesture: dict = field(default_factory=dict)

    def __post_init__(self):
        self.loc.form_square()
        self.loc.expand(0.2)
        self.loc.cast_to_int()
        self.center = (int((self.loc.x1 + self.loc.x2) / 2), int((self.loc.y1 + self.loc.y2) / 2))
        self.area = (self.loc.x2 - self.loc.x1) * (self.loc.y2 - self.loc.y1)
        self.gesture = {"head_gesture": "none", "confidence": 1}

    def add_prediction(self, prediction):
        self.prediction.update(prediction)


class MediapipePredictor:
    def __init__(self):
        path_media_pipe = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "model_checkpoints/face_landmarker_v2_with_blendshapes.task",
        )
        base_options = python.BaseOptions(model_asset_path=path_media_pipe)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process_face(self, frame, face: Face):
        w, h = frame.shape[1], frame.shape[0]
        frame_face = frame[
            max(0, face.loc.y1) : min(h, face.loc.y2), max(0, face.loc.x1) : min(w, face.loc.x2)
        ]
        if frame_face.shape[0] <= 10 or frame_face.shape[1] <= 10:
            return {}

        frame_face_width = frame_face.shape[1]
        frame_face_height = frame_face.shape[0]
        corner = (max(0, face.loc.x1), max(0, face.loc.y1))

        # process the face
        # resize the face
        frame_face = frame_face.astype(np.uint8)
        frame_face = cv2.resize(frame_face, (224, 224))

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_face)
        detection_result = self.detector.detect(mp_image)

        if len(detection_result.facial_transformation_matrixes) == 0:
            return {}

        rotation_matrix = detection_result.facial_transformation_matrixes[0]
        landmarks_3d = detection_result.face_landmarks[0]

        # unormalize the landmarks
        landmarks_3d = np.array(
            [
                [
                    lm.x * frame_face_width + corner[0],
                    lm.y * frame_face_height + corner[1],
                    lm.z * frame_face_width,
                ]
                for lm in landmarks_3d
            ]
        )

        return {"ldmks": get_landmarks(landmarks_3d), "head_pose": get_head_pose(rotation_matrix)}


class FaceTracker:
    """Handle different face detection and tracking algorithm"""

    def __init__(
        self,
        dt=1 / 15.0,
        model_spec={
            "order_pos": 2,
            "dim_pos": 2,
            "order_size": 0,
            "dim_size": 2,
            "q_var_pos": 5000.0,
            "r_var_pos": 0.2,
        },
    ):
        self.dt = dt
        self.face_trackers = []

        self.tracker = MultiObjectTracker(dt=dt, model_spec=model_spec)
        self.tracks_store = {}

    def clean_tracks(self, timestamp):
        # remove old tracks
        remove_tracks = []
        for track_id, track in self.tracks_store.items():
            if timestamp != track[-1].timestamp:
                remove_tracks.append(track_id)

        for track_id in remove_tracks:
            self.tracks_store.pop(track_id)

    def update(self, detections, timestamp):
        # update the tracker
        self.tracker.step(detections)
        tracks = self.tracker.active_tracks(min_steps_alive=5)
        # convert tracks to face object

        for track in tracks:
            face = Face(
                indx=track.id,
                loc=Location(x1=track.box[0], x2=track.box[2], y1=track.box[1], y2=track.box[3]),
                conf=track.score,
                timestamp=timestamp,
            )
            if track.id not in self.tracks_store:
                self.tracks_store[track.id] = [face]
            else:
                self.tracks_store[track.id] += [face]

        self.clean_tracks(timestamp)

        return tracks

    def get_tracks(self):
        tracks_id = []
        # rank them by mean area of the last 10 frames
        for track_id, track in self.tracks_store.items():
            len_track = len(track)
            mean_area = np.mean([face.area for face in track[-min(10, len_track) :]])
            tracks_id.append((track_id, mean_area))

        tracks_id = sorted(tracks_id, key=lambda x: x[1], reverse=True)
        len_tracks_id = len(tracks_id)
        return [track_id for track_id, _ in tracks_id[: min(4, len_tracks_id)]]


class TrackHandler:
    def __init__(self, tcks: FaceTracker):
        self.tcks = tcks
        self.label_names = ["none", "nod", "shake", "tilt", "turn", "up/down"]
        self.label_id_to_name = {i: label for i, label in enumerate(self.label_names)}
        self.name_to_label_id = {v: k for k, v in self.label_id_to_name.items()}
        self.last_event_queue = {}

    def add_track_prediction(self, out_track):
        for track_id, pred in out_track.items():
            track_time_to_face = {
                face.timestamp: i for i, face in enumerate(self.tcks.tracks_store[track_id])
            }
            if pred["timestamp"] in track_time_to_face:
                self.tcks.tracks_store[track_id][track_time_to_face[pred["timestamp"]]].gesture = {
                    "head_gesture": pred["head_gesture"],
                    "confidence": pred["confidence"],
                }

    def last_event(self):
        valid_tracks = self.tcks.get_tracks()
        for track_id in valid_tracks:
            track = self.tcks.tracks_store[track_id]
            len_track = len(track)
            if len_track < 30:
                continue
            track_face = [face for face in track[-min(60, len_track) :]]
            gesture_track = [face.gesture for face in track_face]
            gesture_track_id = [
                self.name_to_label_id[gesture["head_gesture"]] for gesture in gesture_track
            ]
            # smooth the gesture
            gesture_track_id = smooth_pred(gesture_track_id, 3)
            # get the event

            events = get_events(gesture_track_id, 6)
            if len(events) == 0:
                continue
            # get the last event
            event = events[-1]
            # print(gesture_track_id)
            # make sure the event is done
            if event["end"] < len(gesture_track_id) - 10:
                event = Event(
                    start=track_face[event["start"]].timestamp,
                    end=track_face[event["end"]].timestamp,
                    label=self.label_id_to_name[event["label"]],
                    confidence=round(
                        np.mean(
                            [
                                gesture["confidence"]
                                for gesture in gesture_track[event["start"] : event["end"]]
                            ]
                        ),
                        2,
                    ),
                    track_id=track_id,
                )
                if track_id not in self.last_event_queue:
                    self.last_event_queue[track_id] = event
                else:
                    if self.last_event_queue[track_id].idx != event.idx:
                        self.last_event_queue[track_id] = event

    def get_last_event(self):
        self.last_event()
        return self.last_event_queue


WEIGHTS_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
WEIGHTS_PATH = "/tmp/opencv_face_detector.caffemodel"
CONFIG_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CONFIG_PATH = "/tmp/deploy.prototxt"


class FaceDetectorCV2:
    def __init__(
        self,
        weights_url: str = WEIGHTS_URL,
        weights_path: str = WEIGHTS_PATH,
        config_url: str = CONFIG_URL,
        config_path: str = CONFIG_PATH,
        conf_threshold: float = 0.7,
    ) -> None:
        if not os.path.isfile(weights_path) or not os.path.isfile(config_path):
            urlretrieve(weights_url, weights_path)
            urlretrieve(config_url, config_path)

        self.net = cv2.dnn.readNetFromCaffe(config_path, weights_path)

        # specify detector hparams
        self.conf_threshold = conf_threshold

    def process_image(self, image: NpImage) -> Sequence[Detection]:
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        self.net.setInput(blob)
        detections = self.net.forward()

        # convert output from OpenCV detector to tracker expected format [xmin, ymin, xmax, ymax]
        out_detections = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                xmin = int(detections[0, 0, i, 3] * image.shape[1])
                ymin = int(detections[0, 0, i, 4] * image.shape[0])
                xmax = int(detections[0, 0, i, 5] * image.shape[1])
                ymax = int(detections[0, 0, i, 6] * image.shape[0])
                out_detections.append(Detection(box=[xmin, ymin, xmax, ymax], score=confidence))

        return out_detections


class FaceDetectorYUNET:
    def __init__(self, conf_threshold: float = 0.5) -> None:
        backend_target_pairs = [
            [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
            [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA],
            [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16],
            [cv2.dnn.DNN_BACKEND_TIMVX, cv2.dnn.DNN_TARGET_NPU],
            [cv2.dnn.DNN_BACKEND_CANN, cv2.dnn.DNN_TARGET_NPU],
        ]

        backend_id = backend_target_pairs[0][0]
        target_id = backend_target_pairs[0][1]

        # Instantiate YuNet
        self.face_detector = YuNet(
            modelPath=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "model_checkpoints/face_detection_yunet_2023mar.onnx",
            ),
            inputSize=[320, 320],
            confThreshold=conf_threshold,
            nmsThreshold=0.3,
            topK=2000,
            backendId=backend_id,
            targetId=target_id,
        )

    def process_image(self, image) -> Sequence[Detection]:
        # detect the face first
        h, w, _ = image.shape

        # Detect faces
        self.face_detector.setInputSize([w, h])
        face_detected = self.face_detector.infer(image)
        # convert output from OpenCV detector to tracker expected format [xmin, ymin, xmax, ymax]

        out_detections = []
        for det in face_detected:
            bbox = det[0:4].astype(np.int32)
            confidence = det[-1]

            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[0] + bbox[2])
            ymax = int(bbox[1] + bbox[3])
            out_detections.append(Detection(box=[xmin, ymin, xmax, ymax], score=confidence))

        return out_detections


class HGPredictor:
    def __init__(self, device):
        self.device = device
        exp_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "model_checkpoints/cnn_lmk_hp"
        )
        self.predictor = HeadGestureInference(exp_folder, device)

    def preprocess(self, face_tracker, tracks_id):
        """
        Preprocess the tracks to get the input for the model
        """
        min_track_len = 50
        process_track = {}
        for track_id in tracks_id:
            len_track = len(face_tracker.tracks_store[track_id])
            # filter such that the minimum track lenght is 30
            if len_track < min_track_len:
                process_track[track_id] = {
                    "input": None,
                    "timestamp": face_tracker.tracks_store[track_id][-1].timestamp,
                }

            else:
                faces = face_tracker.tracks_store[track_id][-min_track_len:]
                time_stream = []
                data_stream = []
                for face in faces:
                    if "ldmks" not in face.prediction or "head_pose" not in face.prediction:
                        continue
                    else:
                        time_stream.append(face.timestamp)
                        data_stream.append(
                            list(face.prediction["ldmks"]) + list(face.prediction["head_pose"])
                        )

                if len(time_stream) > 20:
                    middle_frame_timestamp, fixed_y = time_interpolation(time_stream, data_stream)
                    x_input = {
                        "landmarks_detector": fixed_y[:, : 3 * 5],
                        "head_pose_detector": fixed_y[:, 3 * 5 :],
                    }

                    x_input = self.predictor.preprocess(None, None, x_input)  # 1,18,31 (b,d,t)
                    process_track[track_id] = {
                        "input": x_input,
                        "timestamp": middle_frame_timestamp,
                    }
                else:
                    process_track[track_id] = {
                        "input": None,
                        "timestamp": face_tracker.tracks_store[track_id][-1].timestamp,
                    }

        return process_track

    def process(self, face_tracker, tracks_id):
        out_track = {}
        process_track = self.preprocess(face_tracker, tracks_id)

        ids_none = [
            track_id for track_id, track in process_track.items() if track["input"] is None
        ]
        ids_valid = [
            track_id for track_id, track in process_track.items() if track["input"] is not None
        ]

        if len(ids_valid) > 0:
            # stack valide track
            x_tensor = torch.cat(
                [process_track[track_id]["input"] for track_id in ids_valid], dim=0
            )

            with torch.no_grad():
                y_pred, _ = self.predictor.model(x_tensor.to(self.device))
                probs = F.softmax(y_pred, dim=1)

            for i in range(len(ids_valid)):
                track_id = ids_valid[i]
                idx_pred = probs.detach().cpu()[i].argmax()
                confidence = probs.detach().cpu()[i][idx_pred]
                gesture = self.predictor.label_name[idx_pred]
                out_track[track_id] = {
                    "head_gesture": gesture,
                    "confidence": confidence,
                    "timestamp": process_track[track_id]["timestamp"],
                }

        for i in range(len(ids_none)):
            track_id = ids_none[i]
            out_track[track_id] = {
                "head_gesture": "none",
                "confidence": 1,
                "timestamp": process_track[track_id]["timestamp"],
            }

        return out_track


def string_to_color(input_string):
    # Generate a hash from the string
    hash_object = hashlib.md5(input_string.encode())
    # Convert the hash to an integer and take the first 6 characters for RGB
    hex_color = hash_object.hexdigest()[:6]
    # Convert the hex value to an RGB tuple
    rgb_color = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    return rgb_color


class Visualizer:
    def __init__(
        self,
        draw_bbox=True,
        draw_landmarks=True,
        draw_head_gesture=True,
        fontsize=2,
        space_legend=55,
    ):
        self.draw_bbox = draw_bbox
        self.draw_landmarks = draw_landmarks
        self.draw_head_gesture = draw_head_gesture
        self.space_legend = space_legend
        self.fontsize = fontsize

        self.label_name = ["None", "Nod", "Shake", "Tilt", "Turn", "Up/down"]
        self.label_color = {
            "Nod": np.array([0.0, 200.0, 0.0]),
            "Shake": np.array([0.0, 150.0, 255.0]),
            "Turn": np.array([255.0, 150.0, 0.0]),
            "Tilt": np.array([0.0, 0.0, 200.0]),
            "Up/down": np.array([200.0, 0.0, 200.0]),
            "None": np.array([0.0, 0.0, 0.0]),
        }

    def draw_legend(self, image):
        # Draw a legend for the head gesture
        for j, (key, color) in enumerate(self.label_color.items()):
            cv2.putText(
                image,
                key,
                (10, 20 + self.space_legend * (j + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.fontsize,
                color,
                int(self.fontsize * 1.6),
            )
        return image

    def process(self, image, face_tracker: FaceTracker, output_track):
        get_valid_track = face_tracker.get_tracks()

        for track_id in get_valid_track:
            track = face_tracker.tracks_store[track_id]
            bbox = track[-1].loc
            color = string_to_color(track_id)

            if self.draw_bbox:
                cv2.rectangle(
                    image, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, int(3 * self.fontsize)
                )

            if self.draw_landmarks:
                if "ldmks" in track[-1].prediction:
                    landmarks = track[-1].prediction["ldmks"]
                    landmarks = np.array(landmarks).reshape(-1, 3)
                    for i in range(0, 5):
                        cv2.circle(
                            image, (int(landmarks[i, 0]), int(landmarks[i, 1])), 5, (0, 255, 0), -1
                        )

            if self.draw_head_gesture:
                self.draw_legend(image)
                if track_id in output_track:
                    head_gesture = output_track[track_id]
                    cv2.putText(
                        image,
                        f"{head_gesture['head_gesture'].capitalize()}: {head_gesture['confidence']:.2f} ",
                        (bbox.x1, bbox.y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.fontsize,
                        self.label_color[head_gesture["head_gesture"].capitalize()],
                        int(self.fontsize),
                    )

        return image
