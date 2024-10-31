# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import warnings

import numpy as np
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report
from torchmetrics import Metric
from torchmetrics.functional import confusion_matrix
from torchmetrics.utilities.data import dim_zero_cat

from .utils_metrics import (
    compute_event_overlap,
    compute_event_overlap_IOU,
    get_event_matching,
    get_events,
    get_transition_frames,
    smooth_pred,
)

# filter the warnings about ill-defined P,R and F1
warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)


class FrameClassification(Metric):
    def __init__(
        self,
        num_classes=6,
        smooth_marge=5,
        transition_marge=2,
        label_ids=[1, 2, 3, 4, 5],
        label_names=["Nod", "Shake", "Tilt", "Turn", "Up_down"],
    ) -> None:
        super().__init__(compute_on_cpu=True, compute_with_cache=False)
        self.num_classes = num_classes
        self.smooth_marge = smooth_marge
        self.transition_marge = transition_marge
        self.label_ids = label_ids
        self.label_names = label_names
        self.task = "multiclass"
        self.add_state("frame_pred", default=[], dist_reduce_fx="cat", persistent=True)
        self.add_state("frame_gt", default=[], dist_reduce_fx="cat", persistent=True)
        self.add_state("frame_id", default=[], dist_reduce_fx="cat", persistent=True)
        self.add_state("video_id", default=[], dist_reduce_fx="cat", persistent=True)

    def update(self, pred, gt, frame_id, video_id):
        self.frame_pred += pred.argmax(dim=1)
        self.frame_gt += gt
        self.frame_id += frame_id
        self.video_id += video_id

    def compute(self):
        if len(self.frame_pred) == 0:
            return {
                "frame_classification_report": None,
                "confusion_matrix_frame": None,
            }

        frame_pred = dim_zero_cat(self.frame_pred).cpu()
        frame_gt = dim_zero_cat(self.frame_gt).cpu()
        frame_id = dim_zero_cat(self.frame_id).cpu()
        video_id = dim_zero_cat(self.video_id).cpu()

        cleaned_frame_pred, cleaned_frame_gt = self._clean_prediction(
            frame_pred, frame_gt, frame_id, video_id
        )

        frame_classification_report = classification_report(
            cleaned_frame_gt,
            cleaned_frame_pred,
            labels=self.label_ids,
            target_names=self.label_names,
            output_dict=True,
        )

        return {
            "frame_classification_report": frame_classification_report,
            "confusion_matrix_frame": confusion_matrix(
                cleaned_frame_pred, cleaned_frame_gt, self.task, num_classes=self.num_classes
            ),
        }

    def _clean_prediction(self, frame_pred, frame_gt, frame_id, video_id):
        """
        Clean the prediction by removing the transition frames and smoothing the prediction
        """
        # Gather all the videos
        videos = torch.unique(video_id, sorted=False)
        cleaned_frame_pred = []
        cleaned_frame_gt = []
        for video in videos:
            videos_idx = torch.where(video_id == video)
            frames = frame_id[videos_idx]

            # Sort the frames
            frames, idx = torch.sort(frames)
            pred = frame_pred[videos_idx][idx].tolist()
            target = frame_gt[videos_idx][idx].tolist()

            # Smooth the prediction
            pred = smooth_pred(pred, self.smooth_marge)

            # Clean the transition frames
            if self.transition_marge is not None:
                target_events = get_events(target, self.smooth_marge, background_label=0)
                transition_target = get_transition_frames(
                    target, target_events, self.transition_marge
                )
                cleaned_frame_pred += [
                    pred[i] for i in range(len(pred)) if transition_target[i] == 0
                ]
                cleaned_frame_gt += [
                    target[i] for i in range(len(target)) if transition_target[i] == 0
                ]
                assert len(cleaned_frame_pred) == len(cleaned_frame_gt)
            else:
                cleaned_frame_pred += pred
                cleaned_frame_gt += target

        cleaned_frame_pred = torch.tensor(cleaned_frame_pred)
        cleaned_frame_gt = torch.tensor(cleaned_frame_gt)

        return cleaned_frame_pred, cleaned_frame_gt


class EventDetectionBase(Metric):
    def __init__(
        self,
        num_classes=6,
        smooth_marge=5,
        thresholds=[0.1, 0.25, 0.5],
        label_ids=[1, 2, 3, 4, 5],
        label_names=["Nod", "Shake", "Tilt", "Turn", "Up_down"],
        backgroud_class=0,
    ) -> None:
        super().__init__(compute_on_cpu=True, compute_with_cache=False)
        self.num_classes = num_classes
        self.smooth_marge = smooth_marge
        self.thresholds = thresholds
        self.label_ids = label_ids
        self.label_names = label_names
        self.background_class = backgroud_class
        self.task = "multiclass"
        self.add_state("frame_pred", default=[], dist_reduce_fx="cat", persistent=True)
        self.add_state("frame_gt", default=[], dist_reduce_fx="cat", persistent=True)
        self.add_state("frame_id", default=[], dist_reduce_fx="cat", persistent=True)
        self.add_state("video_id", default=[], dist_reduce_fx="cat", persistent=True)

    def update(self, pred, gt, frame_id, video_id):
        self.frame_pred += pred.argmax(dim=1)
        self.frame_gt += gt
        self.frame_id += frame_id
        self.video_id += video_id

    def compute(self):
        if len(self.frame_pred) == 0:
            return None

        frame_pred = dim_zero_cat(self.frame_pred).cpu()
        frame_gt = dim_zero_cat(self.frame_gt).cpu()
        frame_id = dim_zero_cat(self.frame_id).cpu()
        video_id = dim_zero_cat(self.video_id).cpu()

        pred_all_video, target_all_video = self.clean_prediction_event(
            frame_pred, frame_gt, frame_id, video_id
        )

        scores = {}
        for th in self.thresholds:
            scores[f"threshold_{th}"] = self.compute_score(pred_all_video, target_all_video, th)

        return scores

    def compute_score(self, pred_all_video, target_all_video, threshold):
        raise NotImplementedError

    def clean_prediction_event(self, frame_pred, frame_gt, frame_id, video_id):
        # Gather all the prediction and ground truth per video
        videos = torch.unique(video_id, sorted=False)
        target_all_video = [[] for _ in range(len(videos))]
        pred_all_video = [[] for _ in range(len(videos))]
        for video in videos:
            videos_idx = torch.where(video_id == video)
            frames = frame_id[videos_idx]
            # sort the frames
            frames, idx = torch.sort(frames)
            pred = frame_pred[videos_idx][idx].tolist()
            target = frame_gt[videos_idx][idx].tolist()

            pred = smooth_pred(pred, self.smooth_marge)

            target_events = get_events(
                target, self.smooth_marge, background_label=self.background_class
            )
            target_all_video[video] += target_events

            pred_events = get_events(
                pred, self.smooth_marge, background_label=self.background_class
            )
            pred_all_video[video] += pred_events

        return pred_all_video, target_all_video

    def pdv(self, num, denom):
        """
        Handls ZeroDivisionError
        """
        results = num / (denom + 1e-7)
        return np.nan_to_num(results)


class EventDetectionAssociation(EventDetectionBase):
    """Metric to evaluate the event detection using the association method.
    The event precission is computed by matching the predicted events with the ground truth events above a threshold.
    The event recall is computed by matching the ground truth events with the predicted events above a threshold.
    The event F1 is the harmonic mean of the event precission and recall.
    # "Head Nod Detection from a Full 3D Model"
    # https://openaccess.thecvf.com/content_iccv_2015_workshops/w12/papers/Chen_Head_Nod_Detection_ICCV_2015_paper.pdf
    # "CCDb-HG: Novel Annotations and Gaze-Aware Representations for Head Gesture Recognition"
    # https://www.idiap.ch/~odobez/publications/VuillecardFarkhondehVillamizarOdobez-IEEE-FG2024-HeadGesture.pdf

    Note: the result is in the same format as the classification report dict from sklearn
    """

    def __init__(
        self,
        num_classes=6,
        smooth_marge=5,
        threshold=[0.1, 0.25, 0.5],
        label_ids=[1, 2, 3, 4, 5],
        label_names=["Nod", "Shake", "Tilt", "Turn", "Up_down"],
        backgroud_class=0,
    ) -> None:
        super().__init__(
            num_classes, smooth_marge, threshold, label_ids, label_names, backgroud_class
        )

    def compute_score(self, pred_events_video, target_events_video, threshold):
        label_id_to_name = {
            self.label_ids[i]: self.label_names[i] for i in range(len(self.label_names))
        }
        results = {
            l: {
                "r_event": 0,
                "p_event": 0,
                "nb_true": 0,
                "nb_pred": 0,
                "precision": None,
                "recall": None,
                "f1-score": None,
                "support": None,
            }
            for l in self.label_names
        }

        for video_id in range(len(pred_events_video)):
            pred_events = pred_events_video[video_id]
            target_events = target_events_video[video_id]

            for pred_event in pred_events:
                results[label_id_to_name[pred_event.label]]["nb_pred"] += 1
                # Check if the event is in the true event
                for true_event in target_events:
                    if pred_event.label == true_event.label:
                        overlap = compute_event_overlap(pred_event, true_event)
                        if overlap >= threshold:
                            results[label_id_to_name[pred_event.label]]["p_event"] += 1
                            break

            for true_event in target_events:
                # Check if the event is in the true event
                results[label_id_to_name[true_event.label]]["nb_true"] += 1
                for pred_event in pred_events:
                    if pred_event.label == true_event.label:
                        overlap = compute_event_overlap(pred_event, true_event)
                        if overlap >= threshold:
                            results[label_id_to_name[pred_event.label]]["r_event"] += 1
                            break

        scores = self.get_score_event_matching_association(results)

        return scores

    def get_score_event_matching_association(self, results):
        for l in self.label_names:
            results[l]["precision"] = self.pdv(results[l]["p_event"], results[l]["nb_pred"])
            results[l]["recall"] = self.pdv(results[l]["r_event"], results[l]["nb_true"])
            p = results[l]["precision"]
            r = results[l]["recall"]
            results[l]["f1-score"] = self.pdv(2 * p * r, p + r)
            results[l]["support"] = results[l]["nb_true"]

        # micro
        results["micro avg"] = {}
        results["micro avg"]["recall"] = self.pdv(
            sum([results[l]["r_event"] for l in self.label_names]),
            sum([results[l]["nb_true"] for l in self.label_names]),
        )
        results["micro avg"]["precision"] = self.pdv(
            sum([results[l]["p_event"] for l in self.label_names]),
            sum([results[l]["nb_pred"] for l in self.label_names]),
        )
        p = results["micro avg"]["precision"]
        r = results["micro avg"]["recall"]
        results["micro avg"]["f1-score"] = self.pdv(2 * p * r, p + r)
        results["micro avg"]["support"] = sum([results[l]["nb_true"] for l in self.label_names])

        # macro
        results["macro avg"] = {}
        results["macro avg"]["recall"] = self.pdv(
            sum([results[l]["recall"] for l in self.label_names]), len(self.label_names)
        )
        results["macro avg"]["precision"] = self.pdv(
            sum([results[l]["precision"] for l in self.label_names]), len(self.label_names)
        )
        results["macro avg"]["f1-score"] = self.pdv(
            sum([results[l]["f1-score"] for l in self.label_names]), len(self.label_names)
        )
        results["macro avg"]["support"] = sum([results[l]["nb_true"] for l in self.label_names])

        # weighted
        results["weighted avg"] = {}
        results["weighted avg"]["recall"] = self.pdv(
            sum([results[l]["recall"] * results[l]["nb_true"] for l in self.label_names]),
            sum([results[l]["nb_true"] for l in self.label_names]),
        )
        results["weighted avg"]["precision"] = self.pdv(
            sum([results[l]["precision"] * results[l]["nb_pred"] for l in self.label_names]),
            sum([results[l]["nb_pred"] for l in self.label_names]),
        )
        p = results["weighted avg"]["precision"]
        r = results["weighted avg"]["recall"]
        results["weighted avg"]["f1-score"] = self.pdv(2 * p * r, p + r)
        results["weighted avg"]["support"] = sum([results[l]["nb_true"] for l in self.label_names])

        return results


class EventDetectionMatching(EventDetectionBase):
    """Metric to evaluate the event detection using a matching method.
    First, the predicted events are matched with the ground truth events based on IOU and classes.
    Then, TP, FP, FN are gathered for each class by looking at each matched event and unmatched events based on threshold.

    Note: the result is in the same format as the classification report dict from sklearn
    """

    def __init__(
        self,
        num_classes=6,
        smooth_marge=5,
        threshold=[0.1, 0.25, 0.5],
        label_ids=[1, 2, 3, 4, 5],
        label_names=["Nod", "Shake", "Tilt", "Turn", "Up_down"],
        backgroud_class=0,
    ) -> None:
        super().__init__(
            num_classes, smooth_marge, threshold, label_ids, label_names, backgroud_class
        )

    def compute_score(self, pred_all_video, target_all_video, threshold):
        label_id_to_name = {
            self.label_ids[i]: self.label_names[i] for i in range(len(self.label_names))
        }
        results = {
            l: {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "nb_true": 0,
                "nb_pred": 0,
                "precision": None,
                "recall": None,
                "f1-score": None,
                "support": None,
            }
            for l in self.label_names
        }

        for video in range(len(pred_all_video)):
            pred_events = pred_all_video[video]
            target_events = target_all_video[video]

            # Matched the events
            row_ind, col_ind, cost_matrix = get_event_matching(pred_events, target_events)

            for i_pred, pred_event in enumerate(pred_events):
                results[label_id_to_name[pred_event.label]]["nb_pred"] += 1

                if i_pred in row_ind:  # Check if the predicted event is matched
                    j_gt = col_ind[np.where(row_ind == i_pred)[0][0]]
                    gt_event = target_events[j_gt]
                    score = cost_matrix[i_pred, j_gt]
                    if score < (
                        1 - threshold
                    ):  # if the score is less than 1, the IOU is greater than 0
                        # TP
                        results[label_id_to_name[pred_event.label]]["tp"] += 1
                    else:
                        # FN
                        results[label_id_to_name[gt_event.label]]["fn"] += 1
                        # FP
                        results[label_id_to_name[pred_event.label]]["fp"] += 1
                else:  # if the predicted event is not matched
                    # FP
                    results[label_id_to_name[pred_event.label]]["fp"] += 1

            # Then, add the unmatched target events
            for j_gt, gt_event in enumerate(target_events):
                results[label_id_to_name[gt_event.label]]["nb_true"] += 1
                if j_gt not in col_ind:
                    # FN
                    results[label_id_to_name[gt_event.label]]["fn"] += 1

        scores = self.get_score_event_matching(results, self.label_names)

        return scores

    def get_score_event_matching(self, results, label_names):
        for l in label_names:
            results[l]["precision"] = self.pdv(
                results[l]["tp"], results[l]["tp"] + results[l]["fp"]
            )
            results[l]["recall"] = self.pdv(results[l]["tp"], results[l]["tp"] + results[l]["fn"])
            p = results[l]["precision"]
            r = results[l]["recall"]
            results[l]["f1-score"] = self.pdv(2 * p * r, p + r)
            results[l]["support"] = results[l]["nb_true"]

        # micro
        results["micro avg"] = {}
        results["micro avg"]["recall"] = self.pdv(
            sum([results[l]["tp"] for l in label_names]),
            sum([results[l]["tp"] + results[l]["fn"] for l in label_names]),
        )
        results["micro avg"]["precision"] = self.pdv(
            sum([results[l]["tp"] for l in label_names]),
            sum([results[l]["tp"] + results[l]["fp"] for l in label_names]),
        )
        p = results["micro avg"]["precision"]
        r = results["micro avg"]["recall"]
        results["micro avg"]["f1-score"] = self.pdv(2 * p * r, p + r)
        results["micro avg"]["support"] = sum([results[l]["nb_true"] for l in label_names])

        # macro
        results["macro avg"] = {}
        results["macro avg"]["recall"] = self.pdv(
            sum([results[l]["recall"] for l in label_names]), len(label_names)
        )
        results["macro avg"]["precision"] = self.pdv(
            sum([results[l]["precision"] for l in label_names]), len(label_names)
        )
        results["macro avg"]["f1-score"] = self.pdv(
            sum([results[l]["f1-score"] for l in label_names]), len(label_names)
        )
        results["macro avg"]["support"] = sum([results[l]["nb_true"] for l in label_names])

        # weighted
        results["weighted avg"] = {}
        results["weighted avg"]["recall"] = self.pdv(
            sum([results[l]["recall"] * results[l]["nb_true"] for l in label_names]),
            sum([results[l]["nb_true"] for l in label_names]),
        )
        results["weighted avg"]["precision"] = self.pdv(
            sum([results[l]["precision"] * results[l]["nb_pred"] for l in label_names]),
            sum([results[l]["nb_pred"] for l in label_names]),
        )
        p = results["weighted avg"]["precision"]
        r = results["weighted avg"]["recall"]
        results["weighted avg"]["f1-score"] = self.pdv(2 * p * r, p + r)
        results["weighted avg"]["support"] = sum([results[l]["nb_true"] for l in label_names])

        return results


class EventDetectionOverlap(EventDetectionBase):
    """Metric to evaluate the event detection using overlap method.
    # "Temporal Convolutional Networks for Action Segmentation and Detection" https://arxiv.org/pdf/1611.05267
    # "MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation" https://arxiv.org/pdf/1903.01945

    Note: the result is in the same format as the classification report dict from sklearn
    """

    def __init__(
        self,
        num_classes=6,
        smooth_marge=5,
        threshold=[0.1, 0.25, 0.5],
        label_ids=[1, 2, 3, 4, 5],
        label_names=["Nod", "Shake", "Tilt", "Turn", "Up_down"],
        backgroud_class=0,
    ) -> None:
        super().__init__(
            num_classes, smooth_marge, threshold, label_ids, label_names, backgroud_class
        )

    def compute_score(self, pred_events_video, target_events_video, threshold):
        label_id_to_name = {
            self.label_ids[i]: self.label_names[i] for i in range(len(self.label_names))
        }
        results = {
            l: {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "nb_true": 0,
                "nb_pred": 0,
                "precision": None,
                "recall": None,
                "f1-score": None,
                "support": None,
            }
            for l in self.label_names
        }

        for video_id in range(len(pred_events_video)):
            pred_events = pred_events_video[video_id]
            target_events = target_events_video[video_id]

            hits = np.zeros(len(target_events))
            for j in range(len(pred_events)):
                pred_event = pred_events[j]
                results[label_id_to_name[pred_event.label]]["nb_pred"] += 1
                # Compute IoU against all others target events
                ious = [
                    compute_event_overlap_IOU(pred_event, target_event)
                    for target_event in target_events
                ]
                ious = np.array(ious) * np.array(
                    [target_event.label == pred_event.label for target_event in target_events]
                )
                idx = np.argmax(ious)

                # If the IoU is high enough and the true segment isn't already used
                # Then it is a true positive. Otherwise is it a false positive.
                if ious[idx] > threshold and hits[idx] == 0:
                    results[label_id_to_name[pred_event.label]]["tp"] += 1
                    hits[idx] = 1
                else:
                    results[label_id_to_name[pred_event.label]]["fp"] += 1

            for i in range(len(target_events)):
                results[label_id_to_name[target_events[i].label]]["nb_true"] += 1
                # If the true segment isn't already used, then it is a false negative
                if hits[i] == 0:
                    results[label_id_to_name[target_events[i].label]]["fn"] += 1

        scores = self.get_scores_overlap(results, self.label_names)
        return scores

    def get_scores_overlap(self, results, label_names):
        for l in label_names:
            results[l]["precision"] = self.pdv(
                results[l]["tp"], results[l]["tp"] + results[l]["fp"]
            )
            results[l]["recall"] = self.pdv(results[l]["tp"], results[l]["tp"] + results[l]["fn"])
            p = results[l]["precision"]
            r = results[l]["recall"]
            results[l]["f1-score"] = self.pdv(2 * p * r, p + r)
            results[l]["support"] = results[l]["nb_true"]

        # micro
        results["micro avg"] = {}
        results["micro avg"]["recall"] = self.pdv(
            sum([results[l]["tp"] for l in label_names]),
            sum([results[l]["tp"] + results[l]["fn"] for l in label_names]),
        )
        results["micro avg"]["precision"] = self.pdv(
            sum([results[l]["tp"] for l in label_names]),
            sum([results[l]["tp"] + results[l]["fp"] for l in label_names]),
        )
        p = results["micro avg"]["precision"]
        r = results["micro avg"]["recall"]
        results["micro avg"]["f1-score"] = self.pdv(2 * p * r, p + r)
        results["micro avg"]["support"] = sum([results[l]["nb_true"] for l in label_names])

        # macro
        results["macro avg"] = {}
        results["macro avg"]["recall"] = self.pdv(
            sum([results[l]["recall"] for l in label_names]), len(label_names)
        )
        results["macro avg"]["precision"] = self.pdv(
            sum([results[l]["precision"] for l in label_names]), len(label_names)
        )
        results["macro avg"]["f1-score"] = self.pdv(
            sum([results[l]["f1-score"] for l in label_names]), len(label_names)
        )
        results["macro avg"]["support"] = sum([results[l]["nb_true"] for l in label_names])

        # weighted
        results["weighted avg"] = {}
        results["weighted avg"]["recall"] = self.pdv(
            sum([results[l]["recall"] * results[l]["nb_true"] for l in label_names]),
            sum([results[l]["nb_true"] for l in label_names]),
        )
        results["weighted avg"]["precision"] = self.pdv(
            sum([results[l]["precision"] * results[l]["nb_pred"] for l in label_names]),
            sum([results[l]["nb_pred"] for l in label_names]),
        )
        p = results["weighted avg"]["precision"]
        r = results["weighted avg"]["recall"]
        results["weighted avg"]["f1-score"] = self.pdv(2 * p * r, p + r)
        results["weighted avg"]["support"] = sum([results[l]["nb_true"] for l in label_names])

        return results
