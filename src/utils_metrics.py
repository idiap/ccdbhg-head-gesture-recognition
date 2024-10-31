# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass
class Event:
    start: int
    end: int
    label: int

    def __repr__(self):
        return f"Event(start={self.start}, end={self.end}, label={self.label})"

    def get_duration(self):
        return self.end - self.start + 1


def get_events(y, marge=7, background_label=0) -> List[Event]:
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
            events.append(Event(**current_event))
            if y[i] != background_label:
                current_event["start"] = i
                current_event["label"] = y[i]
            else:
                current_event = {"start": None, "end": None, "label": None}

    if current_event["start"] is not None:
        current_event["end"] = len(y) - 1
        events.append(Event(**current_event))

    # remove small events
    if (marge is not None) and (marge > 0):
        for e in events:
            if e.get_duration() <= marge:
                events.remove(e)

    return events


def smooth_pred(pred, marge=5, background_label=0):
    """

    Smooth the segmentation prediction using majority voting over a window size (2*marge+1).
    - It removes short events len(event) < marge.
    - It merges events with the same label if they are close len(gap) <= marge.
    - Pads the sequence with zeros to handle boundary conditions.

    """

    if marge == 0 or len(pred) <= marge * 2:  # No smoothing if the sequence is too short
        return pred

    # Pad the prediction with `marge` zeros at the beginning and the end
    padded_pred = [background_label] * marge + pred + [background_label] * marge

    smooth_pred = []

    # Perform majority voting over the window
    for i in range(marge, len(padded_pred) - marge):
        segment = padded_pred[i - marge : i + marge + 1]
        smooth_pred.append(max(set(segment), key=segment.count))

    # The length of `smooth_pred` should match the length of the original `pred`
    assert len(smooth_pred) == len(pred), f"smooth_pred {len(smooth_pred)} != pred {len(pred)}"

    return smooth_pred


def get_transition_frames(y, y_events: List[Event], transition_frame_marge=2):
    """
    Identify the transition frames at the start and end of the groud truth events
    We can exlude them from the frame metrics
    """
    transition = np.zeros(len(y))

    if transition_frame_marge == 0:
        return transition

    for event in y_events:
        transition[
            max(event.start - transition_frame_marge, 0) : min(
                event.start + transition_frame_marge, len(y)
            )
        ] = 1
        transition[
            max(event.end - transition_frame_marge, 0) : min(
                event.end + transition_frame_marge, len(y)
            )
        ] = 1
    return transition


def compute_event_overlap(pred_event: Event, gt_event: Event):
    # compute the overlap between two events with F1 score [0,1]
    set_pred = set(range(pred_event.start, pred_event.end + 1))
    set_true = set(range(gt_event.start, gt_event.end + 1))
    intersection = set.intersection(set_pred, set_true)
    if len(intersection) > 0:
        p = len(intersection) / len(set_true)
        r = len(intersection) / len(set_pred)
        return 2 * p * r / (p + r)
    else:
        return 0


def compute_event_overlap_IOU(pred_event: Event, gt_event: Event):
    # compute the overlap between two events with IOU [0,1]
    set_pred = set(range(pred_event.start, pred_event.end + 1))
    set_true = set(range(gt_event.start, gt_event.end + 1))
    intersection = set.intersection(set_pred, set_true)
    union = set.union(set_pred, set_true)
    return len(intersection) / len(union)


####
# FUNCTION for Event Detection Matching
####


# implementation of the hungarian algorithm to find the best match
def compute_cost_matrix_events(pred_events: List[Event], gt_events: List[Event]):
    # compute the cost matrix
    cost_matrix = np.zeros((len(pred_events), len(gt_events)))
    for i_pred, pred_event in enumerate(pred_events):
        for j_gt, gt_event in enumerate(gt_events):
            if pred_event.label == gt_event.label:
                cost_matrix[i_pred, j_gt] = 1 - compute_event_overlap_IOU(pred_event, gt_event)
            else:
                cost_matrix[i_pred, j_gt] = 2
    return cost_matrix


def get_event_matching(pred_events: List[Event], gt_events: List[Event]):
    # compute the cost matrix
    cost_matrix = compute_cost_matrix_events(pred_events, gt_events)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return row_ind, col_ind, cost_matrix


# Visualize scores


def classification_report(dict_report):
    digits = 2
    L = []
    for k, v in dict_report.items():
        l = []
        metrics = list(v.keys())

        if "precision" in metrics:
            l.append(round(v["precision"], digits))
        else:
            l.append(None)
        if "recall" in metrics:
            l.append(round(v["recall"], digits))
        else:
            l.append(None)
        if "f1-score" in metrics:
            l.append(round(v["f1-score"], digits))
        else:
            l.append(None)
        if "support" in metrics:
            l.append(v["support"])
        else:
            l.append(None)
        L.append(l)
    df = pd.DataFrame(
        L, columns=["precision", "recall", "f1-score", "support"], index=list(dict_report.keys())
    )
    print(df)
