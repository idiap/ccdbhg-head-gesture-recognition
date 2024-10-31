# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import argparse
import os
import pickle

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CcdbhgTestDataset
from src.inference import HeadGestureInference
from src.metrics import (
    EventDetectionAssociation,
    EventDetectionMatching,
    EventDetectionOverlap,
    FrameClassification,
)
from src.utils_metrics import classification_report

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "data")
DATA_NAME = (
    "samples_video_hp_mediapipe_v1_g_xgaze_with_FaceAlignPnPHeadPose_lm_mediapipe_v1_selected.p"
)


def run_evaluation(args):
    # load predictor
    exp_folder = os.path.join(ROOT_PATH, "src", "model_checkpoints", args.model)
    assert os.path.exists(exp_folder), f"Experiment folder {exp_folder} does not exist"
    predictor = HeadGestureInference(exp_folder, args.device)

    data_path = os.path.join(DATA_PATH, "extraction", DATA_NAME)
    assert os.path.exists(data_path), f"Data path {data_path} does not exist"
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # load dataset
    dataset = CcdbhgTestDataset(data=data, predictor=predictor)

    # define dataloader
    dataloader = DataLoader(
        dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False
    )

    # load metric
    metrics = {
        "frame_classification": FrameClassification(),
        "event_detection_association": EventDetectionAssociation(),  # used for event detection in our paper
        "event_detection_overlap": EventDetectionOverlap(),
        "event_detection_matching": EventDetectionMatching(),
    }

    # run evaluation
    for inputs, label, time_video, video_id in tqdm(dataloader):
        output = predictor.inference(inputs)
        for metric in metrics.values():
            metric.update(output["prob"], label, time_video, video_id)

    # compute results
    for k in metrics.keys():
        metrics[k] = metrics[k].compute()

    # display results
    if args.verbose:
        print("Frame Classification")
        print(
            classification_report(metrics["frame_classification"]["frame_classification_report"])
        )
        print("Event Detection")
        print(classification_report(metrics["event_detection_association"]["threshold_0.1"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn_lmk_hp_gaze",
        help="Model name",
        choices=["cnn_lmk_hp_gaze", "cnn_lmk_hp"],
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device", choices=["cuda", "cpu"]
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose")
    args = parser.parse_args()

    run_evaluation(args)
