# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from torch.utils.data import Dataset

from src.inference import HeadGestureInference


class CcdbhgTestDataset(Dataset):
    def __init__(self, data, predictor: HeadGestureInference):
        self.video_test = [
            data[i]
            for i in range(len(data))
            if data[i]["info"]["subject"] in ["P2", "P5", "P20", "P10"]
        ]
        self.sample = [
            (frame_id, video_id)
            for video_id, video in enumerate(self.video_test)
            for frame_id in range(15, len(video["landmarks"]) - 15)
        ]
        # remove waggle from the ground truth
        self.transform_y = lambda y: y if y <= 5 else 0
        self.predictor = predictor

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, index):
        frame_id, video_id = self.sample[index]
        video = self.video_test[video_id]
        sample_data = {
            "landmarks_detector": video["landmarks"][frame_id - 15 : frame_id + 16],
            "head_pose_detector": video["head_pose"][frame_id - 15 : frame_id + 16],
            "gaze_detector": video["gaze"][frame_id - 15 : frame_id + 16],
        }

        label = video["label"][frame_id][0]
        label = self.transform_y(label)
        inputs = self.predictor.preprocess(None, None, sample_data).squeeze(0)
        return inputs, label, frame_id, video_id
