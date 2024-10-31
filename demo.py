# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import argparse
import time

import cv2

from src.utils_demo import (
    FaceDetectorCV2,
    FaceDetectorYUNET,
    FaceTracker,
    HGPredictor,
    MediapipePredictor,
    TrackHandler,
    Visualizer,
)


def main(args):
    # Instantiate
    if args.face_detector == "YUNET":
        face_detector = FaceDetectorYUNET()
    elif args.face_detector == "CV2":
        face_detector = FaceDetectorCV2()
    else:
        raise ValueError("Invalid face detector")
    face_tracker = FaceTracker()
    face_predictor = MediapipePredictor()
    hg_predictor = HGPredictor("cpu")
    track_handler = TrackHandler(face_tracker)
    visualizer = Visualizer(
        draw_bbox=args.draw_bbox,
        draw_landmarks=args.draw_landmarks,
        draw_head_gesture=args.draw_head_gesture,
    )

    vid = cv2.VideoCapture(0)
    while True:
        # Capture the video frame
        # by frame
        start = time.time()
        ret, frame = vid.read()

        frame_time = int(round(time.time() * 1000))

        # Detect faces
        detection = face_detector.process_image(frame)

        # Track the faces
        face_tracker.update(detection, frame_time)

        # Get the current track id
        track_id = face_tracker.get_tracks()

        # Detect the face landmarks in those faces
        for track in track_id:
            face_prediction = face_predictor.process_face(
                frame, face_tracker.tracks_store[track][-1]
            )
            face_tracker.tracks_store[track][-1].add_prediction(face_prediction)

        output_track = hg_predictor.process(face_tracker, track_id)
        track_handler.add_track_prediction(output_track)

        # Draw the output
        frame = visualizer.process(frame, face_tracker, output_track)

        # Draw real time fps
        end = time.time()
        fps = 1 / (end - start)
        cv2.putText(
            frame,
            f"FPS {int(fps)} ",
            (1650, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Display the resulting frame
        cv2.imshow("frame", frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run demo")
    parser.add_argument(
        "--face_detector",
        type=str,
        default="CV2",
        help="Face detector CV2 is faster but less accurate if face more than 2m away",
        choices=["YUNET", "CV2"],
    )
    parser.add_argument(
        "--draw_bbox",
        "--db",
        type=bool,
        default=True,
        help="Draw bounding box",
    )
    parser.add_argument(
        "--draw_landmarks",
        "--dl",
        type=bool,
        default=True,
        help="Draw landmarks",
    )
    parser.add_argument(
        "--draw_head_gesture",
        "--dhg",
        type=bool,
        default=True,
        help="Draw head gesture",
    )
    args = parser.parse_args()

    # run demo head gesture
    main(args)
