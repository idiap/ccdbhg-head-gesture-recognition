# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

# app.py
import os
from io import BytesIO

import cv2
import imageio
import numpy as np
import rootutils
from flask import Flask, jsonify, render_template, request

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils_demo import (
    FaceDetectorCV2,
    FaceTracker,
    HGPredictor,
    MediapipePredictor,
    TrackHandler,
    Visualizer,
)

app = Flask(__name__)

os.makedirs("static/gifs", exist_ok=True)
# Global variable to store frames for creating a GIF
frame_buffer = {}
timestamps = []
event_already_draw = {}

# Instantiate
# face_detector = FaceDetectorYUNET()
face_detector = FaceDetectorCV2()
face_tracker = FaceTracker(
    dt=1 / 15,
    model_spec={
        "order_pos": 1,
        "dim_pos": 2,
        "order_size": 0,
        "dim_size": 2,
        "q_var_pos": 1000.0,
        "r_var_pos": 0.1,
    },
)
face_predictor = MediapipePredictor()
hg_predictor = HGPredictor("cpu")
track_handler = TrackHandler(face_tracker)
visualizer = Visualizer(
    draw_bbox=True,
    draw_landmarks=False,
    draw_head_gesture=True,
    fontsize=1,
    space_legend=20,
)


def clean_frame_buffer():
    if len(timestamps) >= 300:
        for i in range(len(timestamps) - 300):
            time_stamp = timestamps[0]
            frame_buffer.pop(time_stamp)
            timestamps.pop(0)


def create_gif(event):
    track_id = event.track_id
    start_time = event.start
    end_time = event.end
    # get the idx of the start and end time
    start_idx = timestamps.index(start_time)
    end_idx = timestamps.index(end_time)
    # get the frames
    frames = [frame_buffer[timestamps[i]] for i in range(start_idx, end_idx)]
    # get the bbox
    track_timestamp = [face.timestamp for face in face_tracker.tracks_store[track_id]]
    start_idx = track_timestamp.index(start_time)
    end_idx = track_timestamp.index(end_time)
    middle_idx = (start_idx + end_idx) // 2
    bbox_loc = face_tracker.tracks_store[track_id][middle_idx].loc
    # crop the frames making sure the bbox is within the frame
    frames = [frame[bbox_loc.y1 : bbox_loc.y2, bbox_loc.x1 : bbox_loc.x2] for frame in frames]
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    # create the gif
    gif_buffer = BytesIO()
    imageio.mimsave(gif_buffer, frames, format="GIF", duration=1 / 10, loop=0)  # 15 FPS
    gif_buffer.seek(0)

    return gif_buffer


def predict_head_gesture(frame, timestamp):
    # Detect faces
    detection = face_detector.process_image(frame)
    # Track the faces
    face_tracker.update(detection, timestamp)
    # Get the current track id
    track_id = face_tracker.get_tracks()
    # Detect the face landmarks in those faces
    for track in track_id:
        face_prediction = face_predictor.process_face(frame, face_tracker.tracks_store[track][-1])
        if track in face_tracker.tracks_store:
            face_tracker.tracks_store[track][-1].add_prediction(face_prediction)

    output_track = hg_predictor.process(face_tracker, track_id)
    track_handler.add_track_prediction(output_track)
    last_event = track_handler.get_last_event()
    # frame = visualizer.process(frame, face_tracker, output_track)

    return frame, output_track, last_event


def get_gif_clip(track_id, event):
    if track_id in event_already_draw:
        if event.idx == event_already_draw[track_id].idx:
            # same event then we skip
            return None
        else:
            # we draw the event
            gif_buffer = create_gif(event)
            event_already_draw[track_id] = event
            return gif_buffer
    else:
        # we draw the event
        gif_buffer = create_gif(event)
        event_already_draw[track_id] = event
        return gif_buffer


# Route for serving the index.html file
@app.route("/")
def index():
    return render_template(
        "index.html"
    )  # This looks for 'index.html' inside the 'templates/' directory


# Endpoint to receive a frame and timestamp
@app.route("/receive_frame", methods=["POST"])
def receive_frame():
    global current_gif_url
    frame = request.files.get("frame")
    timestamp = int(request.form.get("timestamp"))

    if not frame or not timestamp:
        return jsonify({"error": "Missing frame or timestamp"}), 400

    # Convert frame to numpy array
    frame_bytes = frame.read()
    np_frame = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

    # if img is black the we skip
    if np.all(img == 0):
        # Send the modified image back to the client
        return jsonify({"gif_url": None, "gesture_name": None})

    # Add frame and timestamp to the buffer
    frame_buffer[timestamp] = img.copy()
    timestamps.append(timestamp)

    clean_frame_buffer()

    # Predict head gesture based on the frame and timestamp
    _, _, last_event = predict_head_gesture(img, timestamp)

    gif_filename = None
    gesture_name = None
    # If an event is detected, create a GIF and return it
    for track_id, event in last_event.items():
        gif_to_draw = get_gif_clip(track_id, event)

        if gif_to_draw:
            gif_filename = f"static/gifs/gesture_gif_{event.end}.gif"
            gesture_name = event.label.capitalize()
            # check if the gif already exist
            if not os.path.exists(gif_filename):
                with open(gif_filename, "wb") as f:
                    f.write(gif_to_draw.getvalue())

    file = "/" + gif_filename if gif_filename else None
    return jsonify({"gif_url": file, "gesture_name": gesture_name})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
