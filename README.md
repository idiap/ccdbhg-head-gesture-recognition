# CCDbHG Head Gesture Dataset

Real-time demo and evaluation code of our Face and Gesture 2024 paper: 
"CCDb-HG: Novel Annotations and Gaze-Aware Representations for
Head Gesture Recognition" [Link](https://publications.idiap.ch/attachments/papers/2024/Vuillecard_FG_2024.pdf)

You can download the CCDb dataset from [here](https://ccdb.cs.cf.ac.uk/) and our annotation from [here](https://zenodo.org/records/13927536).

## CCDbHG Data Annotation:

<img src="https://github.com/idiap/ccdbhg-head-gesture-recognition/blob/main/images/all_gestures.png" alt="Head Gestures" width="650"/>

### Head Gestures Definition
- **Nod** is an up-down rotation along the pitch axis. It involves a slight, quick, or repetitive lowering and raising of the head. 
- **Shake** is a left-right horizontal rotation along the yaw axis. It involves a rapid and potentially repeated side-to-side motion, typically with small or moderate amplitude.
- **Tilt** is a sideways rotation along the roll axis involving
a shift of the head in which one ear moves closer to the
shoulder while the other ear moves away.
- **Turn** corresponds to a left or right rotation involving the shifting of the head from its original position to another one facing a different direction. Head turns can vary in amplitude, ranging from a slight turn to a complete reorientation of the head. It differentiates from a shake by being a nonrepetitive movement and often initiated by a gaze shift.
- **Up/Down** is similar to a turn but along the pitch direction and usually involves a gaze shift in the same direction as the head.
- **Waggle** usually happens when speaking and
involves a rhythmic swaying motion typically performed in a
repeated manner. Unlike nod, shake, and tilt, waggle involves
several head axis at the same time.

### CCDbHG Gestures Examples
| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|**Nod** <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/idiap/ccdbhg-head-gesture-recognition/blob/main/images/CCDb_nods.gif">| **Shake** <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/idiap/ccdbhg-head-gesture-recognition/blob/main/images/CCDb_shakes.gif">| **Tilt** <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/idiap/ccdbhg-head-gesture-recognition/blob/main/images/CCDb_tilt.gif">| 
| **Turn** <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/idiap/ccdbhg-head-gesture-recognition/blob/main/images/CCDb_turn.gif"> | **Up Down** <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/idiap/ccdbhg-head-gesture-recognition/blob/main/images/CCDb_up_down.gif">| |


## Project Structure

```
├── data                        <- CCDbHG folder can download from [here](http://example.com)
│   ├── extraction              <- Extracted cues from CCDb dataset
│
├── images                      <- Visualizations of CCDb dataset and annotation
│
├── src                         <- Source code
│   ├── classifier/             <- Model architectures
│   ├── model_checkpoints/      <- Model checkpoints
│   │
│   ├── dataset.py              <- Dataset class
│   ├── inference.py            <- Model inference
│   ├── metrics.py              <- Torchmetrics methods
│   └── utils_*.py              <- Utility scripts
│
├── .gitignore                  <- List of files ignored by git
├── .project-root               <- File to identify the root of the project
├── .pre-commit-config.yaml     <- Configuration of pre-commit hooks for code formatting
├── requirements.txt            <- Python dependencies for the project
├── THIRDPARTY                  <- License file
├── Dockerfile                  <- Dockerfile for the project
├── demo.py                     <- Demo script for real-time head gesture recognition
├── app.py                      <- WebApp script for real-time head gesture gif generation
├── run_evaluation.py           <- Run our evaluation script from the paper on CCDbHG dataset 
└── README.md
```

## Setup & Installation

First, clone this repository:

```bash
git clone https://github.com/idiap/ccdbhg-head-gesture-recognition.git
cd ccdbhg-head-gesture-recognition
```

Then, install the requirements using conda:

```bash
conda create --name head_gesture python=3.11
conda activate head_gesture
conda install pip 
pip install -r requirements.txt
```

## Run the evaluation

Run the evaluation script to evaluate the best model on the CCDbHG dataset:

CNN model with landmarks and head pose features:
```bash
python run_evaluation.py --model cnn_lmk_hp --batch_size 128 
```

CNN model with landmarks, head pose and gaze features:
```bash
python run_evaluation.py --model cnn_lmk_hp_gaze --batch_size 128
```

If GPU is available, you can specify the device:
```bash
python run_evaluation.py --model cnn_lmk_hp_gaze --device cuda --batch_size 128
```

Note: you can increase the batch size for faster evaluation.

## Run the demos

Real-time head gesture recognition for up to 4 people at the same time in the frame.
In this repo, two demo scripts are provided, but you need a webcam to run them.

### Real-time head gesture recognition 
<img src="https://github.com/idiap/ccdbhg-head-gesture-recognition/blob/main/images/demo_python.png" alt="Demo" width="400"/>

In the first demo, a window will open showing the webcam feed with the head gesture recognition, bounding boxes, and landmarks.

```bash
# make sure to have a webcam connected and conda environment activated
python demo.py --face_detector CV2 
```
There are two face detectors available: `CV2` and `YUNET`. The `YUNET` detector is more accurate (especially if far from the camera) but slower compared to CV2.

Note: Please wait 3 seconds for the track and detect face to be set up.

### WebApp head gesture gif generation 

<img src="https://github.com/idiap/ccdbhg-head-gesture-recognition/blob/main/images/webapp_screen.png" alt="WebApp" width="400"/>

In this demo, you will run the app localy, you have to open a browser to see the webcam feed with the captured head gesture gif. 
```bash
# make sure to have a webcam connected and conda environment activated
python app.py
```
Then, open a browser and go to `http://localhost:5000/`

⚠️ The app is saving the head gesture gif in '/static/gifs/' folder. Make sure to delete the gifs folder if you run out of space.

Alternatively, you can run it with docker, but it doesn't work (it's too slow ). Otherwise, the Dockerfile can be used as a template to run the evaluation code. First, build the docker image, if not already done:
```bash
docker build -t head_gesture .
```
Then, run the docker container:
```bash
docker run -p 5000:5000 head_gesture
```
Then, open a browser and go to `http://localhost:5000/`

Note: Please wait  3 seconds for the track and detect face to be set up. The app is not error-proof; if an error occurs, please restart the app.



## Authors and acknowledgment
The work was co-financed by Innosuisse, the Swiss innovation agency,
through the NL-CH Eureka Innovation project ePartner4ALL (a personalized
and blended care solution with a virtual buddy for child health, number
57272.1 IP-ICT).

## License & Third-party resources

Warning: The code is under the license of GPL-3.0-only license. For the model chekpoints, the model cnn_lmk_hp/ is under the license of GPL-3.0-only license but the model cnn_lmk_hp_gaze/ is under the license of CC BY-NC-SA 4.0 license which is non-commercial use. This is because the gaze used and extracted from the ETH-XGaze dataset is under the license of CC BY-NC-SA 4.0 license. 
However, the demo code is using the cnn_lmk_hp/ thus it is under the license of GPL-3.0-only license.

Dataset:
 - [CCDb dataset](https://ccdb.cs.cf.ac.uk/)

Extracted features from the CCDb dataset:
 - Landmarks and head pose [Mediapipe](https://mediapipe-studio.webapps.google.com/demo/face_landmarker)
 - Gaze [Xgaze](https://github.com/xucong-zhang/ETH-XGaze#:~:text=The%20code%20is%20under%20the%20license%20of%20CC%20BY%2DNC%2DSA%204.0%20license) trained on [ETH-XGaze dataset](https://ait.ethz.ch/xgaze#:~:text=LICENSE,license%20file)

Demo: 
 - Tracking of face bounding box MotPy [code](https://github.com/wmuron/motpy)
 - Face detection [YUNET](https://github.com/geaxgx/depthai_yunet) and [CV2](https://opencv.org/)
 - Landmarks and head pose [Mediapipe](https://mediapipe-studio.webapps.google.com/demo/face_landmarker)
## Citation

If you use this dataset, please cite the following paper:

```
@INPROCEEDINGS{Vuillecard_FG_2024,
         author = {Vuillecard, Pierre and Farkhondeh, Arya and Villamizar, Michael and Odobez, Jean-Marc},
          title = {CCDb-HG: Novel Annotations and Gaze-Aware Representations for Head Gesture Recognition},
      booktitle = {18th IEEE Int. Conference on Automatic Face and Gesture Recognition (FG), Istanbul,},
           year = {2024},
            pdf = {https://publications.idiap.ch/attachments/papers/2024/Vuillecard_FG_2024.pdf}
}
```