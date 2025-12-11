# Self-Driving Car Project

## Overview
This project implements a behavioral cloning model to control a virtual car in the Udacity self-driving simulator. The goal was to train a convolutional neural network (based on the NVIDIA architecture) to predict steering angles from front-facing camera images.

The pipeline includes data collection, preprocessing, augmentation, model training, and real-time control through a Flask + Socket.IO server.

## Approach
1. Collected driving data by manually recording several laps in the simulator.
2. Cleaned the dataset by removing most of the straight-driving samples, since they dominate and cause the model to predict 0.0 too often.
3. Preprocessed images by cropping the sky/hood, resizing to 200×66, and normalizing pixel values.
4. Implemented on-the-fly augmentation such as brightness changes, translations, and flips to make the model more robust.
5. Trained an NVIDIA-style CNN to learn steering behavior.
6. Connected the trained model to the simulator using a Socket.IO server to output steering and throttle commands in real time.

## Challenges and Solutions

### 1. Strong Left/Right Bias in Predictions
Many early models drifted heavily due to unbalanced data (too many straight frames or poor recovery examples).

**Solution:** Downsampled straight angles, added recovery driving data, and applied augmentation.

### 2. Simulator Not Responding to Model Output
Different Python and dependency versions caused Socket.IO or TensorFlow to break.

**Solution:** Set up a clean environment using Python 3.10 and installed compatible versions of Flask, eventlet, socketio, numpy, and TensorFlow.

### 3. Unstable Steering During Testing
Small prediction errors quickly accumulated and pushed the car off the road.

**Solution:** Adjusted preprocessing, refined augmentation, and added minor steering bias correction.

## How to Run This Project

### 1. Environment Setup
Create a Python 3.10 virtual environment:

```bash
python -m venv self_driving_env
source self_driving_env/bin/activate       # Linux/Mac
self_driving_env\Scripts\activate          # Windows
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

(If no file is provided, install manually: TensorFlow, numpy, opencv-python, flask, flask-socketio, eventlet, pillow.)

### 2. Training the Model
Place all your simulator data in the `data/` directory.

Run training:

```bash
python -m scripts.train_model
```

The trained model will be saved in `models/`.

### 3. Testing in the Simulator
Switch to the simulation environment (if separate):

```bash
activate sim_env                    # Windows
source sim_env/bin/activate         # Linux/Mac
```

Run the server that communicates with the simulator:

```bash
python -m scripts.test_simulation --model models/nvidia_model.h5
```

Then open the Udacity simulator, select Autonomous Mode, and the car will begin driving once connected.

## Project Structure (Simplified)

```
project/
│
├── data/                 # Raw simulator data
├── models/               # Saved trained models
├── scripts/
│   ├── train_model.py
│   ├── test_simulation.py
│   ├── data_utils.py
│   └── ...
└── README.md
```
