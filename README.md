# Self-Driving Car Simulation Project Using CNN

## Overview

This project implements a behavioral cloning model to control a virtual car in the Udacity self-driving simulator. The goal was to train a convolutional neural network (based on the NVIDIA architecture) to predict steering angles from front-facing camera images.

The pipeline includes data collection, preprocessing, augmentation, model training, and real-time control through a Flask + Socket.IO server.

**Refer to the PDF for more detailed documentation.**

## Approach

1. **Collect Training Data (Manual Driving)**
2. **Explore & Analyze the Raw Data**
3. **Downsample "Straight Driving" Frames & Cleaning up Raw Data**
4. **Split For Training and Validation**
5. **Image Preprocessing and Data Augmentation**
6. **Creating the Model**
7. **Training the Model**
8. **Build the Real-Time Inference Server**
9. **Room for Improvement**

## Challenges and Solutions

### Data & Training Challenges
- **Imbalanced dataset and steering bias:** Early models drifted due to excessive straight-driving frames. Resolved by downsampling zero-angle samples, adding recovery data, and applying augmentation to create a more balanced distribution.

### Technical & Deployment Challenges
- **Environment and compatibility issues:** Multiple dependency conflicts between training and testing environments caused Socket.IO, TensorFlow, and model loading failures. Resolved by separating environments (training vs. simulation), maintaining consistent library versions, loading models with `compile=False`, and installing compatible dependencies (Python 3.10, specific TensorFlow/Keras/eventlet versions).

### Performance Challenges
- **Unstable steering and poor generalization on sharp curves:** Frame-to-frame prediction noise caused wobbling, and the model struggled with sharp turns due to limited high-angle steering examples. Addressed through preprocessing consistency, steering smoothing, bias correction, and identified need for additional targeted data collection on challenging curves.

## Technologies Used

- **Python 3.10** (simulator environment)
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Image processing
- **Flask + Socket.IO** - Real-time communication with simulator
- **NumPy** - Numerical operations
- **Pandas** - Data handling

## Model Architecture

The model is based on the NVIDIA End-to-End Learning architecture:

- Input: 200×66×3 (YUV color space)
- 5 Convolutional layers with increasing depth (24 → 36 → 48 → 64 → 64)
- Flatten layer
- 4 Fully connected layers (100 → 50 → 10 → 1)
- Output: Single steering angle value

## How to Run This Project

### 1. Environment Setup

Create a Python 3.10 virtual environment:
```bash
python -m venv sim_env
source sim_env/bin/activate       # Linux/Mac
sim_env\Scripts\activate          # Windows
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training the Model

Place all your simulator data in the `data/` directory.

Run training:
```bash
python -m scripts.train_model
```

The trained model will be saved in `models/`.

### 3. Testing in the Simulator

Run the server that communicates with the simulator:
```bash
python scripts/test_simulation.py --model models/nvidia_model.h5 --max-speed 7 --steering-bias 0.0 --smooth 0.3
```

Then open the Udacity simulator, select **Autonomous Mode**, and the car will begin driving once connected.

## Project Structure
```
project/
│
├── data/                           # Raw simulator data
│   ├── driving_log.csv
│   └── IMG/
├── models/                         # Saved trained models
│   ├── nvidia_best.h5
│   └── nvidia_model.h5
├── scripts/
│   ├── train_model.py             # Training script
│   ├── test_simulation.py         # Real-time inference server
│   ├── data_utils.py              # Data preprocessing utilities
│   └── model.py                   # NVIDIA model architecture
├── requirements.txt
└── README.md
```

## Results

- **Training Loss:** 0.2114
- **Validation Loss:** 0.2233
- **Demo Video:** [YouTube Demo](https://youtube.com/shorts/lkaNqqNlJ9s?feature=share)

The model successfully navigates most of the track autonomously, including gentle and moderate curves. Performance could be improved with additional data collection focused on sharp turns and recovery scenarios.

## Future Improvements

- Collect more targeted data for sharp curves
- Further downsample straight-driving frames
- Implement speed control in addition to steering
- Add support for multiple camera angles (left, right, center)
- Experiment with different model architectures
- Add data visualization dashboard

## License

This project is for educational purposes as part of the CVI620 course.

## Acknowledgments

- Udacity for the self-driving car simulator
- NVIDIA for the CNN architecture reference
- Course instructor: Ellie Azizi