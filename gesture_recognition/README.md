# Drone Gesture Classifier

This project is an ESP32-based IMU gesture-recognition system for drone-control commands. It uses an MPU6050 accelerometer/gyroscope to capture short motion windows, trains a lightweight SVM classifier in Python, exports the trained model into C, and runs gesture inference onboard the ESP32.

The system was built as a gesture-control pipeline for a drone interface. The current model recognizes eight command gestures plus a `none` class:

- `up`
- `down`
- `left`
- `right`
- `forward`
- `backward`
- `clockwise`
- `counterclockwise`
- `none`

The `none` class is included so the system can distinguish intentional command gestures from stillness or non-command motion.

## Project Overview

The goal of this project was to build a complete gesture-recognition pipeline that could move from raw IMU data to embedded inference.

The system has four main stages:

1. **IMU capture**  
   The ESP32 reads accelerometer and gyroscope data from an MPU6050 over I²C.

2. **Dataset collection**  
   A Python script triggers the ESP32, captures fixed-length IMU windows over serial, and saves them as labeled CSV-style text files.

3. **Model training**  
   Python extracts motion features from each IMU window and trains a linear SVM classifier.

4. **Embedded deployment**  
   The trained SVM parameters are exported into C and compiled into the ESP32 firmware for onboard inference.

## Hardware

- ESP32 development board
- MPU6050 6-axis IMU
- USB serial connection to PC
- Drone-control software interface for command output

The ESP32 reads the MPU6050 using I²C and samples accelerometer/gyroscope data over a fixed window. The firmware can either dump the captured window as CSV for training data collection or classify the window directly using the exported model.

## Gesture Classes

The model is trained on nine total classes:

| Class | Meaning |
|---|---|
| `up` | upward command |
| `down` | downward command |
| `left` | left command |
| `right` | right command |
| `forward` | forward command |
| `backward` | backward command |
| `clockwise` | clockwise rotation command |
| `counterclockwise` | counterclockwise rotation command |
| `none` | no intentional command gesture |

The eight movement/rotation classes are the actual drone-control commands. The `none` class is used to reduce false triggers when the controller is held still or no gesture is intended.

## Repository Structure

```text
gesture_recognition/
├── main/
│   ├── main.c
│   ├── gesture_inference_template.c
│   └── CMakeLists.txt
├── imu_samples/
├── test_samples/
├── capture_triggered.py
├── imu_gesture_classifier.py
├── imu_gesture_infer.py
├── export_svm_to_c.py
├── gesture_classifier.joblib
├── gesture_classifier_metadata.json
└── README.md
