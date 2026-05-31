# Sign Language Gesture Recognition for Human-Robot Interaction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-red.svg)](https://mediapipe.dev/)

## Overview

Real-time American Sign Language (ASL) gesture recognition comparing **Custom CNN** vs **MobileNetV2** for Human-Robot Interaction applications.

**Key Finding:** MobileNetV2 achieves 99% test accuracy with stable real-time performance, while custom CNN overfits to the dataset and fails in real-world conditions.

## Recognized Gestures

24 static ASL letters (A-Y, excluding J and Z):
A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y


## Results

| Model | Test Accuracy | Real-World Performance | Stability |
|-------|---------------|------------------------|-----------|
| CNN | 40% | Poor (biased to 'H') | Unstable |
| **MobileNetV2** | **99%** | **Good** | **Stable** |

## Limitations
Static gestures only (no J or Z)
Single hand only
Requires good lighting

