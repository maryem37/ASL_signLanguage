# Sign Language Gesture Detector

This project aims to detect and recognize sign language gestures using computer vision and machine learning techniques. It provides a real-time system that captures hand gestures through a webcam, processes them, and predicts the corresponding sign language characters.

## Demo Video
https://github.com/En1gma02/Sign-Language-Gesture-Detector/assets/136985129/02323a1f-70fa-4c33-a837-f80fde8d4cb1

## Overview

The project consists of the following components:

1. **Collecting Images**: A script captures images from the webcam, organizes them into folders based on classes (sign language gestures), and saves them for later use as training data.

2. **Creating Dataset**: Hand landmarks are detected in the collected images using the MediaPipe library. These landmarks are extracted and saved along with their corresponding labels as a dataset.

3. **Training Classifier**: A Random Forest Classifier is trained using the extracted hand landmarks as features and the corresponding labels as targets.

4. **Running the Model**: The trained model is loaded, and real-time video from the webcam is captured. Hand landmarks are detected in the video feed, fed into the model for prediction, and the predicted sign language gesture is displayed on the video feed.

## Usage

1. **Collecting Images**: Run the `collect_images.py` script to capture images for training. Adjust the number of images per class and dataset size as needed.

2. **Creating Dataset**: Run the `create_dataset.py` script to extract hand landmarks from the collected images and save them along with labels as a dataset (`data.pickle`).

3. **Training Classifier**: Execute the `train_classifier.py` script to train the Random Forest Classifier using the dataset generated in the previous step.

4. **Running the Model**: Run the `inference_classifier.py` script to start the real-time sign language gesture detection system.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- scikit-learn

## Acknowledgments

This project was inspired by the need for accessible communication tools for individuals with hearing impairments. Special thanks to the contributors of OpenCV, MediaPipe, and scikit-learn for their valuable libraries and resources.
