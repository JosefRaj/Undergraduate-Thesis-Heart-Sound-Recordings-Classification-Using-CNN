# Human Heart Sound Classification based on CNN

This repository contains the Python code and documentation for my B.Sc. thesis in Electrical Engineering (Telecommunications) at Ferdowsi University of Mashhad. The project focuses on classifying human heart sounds into normal and abnormal categories using Convolutional Neural Networks (CNN).

## Overview
Heart sounds provide valuable diagnostic information regarding the mechanical activity of the heart and its valves. However, distinguishing these sounds using a traditional stethoscope is highly challenging due to the limitations of human hearing, the lack of stability in heart sounds, and overlapping respiratory noises. This project utilizes time-frequency features extracted from phonocardiogram (PCG) signals to automatically detect cardiac abnormalities.

## Dataset
* The model is trained and evaluated on the **PhysioNet/CinC Challenge 2016 dataset**. 
* This dataset includes high-quality PCG recordings of both healthy individuals and patients with various heart conditions (such as valve defects or coronary artery disease). 
* All recorded sounds have a sampling frequency of 2000 Hz.

## Methodology

### 1. Preprocessing
* **Normalization:** The audio signals were normalized using their maximum and minimum values.
* **Segmentation:** To ensure data consistency, all audio files were converted into uniform 5-second segments.
* **Spectrogram Generation:** The audio signals were transformed into spectrograms (resized to 128x128x3 pixels) using image processing techniques. Data augmentation techniques, such as random rotations, were also applied.
* **Encoding:** Labels were mapped using the One-Hot Encoding method.

### 2. Model Architecture
* The project employs a Sequential CNN architecture.
* The network features multiple layers, including `Conv2D`, `MaxPooling2D`, `Dropout`, `Flatten`, and `Dense` layers.
* The model utilizes the `adamax` optimizer and uses `categorical_crossentropy` as the loss function.
* The evaluation metric used during training is `binary_accuracy`.

## Results
* The proposed model achieved an **average accuracy of 92%** in classifying normal and abnormal heart sounds. 
* Similar average scores (92%) were achieved for other metrics, such as the F1-score.

## Author
**Yousef Rajabzadeh**
* B.Sc. in Electrical Engineering (Telecommunications), Ferdowsi University of Mashhad
* Supervisor: Dr. Hossein Zamiri
