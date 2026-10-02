# Model card draft

## Intended use

Educational binary classification of phonocardiogram recordings for portfolio and reproducibility study. Not intended for diagnosis, triage or clinical deployment.

## Dataset

Planned: PhysioNet/CinC Challenge 2016. The exact version, label conversion and grouping strategy must be recorded when the dataset is downloaded.

## Models

- Transparent logistic-regression baseline on compact log-spectrum statistics.
- Optional spectrogram CNN implemented in TensorFlow/Keras.

## Evaluation

Required metrics: class-wise precision/recall, F1, balanced accuracy, confusion matrix and support. Split at patient or recording level before segmentation.

## Current results

No full-dataset result has been approved. Synthetic demo metrics validate code paths only and must never be reported as model performance.

## Known risks

- segment leakage between train and test;
- class imbalance;
- recording-device and site bias;
- label uncertainty;
- inflated accuracy from tuning on the test set;
- limited external validity.
