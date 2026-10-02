# Heart Sound Classification Reproducibility Project

This repository draft turns an existing B.Sc. thesis on normal-versus-abnormal phonocardiogram classification into a reproducible signal-processing and machine-learning workflow.

> **Current status:** local draft. The software pipeline and automated tests run successfully, but evaluation on the real PhysioNet/CinC 2016 dataset has not yet been completed. Synthetic-demo scores are software checks, not model-performance claims. See [`BUILD_STATUS.md`](BUILD_STATUS.md).

## Project origin and ownership

This is a reproducibility-focused redevelopment of Yousef Rajabzadeh's authentic B.Sc. thesis—not a newly invented course affiliation. The thesis topic and Python/CNN work are supported by the supplied academic material. The modular package, leakage-aware split, classical baseline, automated tests, model card, and reproducibility documentation are part of the later portfolio rebuild. Historical thesis results and newly reproduced results are kept separate.

Interview-safe description: **“This project began as my B.Sc. thesis. I later restructured it as a reproducible, tested Python project and am rerunning the evaluation with stricter recording-level separation.”**

The historical thesis reported approximately 92% average accuracy and F1 score on PhysioNet/CinC Challenge 2016 data. Those values are retained only as historical thesis claims. This repository does not claim to reproduce them until the full dataset pipeline has been executed and independently checked.

## What this draft demonstrates

- deterministic WAV loading and five-second segmentation;
- log-spectrum feature extraction implemented with NumPy;
- leakage-aware recording-level train/test splitting;
- a transparent logistic-regression baseline implemented from first principles;
- binary classification metrics and confusion matrix generation;
- an optional TensorFlow/Keras CNN architecture;
- unit tests using generated synthetic heart-like signals;
- explicit limitations and an interview-preparation guide.

## Repository structure

```text
heart-sound-ml-reproducibility/
  data/README.md
  docs/interview-preparation.md
  reports/model-card.md
  src/heart_sound_ml/
    __init__.py
    audio.py
    baseline.py
    cnn.py
    evaluate.py
    features.py
    split.py
    synthetic.py
  tests/
  scripts/run_synthetic_demo.py
  pyproject.toml
  requirements.txt
```

## Quick start

Create a virtual environment, install the project and run the tests:

```bash
python -m venv .venv
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/run_synthetic_demo.py
```

The synthetic demo is a pipeline check, not medical evidence and not a benchmark result. It generates two intentionally separable signal classes to verify that segmentation, feature extraction, training and evaluation work end to end.

## Dataset setup

Follow `data/README.md`. The repository intentionally does not redistribute PhysioNet recordings. Dataset licensing, patient grouping and the official challenge labels must be respected.

## Evaluation rules

1. Split by recording or patient identifier before creating segments.
2. Never allow segments from one recording to appear in both train and test sets.
3. Report class counts and the split seed.
4. Report precision, recall, F1, balanced accuracy and the confusion matrix, not accuracy alone.
5. Compare the CNN with the transparent baseline.
6. Treat all results as research/portfolio results, not clinical performance.

## Current status

- Pipeline and synthetic tests: implemented locally.
- Full PhysioNet reproduction: pending dataset acquisition and execution.
- Historical notebook/thesis integration: pending review before any GitHub update.
- CV inclusion: blocked until the measured full-data result is reviewed and approved.

## Ethical and technical limitations

This is an educational signal-classification project. It is not a diagnostic device. The PhysioNet dataset contains heterogeneous acquisition conditions, label uncertainty and class imbalance. A random segment-level split can cause leakage and inflated metrics; grouping by recording/patient is mandatory.
