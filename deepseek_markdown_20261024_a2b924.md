# Low-Light Image Enhancement for Robust Face Detection

## Overview
This repository demonstrates a common failure mode in classical computer vision: **Haar Cascade face detectors fail under low-light conditions because the input signal is degraded.**

Rather than trying to tune the detector (which is a dead end), this pipeline demonstrates that **restoration must happen before perception**. 

## The Pipeline
1. **Simulate Low Light**: We artificially darken a clear image to simulate poor illumination.
2. **Detection Failure**: The Haar cascade fails to detect faces in the dark image.
3. **Restoration**: We apply a classical restoration pipeline:
   - **Gamma Correction**: Brightens shadowed regions.
   - **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Recovers local contrast in the LAB color space.
4. **Detection Success**: The Haar cascade successfully detects the faces in the restored image.

## Visual Results
*(Note to self: Upload the images from the `results/` folder here once you run the code)*

| 1. Low-Light (Detection Fails) | 2. Restored (Detection Succeeds) |
| :---: | :---: |
| `<img src="results/1_low_light_failure.jpg" width="400">` | `<img src="results/2_restored_success.jpg" width="400">` |

## Why this matters
This project reflects my research interest in moving from brute-force detection to principled restoration. It proves that in real-world deployments, **the input data matters more than the model architecture**. This insight drives my interest in applying deep learning-based restoration (like Diffusion Models and Retinex theory) prior to downstream vision tasks.

## How to Run
1. Clone this repository.
2. Install requirements: `pip install opencv-python numpy`
3. Place a clear face image in the folder and name it `sample.jpg`.
4. Run: `python low_light_pipeline.py`