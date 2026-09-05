# Out of Distribution Convolutional Response for Monocular Depth Estimation
Raynold Mezini

# Overview 

A lightweight convolutional depth estimation model is trained on indoor scenes, then
its internal convolutional responses are used to detect when an input falls outside
the training distribution. No retraining or OOD data is required at calibration time.

# Datasets

- NYU Depth v2 (labeled subset, 1449 RGB-D pairs) — in-distribution, used for training
  and validation. Not included in the repo; download link: 
- KITTI (800 images) — out-of-distribution evaluation only. Link:

Place both under `notebook/data/`.

# Setup

pip install torch torchvision numpy matplotlib h5py pillow

# How to run

Open `notebook/oodcoresmde.ipynb` and run all cells in order.

# Results

 RMSE => 0.698 m 
 AbsRel => 0.205 
 δ1 / δ2 / δ3 => 0.714 / 0.916 / 0.971 
 AUROC (combined) => 0.984
 FPR95 => 0.103 

# References
1. Tang, K. et al. "CORES: Convolutional Response-based Score for Out-of-distribution
   Detection." CVPR 2024.
2. Wofk, D. et al. "FastDepth: Fast Monocular Depth Estimation on Embedded Systems."
   ICRA 2019.
3. Papa, L., Russo, P., & Amerini, I. "METER: A Mobile Vision Transformer Architecture
   for Monocular Depth Estimation." IEEE TCSVT, 2023.
4. He, K. et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
5. Silberman, N. et al. "Indoor Segmentation and Support Inference from RGBD Images."
   ECCV 2012.
6. Geiger, A. et al. "Are we ready for Autonomous Driving? The KITTI Vision Benchmark
   Suite." CVPR 2012.