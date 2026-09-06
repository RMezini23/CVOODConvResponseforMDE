# Out of Distribution Convolutional Response for Monocular Depth Estimation
Raynold Mezini

# Overview 

A lightweight convolutional depth estimation model is trained on indoor scenes, then
its internal convolutional responses are used to detect when an input falls outside
the training distribution. No retraining or OOD data is required at calibration time.

# Datasets

- NYU Depth v2 (labeled subset, 1449 RGB-D pairs) — in-distribution, used for training
  and validation. Not included in the repo download via link: 
  https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
- KITTI (800 images) — out-of-distribution evaluation only. Link: 
   https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip
  extract into data/kitti

## Method

Model. ResNet18 or ResNet34 encoder pretrained on ImageNet with the
classification head removed, producing 512-channel features at 1/32 resolution.
The decoder runs five upsample-and-convolve stages back to full resolution and a
single depth channel. Decoder target sizes are computed by ceiling-halving the
input size rather than repeated doubling, since naive doubling produces a width
mismatch. The model predicts log-depth directly and the loss is L1 in log space.

Training. 40 epochs, AdamW at 1e-4 with weight decay 0.05, batch size 16,
input 224 × 304. Horizontal-flip augmentation on the training split only. The
checkpoint with the lowest validation loss is kept.

OOD score. Forward hooks capture activations at all four residual stages. For
each stage, the 95th percentile of activations is taken per channel and the
standard deviation across channels forms the layer score. A percentile is used
rather than a fixed top-k because layer 1 has 4,256 spatial positions and layer 4
has 70, so a percentile means the same thing at every depth. Per-layer scores are
standardised using in-distribution statistics only and averaged.

## Findings

Scoring on the mean of per-channel extremes  the formulation CORES uses for classification  gave AUROC 0.31: real separation, but inverted. KITTI responds more weakly, not more strongly, because road and sky are
textureless while indoor scenes are cluttered. Scoring on the spread of those
extremes across channels gave 0.97 in the expected direction.

The signal migrates with depth. On ResNet18 the four stages perform similarly,
peaking mid-network. On ResNet34 layer 1 is strongest (0.933) and performance falls
steadily to layer 4 (0.676). Deeper features appear more task-specialised and less
sensitive to distribution shift.

Robustness and detectability trade off. Flip augmentation improved validation
loss from 0.185 to 0.176 while AUROC fell from 0.984 to 0.973. Moving to ResNet34
improved RMSE from 0.691 to 0.597 while AUROC fell from 0.986 to 0.956. Two
independent changes, the same relationship both times.


## Limitations

Only one OOD source is tested, and indoor versus outdoor is a wide gap part of
the separation may reflect low-level texture statistics rather than semantic
novelty. Depth metrics on KITTI are not reported: ground-truth depth for the
stereo benchmark split used here was not available, and the model's predictions on
KITTI are meaningless by construction given the 0.7–10 m indoor training range.
Near-distribution shift, such as unseen indoor scenes, is untested.


# Setup

pip install torch torchvision numpy matplotlib h5py pillow

# How to run

Open `notebook/oodcoresmde.ipynb` and run all cells in order.

# Results

| Metric | ResNet18 | ResNet34 |
|---|---|---|
| RMSE (m) | 0.691 | **0.597** |
| AbsRel | 0.191 | **0.165** |
| δ1 / δ2 / δ3 | 0.714 / 0.924 / 0.978 | **0.781 / 0.946 / 0.986** |
| AUROC (combined) | **0.986** | 0.956 |
| FPR95 | **0.083** | 0.186 |

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