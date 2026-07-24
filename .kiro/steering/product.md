# Product

**Temporal Straightening for Latent Planning** is a research codebase (arXiv 2603.12231) for training visual world models whose latent trajectories are "straightened" over time, and using them for goal-reaching planning.

## Core idea
Pretrained visual encoders (e.g. DINOv2) produce strong semantic features but are not tailored to planning. Inspired by the perceptual straightening hypothesis, a curvature regularizer encourages locally straight latent trajectories. This makes Euclidean distance in latent space a better proxy for geodesic distance and improves conditioning of the planning objective, yielding more stable gradient-based (GD) planning and higher success rates.

## What the code does
- **Trains** a visual world model jointly: encoder + predictor (+ optional decoder), optionally with the temporal straightening curvature regularizer.
- **Plans** in latent space using GD, CEM, or MPC to reach image/state goals across a suite of goal-reaching tasks.
- **Reproduces** Table 1 of the paper (goal-reaching success rate, mean ± std over 3 data-sampling seeds, Open-loop and MPC settings, with/without straightening).

## Tasks / environments
PointMaze (umaze, medium), PushT, Wall, and deformable (rope, granular) goal-reaching benchmarks. Datasets are shared from the DINO-WM project.

## Audience
ML researchers reproducing or extending the paper. Correctness is defined by matching the paper's reported success-rate bands, so experimental settings (learning rates, epochs, seeds, straightening strength) must match the paper exactly.

## Attribution
Adapted from the [DINO-WM](https://github.com/gaoyuezhou/dino_wm) codebase.
