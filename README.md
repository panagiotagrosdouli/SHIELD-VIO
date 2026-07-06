# Self-Healing VIO for Robust UAV Navigation

A research-oriented visual-inertial odometry project investigating whether a VIO system can predict imminent tracking degradation and proactively adapt its sensing and recovery strategy under challenging visual conditions.

## Core Research Question

Can a visual-inertial SLAM/VIO system predict tracking degradation before failure and adapt its sensing strategy to improve robustness during blur, low light, noise, and frame dropout?

## Motivation

Visual-inertial odometry is central to UAV autonomy, but performance can degrade sharply under perceptual failures such as motion blur, poor illumination, texture scarcity, dynamic objects, and camera dropout. Classical pipelines often react only after tracking has already failed.

This project explores a self-healing VIO architecture that monitors online health signals, predicts failure risk, and triggers adaptive recovery actions before catastrophic tracking loss.

## Proposed Contributions

1. Online VIO health estimation from visual, inertial, and image-quality signals.
2. Failure prediction before tracking collapse.
3. Adaptive recovery policies for degraded visual conditions.
4. Controlled degradation benchmark for robust UAV navigation.
5. Reproducible evaluation on EuRoC MAV and later TUM-VI.

## Initial Architecture

```text
RGB Images + IMU
       |
       v
Visual-Inertial Frontend
       |
       v
Baseline VIO / ORB-SLAM3
       |
       +-------------------+
       |                   |
       v                   v
Trajectory Estimate   Health Monitor
                           |
                           v
                    Failure Probability
                           |
                           v
                    Recovery Manager
                           |
                           v
                 Adaptive VIO Strategy
```

## Evaluation Plan

Initial experiments will compare clean and degraded sequences using:

- Absolute Trajectory Error (ATE)
- Relative Pose Error (RPE)
- tracking failure count
- recovery event count
- runtime overhead
- failure prediction lead time

## Repository Structure

```text
configs/          Experiment configurations
experiments/      Benchmark and ablation definitions
paper/            Paper draft and generated artifacts
scripts/          Dataset preparation and experiment scripts
src/              Core self-healing VIO modules
tests/            Unit and smoke tests
results/          Local generated outputs, ignored by git
```

## Status

This repository is a clean research-focused successor to an earlier adaptive SLAM prototype. The first milestone is to establish a reproducible ORB-SLAM3/EuRoC baseline before adding learned health prediction and self-healing recovery policies.
