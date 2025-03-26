---

# Bottled Liquor Defect Detection Based on Faster R-CNN

![Python](https://img.shields.io/badge/Python-3.8-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-1.13-orange.svg) ![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A deep learning project using Faster R-CNN to detect defects in bottled liquor images. Supports training with data augmentation, validation, and model saving for identifying imperfections like cracks or contamination.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Results](#results)
7. [Visualizations](#visualizations)
8. [License](#license)

---

## Project Overview
*Bottled Liquor Defect Detection Based on Faster R-CNN* leverages the Faster R-CNN model (ResNet50-FPN backbone) to identify defects in bottled liquor images. Built with PyTorch, it processes annotated datasets to detect imperfections, supporting quality control in liquor production.

---

## Features
- Defect detection with Faster R-CNN (ResNet50-FPN).
- Data augmentation: random flips, color jitter, rotation.
- Custom dataset class for bottled liquor images and annotations.
- Training with Adam optimizer and learning rate scheduling.
- GPU support for efficient training.

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.13+ with CUDA (optional, for GPU)
- Required libraries:
  ```bash
  pip install torch torchvision numpy opencv-python
