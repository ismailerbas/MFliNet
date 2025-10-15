# MFliNet: Macroscopic Fluorescence Lifetime Imaging Network

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

MFliNet is a novel deep learning architecture for Fluorescence Lifetime Imaging (FLI) parameter estimation that leverages Differential Transformer encoder-decoder capabilities to handle complex macroscopic imaging scenarios. Unlike existing models, MFliNet explicitly incorporates the Instrument Response Function (IRF) as an additional input alongside experimental photon time-of-arrival histograms, enabling accurate lifetime estimation across varying topographies and depth-of-field variations.

### Key Features

- **Differential Attention Mechanism**: Amplifies signal components while suppressing noise through subtractive attention mapping
- **Dual-Input Architecture**: Processes both TPSF and pixel-wise IRF for robust parameter estimation
- **Real-Time Performance**: Reduces processing time from ~6 hours (NLSF) to ~63 seconds for full datasets
- **Clinical Applications**: Suitable for fluorescence-guided surgery and in-vivo imaging

## Architecture

MFliNet implements a Differential Transformer encoder-decoder architecture with:
- Two parallel encoder streams for TPSF and IRF inputs
- Cross-attention mechanisms in decoder blocks
- Three output branches predicting: τ₁ (short lifetime), τ₂ (long lifetime), and fractional amplitude
- Learnable λ parameter for adaptive noise cancellation

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Create Conda Environment

Clone the repository
git clone https://github.com/yourusername/MFliNet.git
cd MFliNet

Create conda environment
conda env create -f environment.yml
conda activate mflinet



### Install from Source

pip install -e .



## Data Generation

Training data generation is handled by a separate pipeline. Please refer to:

**Data Generation**: [SS_RLD](https://github.com/vkp217/SS_RLD/tree/main/src/ss) - Synthetic fluorescence lifetime data generation pipeline

This external repository provides:
- MNIST-based spatial pattern generation (28×28 pixels)
- Bi-exponential fluorescence decay models
- Experimentally-derived IRFs with pixel-wise variations
- Realistic system noise characteristics

## Quick Start

### 1. Prepare Training Data

Follow the instructions in the [SS_RLD repository](https://github.com/vkp217/SS_RLD/tree/main/src/ss) to generate synthetic FLI data with:
- τ₁ range: 0.2-0.8 ns (short lifetime component)
- τ₂ range: 0.8-1.5 ns (long lifetime component)
- 176 time gates
- Pixel-wise IRF variations

### 2. Train the Model

from models.mflinet import MFliNet
from src.train import train_model

Initialize model
model = MFliNet(
num_heads=16,
key_dim=176,
ff_dim=176,
lambda_init=0.8
)

Train
history = train_model(
model,
train_data,
val_data,
epochs=100,
batch_size=32
)



### 3. Evaluate on Experimental Data

from src.evaluate import evaluate_model

Load pretrained model
model.load_weights('results/models/mflinet.h5')

Predict on experimental data
tau1, tau2, amplitude = model.predict([tpsf_data, irf_data])

## Training

Training uses:
- Adam optimizer with adaptive learning rate (initial: 0.001)
- Mean Squared Error (MSE) loss for each output branch
- 90/10 train/validation split
- L2 regularization to prevent overfitting

## Results

### Phantom Experiments

MFliNet achieved consistent lifetime estimation across 0-20mm height variations with mean fluorescence lifetime of 1.01±0.02 ns, matching NLSF accuracy while being ~343× faster.

### In-Vivo Experiments

For HER2+ breast tumor xenografts:
- **HCC1954 (A)**: τ₁ = 0.52±0.05 ns, τ₂ = 1.19±0.02 ns
- **HCC1954 (B)**: τ₁ = 0.58±0.04 ns, τ₂ = 1.18±0.03 ns

Results closely matched NLSF gold standard.

## Citation

If you use MFliNet in your research, please cite:

@article{erbas2024enhancing,
title={Enhancing fluorescence lifetime parameter estimation accuracy with differential transformer based deep learning model incorporating pixelwise instrument response function},
author={Erbas, Ismail and Pandey, Vikas and Nizam, Navid Ibtehaj and Yuan, Nanxue and Verma, Amit and Barosso, Margarida and Intes, Xavier},
journal={arXiv preprint arXiv:2411.16896},
year={2024}
}


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Copyright 2025 [RPI]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


## Acknowledgments

- Supported by NIH Grants R01CA237267, R01CA250636, R01CA271371, & R01CA250636-02S1
- Thanks to Dr. Xavier Michalet for AlliGator software support
