# MFliNet
Enhancing Fluorescence Lifetime Imaging with Differential Transformer
# MFliNet: Macroscopic Fluorescence Lifetime Imaging Network

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

MFliNet is a novel deep learning architecture for Fluorescence Lifetime Imaging (FLI) parameter estimation that leverages Differential Transformer encoder-decoder capabilities to handle complex macroscopic imaging scenarios[file:21]. Unlike existing models, MFliNet explicitly incorporates the Instrument Response Function (IRF) as an additional input alongside experimental photon time-of-arrival histograms, enabling accurate lifetime estimation across varying topographies and depth-of-field variations[file:21].

### Key Features

- **Differential Attention Mechanism**: Amplifies signal components while suppressing noise through subtractive attention mapping[file:21]
- **Dual-Input Architecture**: Processes both TPSF and pixel-wise IRF for robust parameter estimation[file:21]
- **Real-Time Performance**: Reduces processing time from ~6 hours (NLSF) to ~63 seconds for full datasets[file:21]
- **Clinical Applications**: Suitable for fluorescence-guided surgery and in-vivo imaging[file:21]

## Architecture

MFliNet implements a Differential Transformer encoder-decoder architecture with:
- Two parallel encoder streams for TPSF and IRF inputs[file:21]
- Cross-attention mechanisms in decoder blocks[file:21]
- Three output branches predicting: τ₁ (short lifetime), τ₂ (long lifetime), and fractional amplitude[file:21]
- Learnable λ parameter for adaptive noise cancellation[file:21]

![Architecture Diagram](results/figures/architecture.png)

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Create Conda Environment

