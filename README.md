# Hybrid Quantum Generative Adversarial Networks (HQGAN)

## Introduction

This project explores the implementation and research of Hybrid Quantum Generative Adversarial Networks (HQGAN). HQGAN is an innovative approach that combines the power of quantum computing with the generative capabilities of GANs to create quantum-enhanced generative models. This repository contains the code and resources needed to understand, implement, and experiment with HQGANs.

## Prerequisites

To ensure a smooth setup process, it is highly recommended to use Conda or Miniconda for managing dependencies and creating a virtual environment. Conda helps to avoid potential conflicts between package dependencies and makes it easier to manage different environments.

### Install Conda/Miniconda

If you don't have Conda or Miniconda installed, you can download and install Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).

## Installation

Follow the steps below to set up the environment and install the necessary packages.

### 1. Create Conda Environment

```bash
conda create --name hqgan python=3.8
```

### 2. Activate Conda Environment
```bash
conda activate hqgan
```

### 3. Install requirement packages
```bash
pip install .
```

### 4. Run the Program

You can run the program either through a Python script or a Jupyter notebook provided in the `src` folder. Additionally, you can configure various model settings in the `config.yml` file before running the program.

**Option 1: Run using `main.py`**

```bash
python src/main.py
```

**Option 2: Run using `main.ipynb**
If you prefer to run the program interactively, you can use Jupyter Notebook:

```bash
jupyter notebook src/main.ipynb
```

### Contributing
Feel free to fork this repository and make your own contributions. Pull requests are welcome!
