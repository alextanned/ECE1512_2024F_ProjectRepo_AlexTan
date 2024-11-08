# ECE1512 24Fall Project A: Dataset Distillation with Attention Matching for Computational Efficiency in Deep Learning

### Authors: Shiqi Tan, Zhiyuan Yaoyuan

Here’s an improved version of your `README.md` file with a more formal and structured style:

# Dataset Distillation with Attention Matching for Computational Efficiency in Deep Learning

## Overview

This repository hosts the implementation of **DataDAM (Dataset Distillation with Attention Matching)**, a framework designed to enhance computational efficiency in deep learning by creating compact synthetic datasets. By employing Spatial Attention Matching (SAM) and Maximum Mean Discrepancy (MMD) regularization, DataDAM enables efficient model training on minimized datasets without significant accuracy loss. 

## Motivation

Deep learning advancements have led to remarkable performance across various domains but often at the expense of high computational and memory demands. Dataset distillation is a promising solution to this challenge. It condenses large datasets into smaller, highly informative subsets that retain essential features of the original data. Our approach, DataDAM, aims to significantly reduce resource requirements, making it suitable for applications in resource-constrained environments, such as edge devices and real-time systems.

## Key Features

- **Spatial Attention Matching (SAM):** Aligns attention maps across network layers to capture discriminative features at multiple spatial levels.
- **MMD Regularization:** Ensures distributional alignment between the synthetic dataset and the original dataset, enhancing generalization.
- **Cross-Architecture Generalization:** The synthetic datasets generated with DataDAM maintain performance across different model architectures.
- **Comparative Analysis with State-of-the-Art Methods:** Includes benchmarks against Prioritize Alignment (PAD) and Distribution Matching (DM), showing DataDAM’s superior performance in both accuracy and computational efficiency.

## Repository Structure

- **DatasetCondensation/**: Contains code for Distribution Matching (DM) method.
- **PAD/**: Directory for code implementation of Prioritize Alignment (PAD).
- **DAM.py**: Core script implementing the DataDAM approach.
- **nas.py**: Neural Architecture Search (NAS) evaluation script.
- **networks.py**: Contains network architecture definitions used in experiments.
- **train.py**: Script for training models on distilled datasets.
- **utils.py**: Utility functions supporting data handling and preprocessing.

## Installation

To set up the project, clone this repository and install the necessary dependencies listed in `requirements.txt`:

```
git clone https://github.com/alextanned/ECE1512_2024F_ProjectRepo_AlexTan_ZhiyuanYaoyuan.git
cd ECE1512_2024F_ProjectRepo_AlexTan_ZhiyuanYaoyuan
pip install -r requirements.txt
```

## Usage

### Dataset Preparation

This project supports the following datasets:
- **MNIST**: Download via torchvision or directly from the [MNIST dataset page](http://yann.lecun.com/exdb/mnist/).
- **MHIST**: Available for download from the [MHIST GitHub repository](https://github.com/kmader/PathologyMHIST).

### Running DataDAM

To execute DataDAM on a specific dataset, use the following command:

```bash
python DAM.py --dataset MNIST --method DataDAM
```

Replace `DataDAM` with `PAD` or `DM` in the `--method` argument to run alternative methods.

### Visualizing Results

Use the following command to visualize synthetic images and performance metrics:

```bash
python visualize.py --dataset MNIST
```

## Results

DataDAM has demonstrated high efficiency and accuracy across benchmark datasets, offering substantial memory and time savings:
- **MNIST**: Achieved 90.1% test accuracy using a distilled dataset with a reduced memory footprint of 0.98 GB.
- **MHIST**: Delivered significant reductions in GPU memory usage and training time, with competitive accuracy to models trained on full datasets.

## Contribution Guidelines

We welcome contributions to this project. To contribute, please fork the repository, create a feature branch, and submit a pull request for review.

## License

This project is licensed under the MIT License. For more details, please refer to the `LICENSE` file.

## References

- Sajedi et al., "DataDAM: Efficient Dataset Distillation with Attention Matching," *Proc. IEEE/CVF International Conference on Computer Vision, 2023*.
- Wang et al., "Dataset Distillation," *arXiv preprint arXiv:1811.10959, 2018*.

For any questions or further information, please contact the authors at `{alexshiqi.tan, zhiyuan.yaoyuan}@mail.utoronto.ca`.
