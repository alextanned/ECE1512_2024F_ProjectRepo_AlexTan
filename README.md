# ECE1512 24Fall Project A: Dataset Distillation with Attention Matching for Computational Efficiency in Deep Learning

### Authors: Shiqi Tan, Zhiyuan Yaoyuan


## Overview

This repository hosts the code file for our ECE1512 24Fall Project A, an implementation of **DataDAM (Dataset Distillation with Attention Matching)**, a framework designed to enhance computational efficiency in deep learning by creating compact synthetic datasets. By employing Spatial Attention Matching (SAM) and Maximum Mean Discrepancy (MMD) regularization, DataDAM enables efficient model training on minimized datasets without significant accuracy loss. 

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


## Results

The DataDAM framework has demonstrated substantial computational savings and maintained high accuracy levels across benchmark datasets when compared to training on full datasets. Here is a summary of the results:

| Dataset          | Test Accuracy | Train Accuracy | GPU Memory Usage | Training Time |
|------------------|---------------|----------------|-------------------|---------------|
| MNIST (Full)     | 99.1%         | 97.3%         | 4.67 GB          | 190.5 s       |
| MNIST (Cond.)    | 90.1%         | 89.0%         | 0.98 GB          | 30.2 s        |
| MHIST (Full)     | 81.2%         | 91.2%         | 24.0 GB          | 80.3 s        |
| MHIST (Cond.)    | 62.0%         | 100.0%        | 2.90 GB          | 10.5 s        |


1. **Memory and Computational Efficiency**:
   - The condensed MNIST dataset requires only **0.98 GB** of GPU memory and completes training in **30.2 seconds**, compared to **4.67 GB** and **190.5 seconds** for the full dataset. This translates to significant savings in both memory and training time.
   - For the MHIST dataset, the condensed version also shows substantial reductions, needing only **2.90 GB** of memory and **10.5 seconds** for training, compared to **24.0 GB** and **80.3 seconds** for the full dataset.

2. **Performance Trade-offs**:
   - The accuracy of models trained on the condensed MNIST dataset (90.1% test accuracy) is slightly lower than those trained on the full dataset (99.1%), but the reduced memory and time requirements make it an effective option for resource-constrained environments.
   - The MHIST dataset, being more complex, shows a more significant drop in test accuracy when condensed (62.0% vs. 81.2% with the full dataset). However, it still achieves notable accuracy while significantly reducing the resource demands, making it useful for applications where speed and memory efficiency are prioritized over the highest accuracy.

3. **Generalization Ability**:
   - While condensed datasets show a drop in test accuracy, they retain the core features necessary for classification tasks, as indicated by relatively high train accuracies. In particular, the condensed MNIST dataset shows strong generalization, as the reduction in test accuracy remains within an acceptable range given the savings.

These results highlight DataDAM’s potential to make deep learning more accessible in resource-constrained environments by reducing the computational and memory footprint without overly compromising model accuracy. 


## License

This project is licensed under the MIT License. For more details, please refer to the `LICENSE` file.

## References

- Sajedi et al., "DataDAM: Efficient Dataset Distillation with Attention Matching," *Proc. IEEE/CVF International Conference on Computer Vision, 2023*.
- Wang et al., "Dataset Distillation," *arXiv preprint arXiv:1811.10959, 2018*.

For any questions or further information, please contact the authors at `{alexshiqi.tan, zhiyuan.yaoyuan}@mail.utoronto.ca`.
