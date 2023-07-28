# Identification of the Adversary from a Single Adversarial Example (ICML'23)
This code is the official implementation of [Identification of the Adversary from a Single Adversarial Example](https://openreview.net/forum?id=HBrQI0tX8F).
<div align=center><img src=pics/framework.png  width="80%" height="60%"></div>


## Abstract

Deep neural networks have been shown vulnerable to adversarial examples. Even though many defense methods have been proposed to enhance the robustness, it is still a long way toward providing an attack-free method to build a trustworthy machine learning system. In this paper, instead of enhancing the robustness, we take the investigator's perspective and propose a new framework to trace the first compromised model copy in a forensic investigation manner. Specifically, we focus on the following setting: the machine learning service provider provides model copies for a set of customers. However, one of the customers conducted adversarial attacks to fool the system. Therefore, the investigator's objective is to identify the first compromised copy by collecting and analyzing evidence from only available adversarial examples. To make the tracing viable, we design a random mask watermarking mechanism to differentiate adversarial examples from different copies. First, we propose a tracing approach in the data-limited case where the original example is also available. Then, we design a data-free approach to identify the adversary without accessing the original example. Finally, the effectiveness of our proposed framework is evaluated by extensive experiments with different model architectures, adversarial attacks, and datasets.

## Dependencies
- PyTorch == 1.12.1
- Torchvision == 0.13.1
- Numpy == 1.21.5
- Adversarial-Robustness-Toolbox == 1.10.3

## Pipeline
### Pretraining
Use the following code to generate the pre-trained ResNet18 model on CIFAR-10 dataset.
```
python train_base_model.py --model_name ResNet18 --dataset_name CIFAR10
```
### Watermarking
For each model copy, we separate the base model into the head and tail (shared with all users) and only fine-tune the model head with a specific watermark while keeping the tail frozen. Here is a demo script for watermarking ResNet18 with the CIFAR-10 dataset. 
```
python train.py --model_name ResNet18 --dataset_name CIFAR10
```
### Tracing
Generate adversarial examples for each user.
```
```
Trace in the data-limited case.
```
```
Trace in the data-free case.
```
```
## Citation

If you find our work interesting, please consider giving a star :star: and cite as:
```
@article{cheng2023identification,
  title={Identification of the Adversary from a Single Adversarial Example},
  author={Cheng, Minhao and Min, Rui and Sun, Haochen and Chen, Pin-Yu},
  year={2023}
}
```
