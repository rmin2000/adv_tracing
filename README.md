### Identification of the Adversary from a Single Adversarial Example (ICML 2023)
This code is the official implementation of [Identification of the Adversary from a Single Adversarial Example](https://openreview.net/forum?id=HBrQI0tX8F).

----
<div align=center><img src=pics/framework.png  width="80%" height="60%"></div>

### Abstract

Deep neural networks have been shown vulnerable to adversarial examples. Even though many defense methods have been proposed to enhance the robustness, it is still a long way toward providing an attack-free method to build a trustworthy machine learning system. In this paper, instead of enhancing the robustness, we take the investigator's perspective and propose a new framework to trace the first compromised model copy in a forensic investigation manner. Specifically, we focus on the following setting: the machine learning service provider provides model copies for a set of customers. However, one of the customers conducted adversarial attacks to fool the system. Therefore, the investigator's objective is to identify the first compromised copy by collecting and analyzing evidence from only available adversarial examples. To make the tracing viable, we design a random mask watermarking mechanism to differentiate adversarial examples from different copies. First, we propose a tracing approach in the data-limited case where the original example is also available. Then, we design a data-free approach to identify the adversary without accessing the original example. Finally, the effectiveness of our proposed framework is evaluated by extensive experiments with different model architectures, adversarial attacks, and datasets.

### Dependencies
- PyTorch == 1.12.1
- Torchvision == 0.13.1
- Numpy == 1.21.5
- Adversarial-Robustness-Toolbox == 1.10.3

### Pipeline
#### Pretraining
Use the following script to generate the pre-trained ResNet18 model on CIFAR-10 dataset. For Tiny-ImageNet, you may need to download the dataset from this [link](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and move the data to your data directory.
```
python train_base_model.py --model_name ResNet18 --dataset_name CIFAR10
```
#### Watermarking
For each model copy, we separate the base model into the head and tail (shared with all users) and only fine-tune the model head with a specific watermark while keeping the tail frozen. Here is a demo script for watermarking ResNet18 with the CIFAR-10 dataset. 
```
python train.py --model_name ResNet18 --dataset_name CIFAR10
```
#### Tracing
You could use the following script to generate adversarial examples for each user. In our demo, we apply the [Bandit](https://arxiv.org/abs/1807.07978) and generate 10 adversarial examples for each user (50*10 in total).
```
python -m attacks.bandit --model_name ResNet18 --dataset_name CIFAR10 -M 50 -n 10
```
We introduce two scenarios for tracing, namely the data-limited setting (with original image) and the data-free setting (without original image). The following script works in the data-limited case, and here we only take one adversarial example for each user to identify the adversary. 
```
python trace_data_limited.py --model_name ResNet18 --dataset_name CIFAR10 --alpha 0.9 --attack Bandit -M 50 -n 1
```
Trace in the data-free case.
```
python trace_data_free.py --model_name ResNet18 --dataset_name CIFAR10 --alpha 0.5 --attack Bandit -M 50 -n 1
```
### Citation

If you find our work interesting, please consider giving a star :star: and cite as:
```
@article{cheng2023identification,
  title={Identification of the Adversary from a Single Adversarial Example},
  author={Cheng, Minhao and Min, Rui and Sun, Haochen and Chen, Pin-Yu},
  year={2023}
}
```
