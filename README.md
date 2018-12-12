# deepdefense.pytorch
Code for NeurIPS 2018 paper [Deep Defense: Training DNNs with Improved Adversarial Robustness](https://papers.nips.cc/paper/7324-deep-defense-training-dnns-with-improved-adversarial-robustness).

Deep Defense is recipe to improve the robustness of DNNs to adversarial perturbations. We integrate an adversarial perturbation-based regularizer into the training objective, such that the obtained models learn to resist potential attacks in a principled way.

## Environments
* Python 3.5
* PyTorch 0.4.1
* glog 0.3.1

## Datasets and Reference Models
For fair comparison with DeepFool, we follow it to use [matconvnet](https://github.com/vlfeat/matconvnet/releases/tag/v1.0-beta24) to pre-process data and train reference models for MNIST and CIFAR-10.

Please download processed datasets and reference models (including MNIST and CIFAR-10) at [download link](https://drive.google.com/open?id=15xoZ-LUbc9GZpTlxmCJmvL_DR2qYEu2J).

## Usage
To train a Deep Defense LeNet model using default parameters on MNIST:

```
python3 deepdefense.py --pretest --dataset mnist --arch LeNet
```

Argument ```--pretest``` indicates evaluating performance before fine-tuning, thus we can check the performance of reference model.

Currently we've implemented ```MLP``` and ```LeNet``` for mnist, and ```ConvNet``` for CIFAR-10.

## Citation
Please cite our work in your publications if it helps your research:

```
@inproceedings{yan2018deep,
  title={Deep Defense: Training DNNs with Improved Adversarial Robustness},
  author={Yan, Ziang and Guo, Yiwen and Zhang, Changshui},
  booktitle={Advances in Neural Information Processing Systems},
  pages={417--426},
  year={2018}
}
```
