# Towards Adversarial Robustness in Unlabeled Target Domains

## Introduction
Adversarial training in unlabeled target domain is a neglected but challenging problem. In our IEEE TIP [paper](https://ieeexplore.ieee.org/abstract/document/10043655), we propose a new framework of __Unsupervised Cross-domain Adversarial Training (UCAT)__ with the novel __Quantity-Quality Psseudo Labeling__ technique to tackle the above problem by effectively utilizing knowledge from the labeled source domain to enhance the representation learning in the unlabeled target domain.

## Requirements
- set up your environment by anaconda, (**python3.7, torch 1.7.0**)
- pip install torchvision=0.4
- pip install imgaug

## Datasets Folder
The Datasets Folder is used for the stroage of the datasets ([MNIST](http://yann.lecun.com/exdb/mnist/), [USPS](https://www.kaggle.com/bistaumanga/usps-dataset), [MNIST-m](https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz), [Office31](https://www.cc.gatech.edu/~judy/domainadapt/#datasets_code) and [VisDA](http://ai.bu.edu/visda-2017/)).

## Dataloader
Use prepare_data.py for the Dataloader generator for generating source and target domain dataset and dataloader.

## Model construction
Use models/model_construct.py for the initialization of the ResNet-50 used as our backbone model.

## Training and Testing
Main file for implement UCAT_SRDC, the SRDC part is attribute to the original paper.
Use command line below for training
```
python main.py
```
You can choose __trainer_ucat.py__ as training function for UCAT with naive PL or __trainer_ucat_PL.py__ as training function for UCAT with QUA-PL.

## Different Adversarial Attacks
Different adversarial attack methods for model robustness evaluation are provided in __AdvAttacks.py__ 

## Citation
```
@ARTICLE{10043655,
  author={Zhang, Jiajin and Chao, Hanqing and Yan, Pingkun},
  journal={IEEE Transactions on Image Processing}, 
  title={Toward Adversarial Robustness in Unlabeled Target Domains}, 
  year={2023},
  volume={32},
  number={},
  pages={1272-1284},
  doi={10.1109/TIP.2023.3242141}}
```
