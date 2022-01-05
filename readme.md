# Towards Adversarial Robustness in UnlabeledTarget Domains

## Introduction
In the past several years, various adversarial train-ing (AT) approaches have been invented to robustify deeplearning model against adversarial attacks. However, mainstream AT methods assume the training and testing data are drawnfrom the same distribution and the training data are annotated. When the two assumptions are  violated, existing AT methodsfail because either they cannot pass knowledge learnt from a source domain to an unlabeled target domain or they areconfused by the  adversarial samples in that unlabeled space. In this paper, we first point out this new and challengingproblem—adversarial training in unlabeled target domain. We  then propose a new framework ofUnsupervised Cross-domain  Adversarial  Training(UCAT) to  tackle the above problem by effectively utilizing knowledge from the labeled source domain to enhance the representation learning in the unlabeled target domain.

## Instruction
- set up your environment by anaconda, (**python3.7, torch 1.7.0**)
- pip install torchvision=0.4
- pip install imgaug

## Datasets Folder
The Datasets Folder is used for the stroage of three datasets (Digits, Office31 and VisDA).

## Dataloader
Use prepare_data.py for the Dataloader generator for generating source and target domain dataset and dataloader.

## Model construction
Use models/model_construct.py for the initialization of the ResNet-50 used as our backbone model.

## Training and Testing
Main file for implement UCAT_SRDC, the SRDC part is attribute to the original paper.
Use command line below for training
```
CUDA_VISIBLE_DEVICES=0 python main.py
```
You can choose __trainer_ucat.py__ as training function for UCAT with naive PL or __trainer_ucat_PL.py__ as training function for UCAT with QUA-PL.

## PseudoLabeling
The implementation of __QUA-PL__ for the generation of class-wise adaptive pseudo labels is provided in __PseudoLabeling.py__.

## Different Adversarial Attacks
Different adversarial attack methods for model robustness evaluation are provided in __AdvAttacks.py__ 

