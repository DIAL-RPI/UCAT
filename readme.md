# Source code for Unsupervised Cross-domain Adversarial Training (UCAT) ####

## 1. datasets
----> Folder for storage datasets.

## 2. data/prepare_data.py
----> Dataloader generator for generating source and target domain dataset and dataloader.

## 3. models/model_construct.py
----> Init the ResNet-50 used as our backbone model

## 4. main.py 
----> Main file for implement UCAT_SRDC, the SRDC part is attribute to the original paper

## 5. trainer_ucat.py 
----> Training function for UCAT with naive PL

## 6. trainer_ucat.py 
----> Training function for UCAT with QUA-PL

## 7. PseudoLabeling.py
----> Implementation of QUA-PL to generate class-wise adaptive pseudo labels

## 8. AdvAttacks.py and ./attacks.py (./fab.py FAB attack)
----> Different attack methods for model robustness evaluation

## 9. utils/... 
----> utility functions and losses for SRDC
