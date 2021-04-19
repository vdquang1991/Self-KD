# Online Teaching Yourself: An Self-Knowledge Distillation Approach to Action Recognition

By Duc-Quang Vu, Ngan Le, Jia-Ching Wang

## Overview


## Running the code

### Requirements
- Python3
- Tensorflow (>=2.3.0)
- numpy 
- Pillow
- opencv

### Training

In this code, you can reproduce the experiment results of classification task in submitted paper.
The datasets are all open-sourced, so it is easy to download.
Example training settings are for ResNet18 on Kinetics400.
Detailed hyperparameter settings are enumerated in the paper.

- Training with Self-KD
~~~
python train.py --model='res18' --clip_len=16 --crop_size=112 --temperature=10 --lambd=0.1 \
--gpu 0 --batch_size=16
~~~

