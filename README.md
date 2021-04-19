# Online Teaching Yourself: An Self-Knowledge Distillation Approach to Action Recognition

By Duc-Quang Vu, Ngan Le, Jia-Ching Wang

## Overview

<p align="center">
  <img width="800" alt="fig_method" src="https://github.com/vdquang1991/Self-KD/blob/main/model/Self-kD.png">
</p>

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

### Evaluation

~~~
python eval.py --model='res18' --clip_len=16 --crop_size=112 --gpu 0
~~~

## Further details
The directory `./train/` `./val/` and `./test/` contains the training videos and val videos and test videos, respectively. Each video converted to a folder contains RGB frames with fps=25. 
To extract frames from the video, you can use `ffmpeg` command. 
To create .csv files, run the command in the following:

~~~
python make_csvfiles.py
~~~

The structure of the csv files is as follows:
~~~
<ROOT_PATH>,<LABEL>,<FOLDER>,<#FRAMES>
~~~
For examples:
~~~
train,SalsaSpin,v_SalsaSpin_g23_c02,134
train,SalsaSpin,v_SalsaSpin_g12_c02,160
train,SalsaSpin,v_SalsaSpin_g20_c01,134
~~~
In which, `train` denotes root path and indicate type of videos (train/val/test).
`SalsaSpin` is the label, `v_SalsaSpin_g23_c02` is the folder that contains all RGB frames of a video.
`134` is the number of the frames in the folder.



