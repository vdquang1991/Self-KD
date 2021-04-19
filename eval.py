import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from model import build_resnet_18, build_resnet_50
from util import get_data, get_classes, clean_data
from PIL import Image
import cv2
import argparse
import random

model_path = 'save_model/semi_10percent_onemodel/best_model.h5'

def parse_args():
    parser = argparse.ArgumentParser(description='Testing model')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--model', type=str, default='res18', help='resnet18/resnet34/resnet50/resnet101/resnet152')
    parser.add_argument('--clip_len', type=int, default=16, help='clip length')
    parser.add_argument('--crop_size', type=int, default=112, help='crop size')
    parser.add_argument('--num_classes', type=int, default=101, help='num_classes')
    args = parser.parse_args()
    return args

def read_dataset(clip_len):
    # Read dataset
    test_dataset = get_data('test.csv')
    classes_list = get_classes(test_dataset)
    print('Number of classes:', len(classes_list))
    print('Test set:', len(test_dataset))
    test_dataset = clean_data(test_dataset, clip_len + 1, classes=classes_list, MAX_FRAMES=3000)
    print('Test set after clean:', len(test_dataset))
    return test_dataset, classes_list

def load_model_and_weight(model_name, input_shape, num_classes, model_path='epoch_228.h5', reg_factor=5e-4, drop_rate=0.5):
    model = None
    if model_name == 'res18':
        model = build_resnet_18(input_shape, num_classes, reg_factor, activation=None, drop_rate=drop_rate)
    elif model_name == 'res50':
        model = build_resnet_50(input_shape, num_classes, reg_factor, activation=None, drop_rate=drop_rate)
    else:
        print('Error model name')
    model.load_weights(model_path)
    model.trainable = False
    loss = CategoricalCrossentropy()
    model.compile(optimizer='sgd', loss=loss, metrics=['accuracy'])
    return model

def get_frames_for_sample(sample):
    """Given a sample row from the data file, get all the corresponding frame
    filenames."""
    path = os.path.join(sample[0], sample[1])
    folder_name = sample[2]
    images = sorted(glob.glob(os.path.join(path, folder_name + '/*jpg')))
    num_frames = sample[3]
    return images, int(num_frames)

def read_images(frames, start_idx, num_frames_per_clip):
    img_data = []
    for i in range(start_idx, start_idx + num_frames_per_clip):
        img = Image.open(frames[i])
        img = np.asarray(img)
        img_data.append(img)
    return img_data

def data_process(tmp_data, crop_size):
    img_datas = []
    crop_x = 0
    crop_y = 0

    if crop_size==224:
        resize_value=256
    else:
        resize_value=129

    for j in range(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if img.width > img.height:
            scale = float(resize_value) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), resize_value))).astype(np.float32)
        else:
            scale = float(resize_value) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (resize_value, int(img.height * scale + 1)))).astype(np.float32)
        if j == 0:
            crop_x = int((img.shape[0] - crop_size) / 2)
            crop_y = int((img.shape[1] - crop_size) / 2)
        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
        img = np.asarray(img) / 127.5
        img -= 1.
        img_datas.append(img)
    return img_datas

def get_class_one_hot(class_str, classes):
    label_encoded = classes.index(class_str)
    return label_encoded

def softmax(x):
    result = np.exp(x) / sum(np.exp(x))
    return result

def test(test_dataset, model, classes_list, clip_len, crop_size):
    accuracy = 0.
    accuracy_softmax = 0.0
    for (i, row) in enumerate(test_dataset):
        frames, num_frames = get_frames_for_sample(row)
        repeat = num_frames // clip_len
        X = []
        predict_softmax = []
        label = get_class_one_hot(row[1], classes=classes_list)
        if i%100 == 0:
            print('Processing video at {} per {}.'.format(i, len(test_dataset)))
        if repeat > 30:
            repeat = 30
        for start_idx in range(0, repeat):
            clip = read_images(frames, start_idx * clip_len, clip_len)
            clip = data_process(clip, crop_size)
            clip = np.asarray(clip)
            X.append(clip)

        predict = model.predict(np.asarray(X))
        for k in range(repeat):
            result_softmax = softmax(predict[k])
            predict_softmax.append(result_softmax)
        predict_softmax = np.asarray(predict_softmax)
        predict_softmax = np.sum(predict_softmax, axis=0)
        predict_label_softmax = np.argmax(predict_softmax)
        if predict_label_softmax == label:
            accuracy_softmax +=1.0

        predict = np.sum(predict, axis=0)
        predict_label = np.argmax(predict)
        if predict_label == label:
            accuracy +=1.0
    print('Accuracy = ', accuracy/len(test_dataset))
    print('Accuracy softmax = ', accuracy_softmax/len(test_dataset))
    return accuracy/len(test_dataset)



def main():
    args = parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) # Choose GPU for training

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    clip_len = args.clip_len
    crop_size = args.crop_size
    num_classes = args.num_classes
    input_shape = (clip_len, crop_size, crop_size, 3)

    test_dataset, classes_list = read_dataset(clip_len)
    print(classes_list)
    model = load_model_and_weight(args.model, input_shape, num_classes, model_path)

    accuracy = test(test_dataset, model, classes_list, clip_len, crop_size)

if __name__ == '__main__':
    print(tf.__version__)
    main()
