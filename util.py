import csv
import os
import glob
import random
import numpy as np
import cv2
from PIL import Image
import itertools
# from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def get_data(csv_file):
    """Load our data from file."""
    with open(csv_file, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)
    return data

def split_train_test(data):
    train, test = [], []
    for item in data:
        if item[0] == 'train' or item[0] == 'TRAIN':
            train.append(item)
        else:
            test.append(item)
    return train, test

def get_classes(data):
    """Extract the classes from our data. If we want to limit them,
    only return the classes we need."""
    classes = []
    for item in data:
        if item[1] not in classes:
            classes.append(item[1])
    # Sort them.
    classes = sorted(classes)
    return classes

def clean_data(data, CLIPS_LENGTH, classes=None, MAX_FRAMES=3000):
    """Limit samples to greater than the sequence length and fewer
    than N frames. Also limit it to classes we want to use."""
    data_clean = []
    if classes is None:
        for item in data:
            if int(item[3]) >= CLIPS_LENGTH and int(item[3]) <= MAX_FRAMES:
                data_clean.append(item)
    else:
        for item in data:
            if int(item[3]) >= CLIPS_LENGTH and int(item[3]) <= MAX_FRAMES and item[1] in classes:
                data_clean.append(item)
    return data_clean

def get_class_one_hot(class_str, classes):
    label_encoded = classes.index(class_str)
    # Now one-hot it.
    label_hot = to_categorical(label_encoded, len(classes))
    assert len(label_hot) == len(classes)
    return label_hot

def get_class_index(class_str, classes):
    label_encoded = classes.index(class_str)
    return label_encoded

def get_frames_for_sample(sample):
    """Given a sample row from the data file, get all the corresponding frame
    filenames."""
    path = os.path.join(sample[0], sample[1]).decode('UTF-8')
    folder_name = sample[2].decode('UTF-8')
    images = sorted(glob.glob(os.path.join(path, folder_name + '/*jpg')))
    num_frames = sample[3]
    return images, int(num_frames)

def random_start_idx(tuple_len, num_frames_per_clip, video_length):
    start_idx = []
    idx = random.randint(0, video_length - num_frames_per_clip * tuple_len)
    start_idx.append(idx)
    for i in range(1, tuple_len):
        idx = random.randint(idx + num_frames_per_clip, video_length - num_frames_per_clip * (tuple_len - i))
        start_idx.append(idx)
    return start_idx

def read_images(frames, start_idx, num_frames_per_clip):
    img_data = []
    for i in range(start_idx, start_idx + num_frames_per_clip):
        img = Image.open(frames[i])
        img = np.asarray(img)
        img_data.append(img)
    return img_data

def data_process(tmp_data, crop_size, is_train):
    img_datas = []
    crop_x = 0
    crop_y = 0

    if crop_size==224:
        resize_value=256
    else:
        resize_value=129

    if is_train and random.random()>0.5:
        flip = True
    else:
        flip = False

    if is_train and random.random()>0.8:
        cvt_color = True
    else:
        cvt_color = False

    if is_train and random.random()>0.5:
        channel1, channel2 = random.choices([0, 1, 2], k=2)
    else:
        channel1, channel2 = 0, 0

    size = crop_size
    if is_train and crop_size==112:
        size = random.choice([129, 112, 96, 84])

    if is_train and crop_size==224:
        size = random.choice([256, 224, 192, 168])

    for j in range(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if img.width > img.height:
            scale = float(resize_value) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), resize_value))).astype(np.float32)
        else:
            scale = float(resize_value) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (resize_value, int(img.height * scale + 1)))).astype(np.float32)
        if j == 0:
            if is_train:
                crop_x = random.randint(0, int(img.shape[0] - size))
                crop_y = random.randint(0, int(img.shape[1] - size))
            else:
                crop_x = int((img.shape[0] - crop_size) / 2)
                crop_y = int((img.shape[1] - crop_size) / 2)
        img = img[crop_x:crop_x + size, crop_y:crop_y + size, :]
        img = np.array(cv2.resize(img, (crop_size, crop_size))).astype(np.float32)
        img = np.asarray(img) / 127.5
        img -= 1.

        if flip:
            img = np.flip(img, axis=1)

        if cvt_color:
            img = -img

        if channel1 != channel2:
            img = Channel_splitting(img, channel1, channel2)

        img_datas.append(img)
    return img_datas

def generator_data(data, classes, num_frames_per_clip=16, crop_size=224, is_train=True):
    while True:
        np.random.shuffle(data)
        for i in range(len(data)):
            row = data[i]
            frames, num_frames = get_frames_for_sample(row)  # read all frames in video and length of the video
            start_idx = random.randint(0, num_frames - num_frames_per_clip)

            clip = read_images(frames, start_idx, num_frames_per_clip)
            clip = data_process(clip, crop_size, is_train)
            label = get_class_one_hot(row[1], classes=list(classes))
            # label = get_class_index(row[1], classes=list(classes))
            clip = np.asarray(clip)

            yield clip, label


def adjust_constrast_and_brightness(clip, alpha, beta):
    clip = clip * alpha + beta
    return clip


def add_noise_clip(clip, stddev_value=0.1):
    noise_clip = tf.random.normal(shape=clip.shape, mean=0, stddev=stddev_value)
    # noise_clip = np.random.normal(loc=0, scale=stddev_value, size=clip.shape)
    return clip + noise_clip

def Channel_splitting(clip, channel1, channel2):
    clip[..., channel1] = clip[...,channel2]
    return clip


def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
    radius = tf.compat.v1.to_int32(kernel_size / 2)
    kernel_size = radius * 2 + 1
    x = tf.compat.v1.to_float(tf.range(-radius, radius + 1))
    blur_filter = tf.exp(-tf.pow(x, 2.0) / (2.0 * tf.pow(tf.compat.v1.to_float(sigma), 2.0)))
    blur_filter /= tf.reduce_sum(blur_filter)
    # One vertical and one horizontal filter.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
        image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(
      image, blur_h, strides=[1, 1, 1, 1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(
      blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)
    return blurred


def adjust_hue(clip, delta):
    clip = tf.image.adjust_hue(clip, delta=delta)
    return clip


def data_augmentation(clip, label):
    if random.random()> 0.5:
        alpha = random.uniform(0.5, 1.5)
        beta = random.uniform(-0.5, 0.5)
        clip = adjust_constrast_and_brightness(clip, alpha, beta)
        clip = tf.clip_by_value(clip, -1., 1.)

    if random.random()>0.5:
        sigma = random.uniform(0.1, 2.0)
        clip = gaussian_blur(clip, kernel_size=7, sigma=sigma)
        clip = tf.clip_by_value(clip, -1., 1.)

    if random.random()> 0.5:
        clip = add_noise_clip(clip, 0.1)
        clip = tf.clip_by_value(clip, -1., 1.)

    if random.random()> 0.5:
        hue_value = random.uniform(0, 0.1)
        clip = adjust_hue(clip, hue_value)
        clip = tf.clip_by_value(clip, -1., 1.)

    return tf.clip_by_value(clip, -1., 1.), label