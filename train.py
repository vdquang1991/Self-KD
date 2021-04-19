import os
import numpy as np
import time
import argparse
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import KLDivergence, CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from model import build_resnet_18, build_resnet_34, build_resnet_50, build_resnet_101, build_resnet_152
from util import generator_data, data_augmentation, get_data, get_classes, clean_data

from call_backs import EpochCheckpoint, ModelCheckpoint_And_LoadTeacher, SaveLog


def parse_args():
    parser = argparse.ArgumentParser(description='Training Teacher self-supervised learning')
    parser.add_argument('--model', type=str, default='res18', help='resnet18/resnet34/resnet50/resnet101/resnet152')
    parser.add_argument('--clip_len', type=int, default=16, help='clip length')
    parser.add_argument('--crop_size', type=int, default=112, help='crop size')
    parser.add_argument('--temperature', type=int, default=10, help='temperature')
    parser.add_argument('--lambd', type=float, default=0.1, help='lambda factor')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--drop_rate', type=float, default=0.5, help='drop rate')
    parser.add_argument('--reg_factor', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--start_epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--save_path', type=str, default='./save_model/Self_KD/', help='save model weights')

    args = parser.parse_args()
    return args

class Distiller(Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=10,
    ):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    @tf.function
    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, tf.nn.softmax(student_predictions, axis=1))
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = student_loss + self.alpha * distillation_loss
            loss += sum(self.student.losses)

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, tf.nn.softmax(student_predictions, axis=1))

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss, "student_loss": student_loss, "distillation_loss": distillation_loss})
        return results

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, tf.nn.softmax(y_prediction, axis=1))
        loss = student_loss + sum(self.student.losses)

        # Update the metrics.
        self.compiled_metrics.update_state(y, tf.nn.softmax(y_prediction, axis=1))

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss, "student_loss": student_loss})
        return results

def build_model(model_name, input_shape, num_classes, reg_factor=1e-4, activation='softmax', drop_rate=None):
    if model_name == 'res18':
        model = build_resnet_18(input_shape, num_classes, reg_factor, activation=activation)
    elif model_name == 'res34':
        model = build_resnet_34(input_shape, num_classes, reg_factor, activation=activation, drop_rate=drop_rate)
    elif model_name == 'res50':
        model = build_resnet_50(input_shape, num_classes, reg_factor, activation=activation, drop_rate=drop_rate)
    elif model_name == 'res101':
        model = build_resnet_101(input_shape, num_classes, reg_factor, activation=activation, drop_rate=drop_rate)
    else:
        model = build_resnet_152(input_shape, num_classes, reg_factor, activation=activation, drop_rate=drop_rate)
    return model

def build_callbacks(save_path, filepath, every, startAt, monitor, mode, has_teacher=True):
    earlyStopping = EarlyStopping(monitor=monitor, patience=30, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=10, verbose=1, factor=0.1, min_lr=1e-8)

    checkpoint_path = os.path.join(save_path, 'checkpoints')
    epoch_checkpoint = EpochCheckpoint(outputPath=checkpoint_path, every=every, startAt=startAt, has_student=has_teacher)

    jsonName = 'distil_result.json'
    jsonPath = os.path.join(save_path, "output")
    if has_teacher:
        Checkpoint_and_loadTeacher = ModelCheckpoint_And_LoadTeacher(filepath=filepath, jsonPath=jsonPath,
                                                                     jsonName=jsonName, startAt=startAt,
                                                                     monitor=monitor, mode=mode, verbose=1)
        cb = [epoch_checkpoint, Checkpoint_and_loadTeacher, earlyStopping, reduce_lr]
    else:
        saveLog = SaveLog(filepath=filepath, jsonPath=jsonPath, jsonName=jsonName, startAt=startAt, verbose=1)
        cb = [epoch_checkpoint, saveLog, earlyStopping, reduce_lr]

    return cb

def train_self_KD(train_dataset, val_dataset, model_name, input_shape, classes_list, lr_init, weight_model_path,
                        start_epoch=1, reg_factor=1e-4, save_path='save_model', alpha=0.1,
                        temperature=10, batch_size=16, every=1, epochs=100, drop_rate=None):
    # Build pseudo teacher and loss weight
    pseudo_teacher = build_model(model_name, input_shape, len(classes_list), reg_factor=reg_factor,
                                 activation=None, drop_rate=drop_rate)
    pseudo_teacher.trainable = False

    # Build student model
    student_model = build_model(model_name, input_shape, len(classes_list), reg_factor=reg_factor,
                                activation=None, drop_rate=drop_rate)

    # Load weight for student model
    if start_epoch > 1:
        print('--------------------------START LOAD WEIGHT---------------------------------')
        path = os.path.join(save_path, 'checkpoints', 'epoch_' + str(start_epoch) + '.h5')
        student_model.load_weights(path)
        pseudo_teacher.load_weights(weight_model_path)
        print('--------------------------LOAD WEIGHT COMPLETED---------------------------------')

    # Build callbacks
    callback = build_callbacks(save_path=save_path, filepath=weight_model_path, every=every, startAt=start_epoch,
                               monitor='val_accuracy', mode='max')
    sgd = SGD(learning_rate=lr_init, momentum=0.99, nesterov=True)

    # Build distil model
    distiller = Distiller(student=student_model, teacher=pseudo_teacher)
    distiller.compile(
        optimizer=sgd,
        metrics=['accuracy'],
        student_loss_fn=CategoricalCrossentropy(),
        distillation_loss_fn=KLDivergence(),
        alpha=alpha,
        temperature=temperature,
    )

    # Prepare data for training phase
    data_train = tf.data.Dataset.from_generator(generator_data, (tf.float32, tf.float32),
                                                (tf.TensorShape(input_shape), tf.TensorShape([len(classes_list)])),
                                                args=[train_dataset, classes_list, input_shape[0], input_shape[1],True])
    data_val = tf.data.Dataset.from_generator(generator_data, (tf.float32, tf.float32),
                                               (tf.TensorShape(input_shape), tf.TensorShape([len(classes_list)])),
                                               args=[val_dataset, classes_list, input_shape[0], input_shape[1], False])

    data_train = data_train.map(data_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data_train = data_train.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    data_val = data_val.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Training
    distiller.fit(data_train, epochs=epochs - start_epoch + 1, verbose=1,
                  steps_per_epoch=len(train_dataset) // batch_size,
                  validation_data=data_val,
                  validation_steps=len(val_dataset) // (batch_size), callbacks=callback
                  )

def main():
    args = parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # Choose GPU for training

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    input_shape = (args.clip_len, args.crop_size, args.crop_size, 3)
    model_name = args.model
    reg_factor = args.reg_factor
    batch_size = args.batch_size
    epochs = args.epochs
    lr_init = args.lr
    start_epoch = args.start_epoch
    save_path = args.save_path
    temperature = args.temperature
    alpha = args.lambd
    drop_rate = args.drop_rate
    every = 1

    # Create folders for callback
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(os.path.join(save_path, "output")):
        os.mkdir(os.path.join(save_path, "output"))
    if not os.path.exists(os.path.join(save_path, "checkpoints")):
        os.mkdir(os.path.join(save_path, "checkpoints"))

    # Write all config to file
    f = open(os.path.join(save_path, 'config.txt'), "w")
    f.write('input shape: ' + str(input_shape) + '\n')
    f.write('model name: ' + model_name + '\n')
    f.write('reg factor: ' + str(reg_factor) + '\n')
    f.write('batch size: ' + str(batch_size) + '\n')
    f.write('numbers of epochs: ' + str(epochs) + '\n')
    f.write('lr init: ' + str(lr_init) + '\n')
    f.write('Temperature: ' + str(temperature) + '\n')
    f.write('Alpha: ' + str(alpha) + '\n')
    f.write('start epoch: ' + str(start_epoch) + '\n')
    f.write('Drop rate: ' + str(drop_rate) + '\n')
    f.close()

    # Read dataset
    train_dataset = get_data('train.csv')
    val_dataset = get_data('val.csv')
    classes_list = get_classes(train_dataset)
    print('Number of classes:', len(classes_list))
    print('Train set:', len(train_dataset))
    print('Val set:', len(val_dataset))

    weight_model_path = os.path.join(save_path, "best_" + model_name + "_.h5")

    train_dataset = clean_data(train_dataset, args.clip_len + 1, classes=classes_list, MAX_FRAMES=3000)
    val_dataset = clean_data(val_dataset, args.clip_len + 1, classes=classes_list, MAX_FRAMES=3000)
    print('Train set after clean:', len(train_dataset))
    print('Val set after clean:', len(val_dataset))

    # --------------------------------------Continuous training with Self Knowledge Distillation----------------------------------------
    train_self_KD(train_dataset, val_dataset, model_name, input_shape, classes_list, lr_init, weight_model_path,
                  start_epoch=start_epoch, reg_factor=reg_factor, save_path=save_path, alpha=alpha,
                  temperature=temperature, batch_size=batch_size, every=every, epochs=epochs, drop_rate=drop_rate)


if __name__ == '__main__':
    print(tf.__version__)
    main()