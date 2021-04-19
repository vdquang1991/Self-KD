# import the necessary packages
# from keras.callbacks import Callback
import os
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import BaseLogger
import numpy as np
import json


class EpochCheckpoint(Callback):
    def __init__(self, outputPath, every=5, startAt=1, has_student=False):
        # call the parent constructor
        super(Callback, self).__init__()

        # store the base output path for the model, the number of
        # epochs that must pass before the model is serialized to
        # disk and the current epoch value
        self.outputPath = outputPath
        self.every = every
        self.intEpoch = startAt
        self.has_Student = has_student

    def on_epoch_end(self, epoch, logs={}):
        # check to see if the model should be serialized to disk
        if (self.intEpoch) % self.every == 0:
            p = os.path.sep.join([self.outputPath, "epoch_{}.h5".format(self.intEpoch)])

            # Save current model weight
            if self.has_Student:
                save_model = self.model.student
            else:
                save_model = self.model
            save_model.save_weights(p, overwrite=True)

            # Delete old model weight
            old_p = os.path.sep.join([self.outputPath, "epoch_{}.h5".format(self.intEpoch - self.every)])
            if os.path.exists(old_p):
                os.remove(old_p)

            # Save lr every epoch
            lr_path = os.path.sep.join([self.outputPath, "config_lr.txt"])
            f = open(lr_path, "a")
            f.write("LR at epoch {}: {}\n".format(self.intEpoch, self.model.optimizer.learning_rate))
            f.close()


        # increment the internal epoch counter
        self.intEpoch += 1


class ModelCheckpoint_And_LoadTeacher(BaseLogger):
    def __init__(self, filepath, jsonPath=None, jsonName=None, startAt=0, monitor='val_accuracy', mode='max', verbose=0):
        super(ModelCheckpoint_And_LoadTeacher, self).__init__()
        self.filepath = filepath
        self.jsonPath = jsonPath
        self.jsonName = jsonName
        self.jsonfile = os.path.join(self.jsonPath, self.jsonName)
        self.startAt = startAt
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose

        if self.mode == 'max':
            self.monitor_op = np.greater
            self.current_best = -np.Inf
        else:
            self.monitor_op = np.less
            self.current_best = np.Inf

    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}

        # if the JSON history path exists, load the training history
        if self.jsonfile is not None:
            if os.path.exists(self.jsonfile):
                self.H = json.loads(open(self.jsonfile).read())

                if self.startAt > 1:
                    if self.mode == 'max':
                        self.current_best = max(self.H[self.monitor])
                    else:
                        self.current_best = min(self.H[self.monitor])
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        current = logs.get(self.monitor)
        print('\nCurrent best accuracy: ', self.current_best)

        if self.monitor_op(current, self.current_best):
            if self.verbose > 0:
                print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                      ' saving model to %s'
                      % (epoch + 1, self.monitor, self.current_best, current, self.filepath))
            self.model.student.save_weights(self.filepath, overwrite=True)
            self.current_best = current
            print('-------------Load new weight for teacher model------------')
            self.model.teacher.load_weights(self.filepath)

        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(float(v))
            self.H[k] = l

        # check to see if the training history should be serialized
        # to file
        if self.jsonfile is not None:
            f = open(self.jsonfile, "w")
            f.write(json.dumps(self.H))
            f.close()


class SaveLog(BaseLogger):
    def __init__(self, filepath, jsonPath=None, jsonName=None, startAt=0, verbose=0):
        super(SaveLog, self).__init__()
        self.filepath = filepath
        self.jsonPath = jsonPath
        self.jsonName = jsonName
        self.jsonfile = os.path.join(self.jsonPath, self.jsonName)
        self.startAt = startAt
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}

        # if the JSON history path exists, load the training history
        if self.jsonfile is not None:
            if os.path.exists(self.jsonfile):
                self.H = json.loads(open(self.jsonfile).read())

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(float(v))
            self.H[k] = l

        # check to see if the training history should be serialized
        # to file
        if self.jsonfile is not None:
            f = open(self.jsonfile, "w")
            f.write(json.dumps(self.H))
            f.close()