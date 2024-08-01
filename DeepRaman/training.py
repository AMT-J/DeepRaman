import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from pathlib import Path

class SpatialPyramidPooling(Layer):
    def __init__(self, pool_list, **kwargs):
        super().__init__(**kwargs)
        self.pool_list = pool_list
        self.num_outputs_per_channel = sum([i * i for i in pool_list])

    def build(self, input_shape):
        self.nb_channels = input_shape[-1]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def call(self, x):
        input_shape = tf.shape(x)
        num_rows = input_shape[1]
        num_cols = input_shape[2]

        row_length = [tf.cast(num_rows, 'float32') / i for i in self.pool_list]
        col_length = [tf.cast(num_cols, 'float32') / i for i in self.pool_list]

        outputs = []
        for pool_num, num_pool_regions in enumerate(self.pool_list):
            for ix in range(num_pool_regions):
                for iy in range(num_pool_regions):
                    x1 = ix * col_length[pool_num]
                    x2 = ix * col_length[pool_num] + col_length[pool_num]
                    y1 = iy * row_length[pool_num]
                    y2 = iy * row_length[pool_num] + row_length[pool_num]

                    x1 = tf.cast(tf.round(x1), 'int32')
                    x2 = tf.cast(tf.round(x2), 'int32')
                    y1 = tf.cast(tf.round(y1), 'int32')
                    y2 = tf.cast(tf.round(y2), 'int32')

                    x_crop = x[:, y1:y2, x1:x2, :]
                    pooled_val = tf.reduce_max(x_crop, axis=(1, 2))
                    outputs.append(pooled_val)

        outputs = tf.concat(outputs, axis=-1)
        return outputs

def SSPmodel(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    inputA, inputB = inputs[:, 0, :], inputs[:, 1, :]

    convA1 = tf.keras.layers.Conv1D(32, kernel_size=7, strides=1, padding='same', kernel_initializer='he_normal')(inputA)
    convA1 = tf.keras.layers.BatchNormalization()(convA1)
    convA1 = tf.keras.layers.Activation('relu')(convA1)
    poolA1 = tf.keras.layers.MaxPooling1D(3)(convA1)

    convB1 = tf.keras.layers.Conv1D(32, kernel_size=7, strides=1, padding='same', kernel_initializer='he_normal')(inputB)
    convB1 = tf.keras.layers.BatchNormalization()(convB1)
    convB1 = tf.keras.layers.Activation('relu')(convB1)
    poolB1 = tf.keras.layers.MaxPooling1D(3)(convB1)

    con = tf.keras.layers.concatenate([poolA1, poolB1], axis=2)
    con = tf.expand_dims(con, -1)

    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal')(con)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation('relu')(conv1)

    spp = SpatialPyramidPooling([1, 2, 3, 4])(conv1)

    full1 = tf.keras.layers.Dense(1024, activation='relu')(spp)
    drop1 = tf.keras.layers.Dropout(0.5)(full1)
    outputs = tf.keras.layers.Dense(2, activation='sigmoid')(drop1)

    model = tf.keras.Model(inputs, outputs)
    return model

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def cleardir(path):
    for item in Path(path).rglob('*'):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            item.rmdir()

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    return dataset[permutation], labels[permutation]

def reader(dataX, dataY, batch_size):
    steps = dataX.shape[0] // batch_size
    while True:
        for step in range(steps):
            dataX_batch = dataX[step*batch_size:(step+1)*batch_size]
            dataY_batch = dataY[step*batch_size:(step+1)*batch_size]
            yield dataX_batch, dataY_batch

def load_data(datapath):
    datafileX = datapath / 'training_X.npy'
    datafileY = datapath / 'training_Y.npy'

    Xtrain0 = np.load(datafileX)
    Ytrain0 = np.load(datafileY)

    XtrainA, YtrainA = process(Xtrain0, Ytrain0)

    split_idx = int(0.9 * XtrainA.shape[0])
    XXtrain, YYtrain = XtrainA[:split_idx], YtrainA[:split_idx]
    XXvalid, YYvalid = XtrainA[split_idx:], YtrainA[split_idx:]

    return XXtrain, YYtrain, XXvalid, YYvalid

def process(Xtrain, Ytrain):
    Xtrain /= np.max(Xtrain, axis=3, keepdims=True)
    Xtrain = Xtrain.reshape(Xtrain.shape[0]*Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3], 1)
    Ytrain = Ytrain.reshape(Ytrain.shape[0]*Ytrain.shape[1], 1)
    Ytrain = tf.keras.utils.to_categorical(Ytrain)
    return randomize(Xtrain, Ytrain)

initial_learning_rate = 0.01
lr_schedule = ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=170,
    decay_rate=0.94,
    staircase=True
)

if __name__ == '__main__':
    datapath = Path('../data')
    savepath = Path('../model')
    mkdir(savepath)

    Xtrain, Ytrain, Xvalid, Yvalid = load_data(datapath)

    tf.keras.backend.clear_session()

    model = SSPmodel((2, None, 1))
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    batch_size = 100
    epochs = 50

    history = model.fit(
        reader(Xtrain, Ytrain, batch_size),
        steps_per_epoch=Xtrain.shape[0] // batch_size,
        epochs=epochs,
        validation_data=reader(Xvalid, Yvalid, batch_size),
        validation_steps=10
    )

    plt.figure(figsize=(6, 4.5))
    plt.plot(history.history['accuracy'], label='Acc_training', color='r')
    plt.plot(history.history['val_accuracy'], label='Acc_validation', color='g')
    plt.xlabel('Epoch', size=15)
    plt.ylabel('Accuracy', size=15)
    ax2 = plt.gca().twinx()
    ax2.plot(history.history['loss'], label='Loss_training', color='b')
    ax2.plot(history.history['val_loss'], label='Loss_validation', color='orange')
    ax2.set_ylabel('Loss', size=15)
    ax2.legend(loc=8)
    plt.legend(loc=7)
    plt.show()

    model.save(savepath / 'model.h5')
    del model
