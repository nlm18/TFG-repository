from __future__ import print_function

import pickle
import tensorflow as tf
import numpy as np
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import regularizers, initializers

from art.estimators.classification import TensorFlowV2Classifier

def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#%% Data Preparation
# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

print('Training set', X_train.shape, y_train.shape)
print('Test set', X_test.shape, y_test.shape)

X_train.shape[1:]

# Normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# One hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
model = Sequential()
# Conv Stack 1
model.add(Conv2D(input_shape=(32, 32, 3),
                 filters=32,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 kernel_initializer=initializers.he_normal(),
                 bias_initializer='zeros'))
model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 kernel_initializer=initializers.he_normal(),
                 bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.25))
# Conv Stack 2
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 kernel_initializer=initializers.he_normal(),
                 bias_initializer='zeros'))
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 kernel_initializer=initializers.he_normal(),
                 bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.25))
# Conv Stack 3
model.add(Conv2D(filters=128,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 kernel_initializer=initializers.he_normal(),
                 bias_initializer='zeros'))
model.add(Conv2D(filters=128,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 kernel_initializer=initializers.he_normal(),
                 bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.25))
#↨ FC stack
model.add(Flatten())
model.add(Dense(units=512,
                activation='relu',
                kernel_initializer=initializers.he_normal(),
                bias_initializer='zeros',
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(units=128,
                activation='relu',
                kernel_initializer = initializers.he_normal(),
                bias_initializer='zeros',
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

#%% Run & Evaluate & Save
# Set parameters and optimizer
epochs = 20
lr=1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
# Compile model
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'], run_eagerly=True)
print(model.summary())

# Create the ART classifier
classifier = TensorFlowV2Classifier(model=model, clip_values=(0, 1), nb_classes=num_classes, input_shape=(32, 32, 3), loss_object=loss_object, train_step=train_step)
#https://git.scc.kit.edu/uyxxq/adversarial-robustness-toolbox/-/blob/dev/examples/get_started_tensorflow_v2.py
# Train the ART classifier
classifier.fit(X_train, y_train, batch_size=64, nb_epochs=epochs)
model = classifier.model
predictions = classifier.predict(X_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Save
Model_name = 'Tensorflow_3x2Conv-3Fc_rms-lr%s_epoch%s' % (lr, epochs)
classifier.model.save(Model_name)