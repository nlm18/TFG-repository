# https://github.com/Jules-Diez/Personal-examples/blob/master/CIFAR10/Keras_CIFAR10/Keras_CIFAR10_gridsearch.py


# https://github.com/Jules-Diez/Personal-examples/blob/master/CIFAR10/Keras_CIFAR10/Keras_CIFAR10_basic.py

from __future__ import print_function
from matplotlib import pyplot

import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import regularizers, initializers
import time

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier

tf.compat.v1.disable_eager_execution()
#%% Data Preparation
# Load data
t1 = time.time()
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

print('Training set', X_train.shape, y_train.shape)
print('Test set', X_test.shape, y_test.shape)

# Visualize some of them
# Create a grid of 3x3 images
for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow((X_train[i]))
# Show the plot
pyplot.show()
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
epochs = 10
lr=1e-3
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
print(model.summary())
# Fit the model
'''model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=128)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
# Save
Model_name='Keras_3x2Conv-3Fc_rms-lr%s_epoch%s' % (lr,epochs)
model.save(Model_name)'''


# Create the ART classifier
classifier = KerasClassifier(model=model, clip_values=(0, 255), use_logits=False)
# Train the ART classifier
classifier.fit(X_train, y_train, batch_size=64, nb_epochs=10)

# Evaluate the ART classifier on benign test examples
predictions = classifier.predict(X_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))
# Print time
t2 = time.time()
print("Time: %0.2fs" % (t2 - t1))

# Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=X_test)

# Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
# Print time
t3 = time.time()
print("Time: %0.2fs" % (t3 - t2))