from __future__ import print_function

import gradCamInterface

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image

import random
from keras.utils import np_utils
import time

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import TensorFlowV2Classifier

def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def executeGradCam(i, adversarial, visualize):
    # Prepare image
    if adversarial:
        img = x_test_adv[i]
    else:
        img = X_test[i]
    img_array = gradCamInterface.get_img_array(img)
    if visualize:
        plt.imshow((img))
    preds = classifier.predict(img_array)
    predicted = gradCamInterface.decode_predictions(preds, num_classes, classes)
    real_value = gradCamInterface.decode_predictions(y_test[i], num_classes, classes)
    if adversarial:
        print("Predicted adversarial example: ", predicted, "   real value: ", real_value)
    else:
        print("Predicted benign example: ", predicted)

    # Generate class activation heatmap
    heatmap = gradCamInterface.make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # Display heatmap
    if visualize:
        plt.matshow(heatmap)  # tiene que estar el eagle enable, eagle error: https://github.com/Trusted-AI/adversarial-robustness-toolbox/issues/330
        plt.show()
    img_255 = img * 255
    # https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network

    if adversarial:
        File_name = 'gradCam_example_Adversarial_image-%s_predicted-%s.jpg' % (i, predicted)
    else:
        File_name = 'gradCam_example_Benign_image-%s_predicted-%s.jpg' % (i, predicted)
    gradCamInterface.save_and_display_gradcam(img_255, heatmap, File_name)

#%% Data Preparation
# Load data
t1 = time.time()
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


visualize = False
# Visualize some of them
# Create a grid of 3x3 images
if visualize:
    print('Training set', X_train.shape, y_train.shape)
    print('Test set', X_test.shape, y_test.shape)
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow((X_train[i]))
    # Show the plot
    plt.show()
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

# Load model
epochs = 20
lr=1e-3
Model_name = 'Tensorflow_3x2Conv-3Fc_rms-lr%s_epoch%s' % (lr, epochs)
model = keras.models.load_model(Model_name)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
classifier = TensorFlowV2Classifier(model=model, clip_values=(0, 1), nb_classes=num_classes, input_shape=(32, 32, 3), loss_object=loss_object,train_step=train_step)

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

#GRAD CAM

# Remove last layer's softmax
model.layers[-1].activation = None
last_conv_layer_name = 'conv2d_5' # 'conv2d_5'#model.layers[-1].name
# Print what the top predicted class is

A  =  [ ]
i = 0
while(i < 20):
    A.append(random.randint(1, 700))
    i+=1

for image in range(0, 20):
    executeGradCam(A[image], False, False)
    executeGradCam(A[image], True, False) #adversarios
    print("     ------------------")
'''Con una imagen cargada
img_path = "ship-icon.jpg"
size= (32, 32)
img = keras.preprocessing.image.load_img(img_path, target_size=size)
# `array` is a float32 Numpy array of shape (299, 299, 3)
img_array = keras.preprocessing.image.img_to_array(img)
# We add a dimension to transform our array into a "batch"
# of size (1, 299, 299, 3)
plt.imshow(img_array)
img_array = np.expand_dims(img_array, axis=0)
model.layers[-1].activation = None
last_conv_layer_name = 'conv2d_5' # 'conv2d_5'#model.layers[-1].name
# Print what the top predicted class is
preds = model.predict(img_array)
print("Predicted benign example:", gradCamInterface.decode_predictions(preds, num_classes, classes))
heatmap = gradCamInterface.make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# Display heatmap
plt.matshow(heatmap)#tiene que estar el eagle enable, eagle error: https://github.com/Trusted-AI/adversarial-robustness-toolbox/issues/330
plt.show()
#https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network
gradCamInterface.save_and_display_gradcam_path(img_path, heatmap, "gradcam_example_path.jpg")
'''