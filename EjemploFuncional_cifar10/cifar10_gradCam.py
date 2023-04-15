from __future__ import print_function

import gradCamInterface
import os
import errno
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

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

def executeGradCam(i):
    # https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network
    # Prepare image
    img_adv = x_test_adv[i]
    img_orig = X_test[i]
    list_img = []
    img_array = []
    preds = []
    predicted = []
    heatmap = []
    gradCam_img = []
    img_255 = []
    list_img.append(img_orig)
    list_img.append(img_adv)

    for ind in range(2): #ind == 0 es la imagen sin modificar
        img_array.append(gradCamInterface.get_img_array(list_img[ind]))
        preds.append(classifier.predict(img_array[ind]))
        predicted.append(gradCamInterface.decode_predictions(preds[ind], num_classes, classes))
        real_value = gradCamInterface.decode_predictions(y_test[i], num_classes, classes)

        # Generate class activation heatmap
        heatmap.append(gradCamInterface.make_gradcam_heatmap(img_array[ind], model, last_conv_layer_name))
        #plt.matshow(heatmap)  # tiene que estar el eagle enable, eagle error: https://github.com/Trusted-AI/adversarial-robustness-toolbox/issues/330

        # Display heatmap
        img_255.append(list_img[ind] * 255)
        gradCam_img.append(gradCamInterface.display_gradcam(img_255[ind], heatmap[ind]))

        if ind == 0:
            print("Predicted benign example: ", predicted[ind])
        else:
            print("Predicted adversarial example: ", predicted[ind], "   real value: ", real_value)
    print("     ------------------")
    plot_img = []
    plot_img.append(keras.preprocessing.image.array_to_img(img_255[0]))
    plot_img.append(gradCam_img[0])
    plot_img.append(keras.preprocessing.image.array_to_img(img_255[1]))
    plot_img.append(gradCam_img[1])
    return plot_img, predicted, real_value

def save_and_plot_results(num, list_of_images, predicted, real_value, attack):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(5, 5), subplot_kw={'xticks' : [], 'yticks' : []}, sharey=True, layout='compressed')#
    ind = 0;
    ind_pred=0;
    for ax in axs.flat:
        ax.imshow(list_of_images[ind])
        #Ponemos titulos y nombre de los ejes
        if ind == 0:
            ax.set_ylabel('Original')

        if ind % 2 == 0: #Los pares tendran el valor predecido
            predText = 'Predicted: %s'% (predicted[ind_pred])
            ax.set_title(predText)
            if ind > 1:
                ax.set_ylabel('Adversarial')
        else: #Los impares seran las imagenes con gradCam
            ax.set_title('GradCam')
            ind_pred+=1
        ind+=1
    suptitle = 'Real value: %s, attack method used: %s' % (real_value, attack)
    fig.suptitle(suptitle)
    try :
        os.mkdir('gradCam_examples_attack_method-%s' % (attack))
    except OSError as e :
        if e.errno != errno.EEXIST :
            raise
    File_name = 'gradCam_examples_attack_method-%s/gradCam_example_image-%s_attack_method-%s_Real-%s.jpg' % (attack, num, attack, real_value)
    fig.savefig(File_name)
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
attack = FastGradientMethod(estimator=classifier, eps=0.01)
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
last_conv_layer_name = 'conv2d_5' #model.layers[-1].name
# Print what the top predicted class is

A  =  [ ]
i = 0
while(i < 20):
    A.append(random.randint(1, 700))
    i+=1

for index in range(0, 20):
    list_of_images, predicted, real_value = executeGradCam(A[index])
    save_and_plot_results(A[index], list_of_images, predicted, real_value, 'FastGradient')

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