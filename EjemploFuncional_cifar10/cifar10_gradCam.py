from __future__ import print_function

import art.attacks.evasion

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
#Mas ataques: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#fast-gradient-method-fgm
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent
from art.estimators.classification import TensorFlowV2Classifier
import pickle

# ------------------------ Funciones auxiliares ---------------------------------
def guardar_datos(datos, filename):
    with open(filename, "wb") as f:
        pickle.dump(datos, f)

def cargar_datos(filename):
     with open(filename, "rb") as f:
         return pickle.load(f)

def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def getAttackMethod(name, classifier, epsilon):
    if name == 'FastGradientMethod':
        return FastGradientMethod(estimator=classifier, eps=epsilon)
    elif name == 'BasicIterativeMethod':
        return BasicIterativeMethod(estimator=classifier, eps=epsilon, eps_step=0.5, max_iter=100)
    elif name == 'ProjectedGradientDescent':
        return ProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=0.5, max_iter=100)

def executeGradCam(num, epsilon, n_iter):
    # https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network
    # Prepare image
    img_orig = X_test[num]
    list_img = [img_orig]
    for i in range(0, len(epsilon)):
        list_img.append(x_test_adv[i+n_iter*len(epsilon)][num])
    img_array = []
    preds = []
    predicted = []
    heatmap = []
    gradCam_img = []
    img_255 = []
    plot_img = []
    # Para la lista de imagenes que tendra la forma: [imagOriginal, adv_eps1, adv_eps2...]
    for ind in range(0, len(list_img)): #ind == 0 es la imagen sin modificar
        img_array.append(gradCamInterface.get_img_array(list_img[ind]))
        preds.append(classifier.predict(img_array[ind]))
        predicted.append(gradCamInterface.decode_predictions(preds[ind], num_classes, classes))
        real_value = gradCamInterface.decode_predictions(y_test[num], num_classes, classes)

        # Generate class activation heatmap
        heatmap.append(gradCamInterface.make_gradcam_heatmap(img_array[ind], model, last_conv_layer_name))
        #plt.matshow(heatmap)  # tiene que estar el eagle enable, eagle error: https://github.com/Trusted-AI/adversarial-robustness-toolbox/issues/330

        # Display heatmap
        img_255.append(list_img[ind] * 255)
        gradCam_img.append(gradCamInterface.display_gradcam(img_255[ind], heatmap[ind]))

        if ind == 0:
            print("Predicted benign example: ", predicted[ind])
        else:
            print("AttackMethod: %s with epsilon = %s" % (attackName[atck], epsilon[ind-1]))
            print("Predicted adversarial example: ", predicted[ind], "   real value: ", real_value)

        plot_img.append(keras.preprocessing.image.array_to_img(img_255[ind]))
        plot_img.append(gradCam_img[ind])
    print("     ------------------")
    return plot_img, predicted

def save_and_plot_results(num, list_of_images, predicted, epsilon, attack):
    num_rows=1+len(epsilon)
    fig, axs = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, 15), subplot_kw={'xticks': [], 'yticks': []}, layout='compressed')
    ind = 0;
    ind_pred = 0;
    real_value = gradCamInterface.decode_predictions(y_test[num], num_classes, classes)
    for ax in axs.flat:
        ax.imshow(list_of_images[ind])
        #Ponemos titulos y nombre de los ejes
        if ind == 0:
            ax.set_ylabel('Original')

        if ind % 2 == 0: #Los pares tendran el valor predecido
            predText = 'Predicted: %s'% (predicted[ind_pred])
            ax.set_title(predText)
            if ind > 1:
                ax.set_ylabel('Adversarial, $\epsilon$=%s'% (epsilon[ind_pred-1]))
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

# ------------------------ Código principal ---------------------------------
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


# Para distintos valores de epsilon
epsilon = [0.01, 0.05]#, 0.05, 0.1, 0.15]
x_test_adv = []
attackName = ['FastGradientMethod', 'BasicIterativeMethod', 'ProjectedGradientDescent']
loadImages = True
if loadImages == True:
    for atck in range(0, len(attackName)):
        individual_atck = []
        for i in range(0, len(epsilon)):
            # Generate adversarial test examples
            attack = getAttackMethod(attackName[atck], classifier, epsilon[i])
            x_test_adv.append(attack.generate(x=X_test))
            individual_atck.append(attack.generate(x=X_test))

            # Evaluate the ART classifier on adversarial test examples
            predictions = classifier.predict(x_test_adv[i+atck*len(epsilon)])
            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
            print("AttackMethod: %s with epsilon = %s" % (attackName[atck], epsilon[i]))
            print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
            # Print time
            t3 = time.time()
            print("Time: %0.2fs" % (t3 - t2))
            t2 = t3
        filename = "Adv_Images_AttackMethod_" + attackName[atck] + "_Epsilon_%s" % (epsilon) + ".pkl"
        guardar_datos(individual_atck, filename)
    guardar_datos(x_test_adv, "atcks_%s" % (attackName) +"_Epsilon_%s" % (epsilon) + ".pkl")
else:
    for atck in range(0, len(attackName)):
        filename = "Adv_Images_AttackMethod_" + attackName[atck] +"_Epsilon_%s" % (epsilon) + ".pkl"
        if atck == 0:
            x_test_adv = cargar_datos(filename)
        else:
            x_test_adv.append(cargar_datos(filename))
#GRAD CAM

# Remove last layer's softmax
model.layers[-1].activation = None
last_conv_layer_name = 'conv2d_5' #model.layers[-1].name

A = []
i = 0
n = 20
while(i < n):
    A.append(random.randint(1, 700))
    i+=1
for atck in range(0, len(attackName)):
    for index in range(0, n):
        list_of_images, predicted = executeGradCam(A[index], epsilon, atck)
        save_and_plot_results(A[index], list_of_images, predicted, epsilon, attackName[atck])


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