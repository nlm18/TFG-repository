from __future__ import print_function

import art.attacks.evasion

import gradCamInterface
import os
import errno
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.applications.resnet import preprocess_input, decode_predictions
    #Xception, preprocess_input, decode_predictions
import random
from keras.utils import np_utils
import time

#Mas ataques: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#fast-gradient-method-fgm
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent
from art.estimators.classification import TensorFlowV2Classifier

# ------------------------ Funciones auxiliares ---------------------------------
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
        return BasicIterativeMethod(estimator=classifier, eps=epsilon, eps_step=0.5, max_iter=15)
    elif name == 'ProjectedGradientDescent':
        return ProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=0.5, max_iter=15)

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
        predicted.append(decode_predictions(preds[ind]))
        real_value = decode_predictions(preds[ind])

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
# Load data   t1 = time.time()
num_classes = 1000
input_images_path = "C:/Users/User/TFG-repository/EjemploFuncional_imagenet/imagenes/"
files_names = os.listdir(input_images_path)
img_size = (224, 224)
A = []
i = 0
n = 25
while(i < n):
    A.append(random.randint(0, 47))#4434
    i+=1
# Load model
shape = (224, 224, 3)
model = tf.keras.applications.resnet.ResNet50(weights="imagenet", include_top=False, classes=num_classes, input_shape=shape)
model.trainable = False
lr=0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
classifier = TensorFlowV2Classifier(model=model, clip_values=(0, 1), nb_classes=num_classes, input_shape=shape, loss_object=loss_object, train_step=train_step)


X_test = np.ndarray(shape=(50, 224, 224, 3), dtype='float32')
img_path =[]
for index in range(0, n):
    # Preprocess data
    img_path.append(input_images_path+files_names[A[index]])
    X_test[index] = preprocess_input(gradCamInterface.get_img_array_path(img_path[index], img_size))


# Para distintos valores de epsilon
epsilon = [0.01]#[0.01, 0.05, 0.1, 0.15]
x_test_adv = []
attackName = ['FastGradientMethod']# ['FastGradientMethod', 'BasicIterativeMethod', 'ProjectedGradientDescent']
for atck in range(0, len(attackName)):
    for i in range(0, len(epsilon)):
        # Generate adversarial test examples
        attack = getAttackMethod(attackName[atck], classifier, epsilon[i])
        x_test_adv.append(attack.generate(x=X_test))

#GRAD CAM
# Remove last layer's softmax
model.layers[-1].activation = None
last_conv_layer_name = "block14_sepconv2_act"

for atck in range(0, len(attackName)):
    for index in range(0, n):
        list_of_images, predicted = executeGradCam(A[index], epsilon, atck)
        save_and_plot_results(A[index], list_of_images, predicted, epsilon, attackName[atck])

#https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network
