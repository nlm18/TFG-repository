from __future__ import print_function

import art.attacks.evasion

import gradCamInterface
import os
import errno
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.applications.efficientnet import preprocess_input, decode_predictions
    #Xception, preprocess_input, decode_predictions
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
        return FastGradientMethod(estimator=classifier, eps=epsilon, batch_size=4)
    elif name == 'BasicIterativeMethod':
        return BasicIterativeMethod(estimator=classifier, eps=epsilon, max_iter=15, batch_size=10)
    elif name == 'ProjectedGradientDescent':
        return ProjectedGradientDescent(estimator=classifier, eps=epsilon, max_iter=15, batch_size=10)

def executeGradCam(num, classifier, epsilon, n_iter):
    # https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network
    # Prepare image
    img_orig = X_test[num]
    list_img = []
    list_img.append(img_orig)
    for j in range(0, len(epsilon)):
        list_img.append(x_test_adv[j+n_iter*len(epsilon)][num])
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

        # Generate class activation heatmap
        heatmap.append(gradCamInterface.make_gradcam_heatmap(img_array[ind], model, last_conv_layer_name))
        #plt.matshow(heatmap)  # tiene que estar el eagle enable, eagle error: https://github.com/Trusted-AI/adversarial-robustness-toolbox/issues/330

        # Display heatmap
        img_255.append(list_img[ind] * 255)
        gradCam_img.append(gradCamInterface.display_gradcam(img_255[ind], heatmap[ind]))

        if ind == 0:
            predicted = decode_predictions(preds[ind], top=1)
            print("Predicted benign example: ", predicted[ind][0][1])
        else:
            predicted.append(decode_predictions(preds[ind], top=1))
            print("AttackMethod: %s with epsilon = %s" % (attackName[atck], epsilon[ind-1]))
            print("Predicted adversarial example: ", predicted[ind][0][0][1])

        plot_img.append(keras.preprocessing.image.array_to_img(img_255[ind]))
        plot_img.append(gradCam_img[ind])
    print("     ------------------")
    return plot_img, predicted

def save_and_plot_results(num, list_of_images, predicted, epsilon, attack):
    num_rows=1+len(epsilon)
    fig, axs = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, 15), subplot_kw={'xticks': [], 'yticks': []}, layout='compressed')
    ind = 0;
    ind_pred = 0;
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
    suptitle = 'Attack method used: %s' % (attack)
    fig.suptitle(suptitle)
    try :
        os.mkdir('gradCam_examples_attack_method-%s' % (attack))
    except OSError as e :
        if e.errno != errno.EEXIST :
            raise
    File_name = 'gradCam_examples_attack_method-%s/gradCam_example_image-%s_attack_method-%s.jpg' % (attack, num, attack )
    fig.savefig(File_name)

# ------------------------ Código principal ---------------------------------
#%% Data Preparation
# Load data   t1 = time.time()
num_classes = 1000
img_size = (300, 300)
A = []
i = 0
n = 25
total_img=47
while(i < n):
    A.append(random.randint(0, total_img))#4434
    i+=1
# Load model
img_shape = (300, 300, 3)
model = tf.keras.applications.EfficientNetB3(weights="imagenet", include_top=True, classes=num_classes, input_shape=img_shape)
model.trainable = False
lr=0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
classifier = TensorFlowV2Classifier(model=model, clip_values=(0, 1), nb_classes=num_classes, input_shape=img_shape, loss_object=loss_object, train_step=train_step)

#Load Images
X_test = np.ndarray(shape=(total_img+1, 300, 300, 3), dtype='float32')
loadImages = False
if loadImages== True:
    input_images_path = "C:/Users/User/TFG-repository/Intento_imagenet/imagenes/"
    files_names = os.listdir(input_images_path)
    img_path =[]
    for index in range(0, total_img):
        # Preprocess data
        img_path.append(input_images_path+files_names[index])
        X_test[index] = preprocess_input(gradCamInterface.get_img_array_path(img_path[index], img_size))
    guardar_datos(X_test, "testImages.pkl")
else:
    X_test = cargar_datos("testImages.pkl")


# Para distintos valores de epsilon
loadImages = True
epsilon = [1]#[0.01, 0.05, 0.1, 0.15]
x_test_adv = []
attackName = ['FastGradientMethod']# ['FastGradientMethod', 'BasicIterativeMethod', 'ProjectedGradientDescent']
if loadImages == True:
    for atck in range(0, len(attackName)):
        for i in range(0, len(epsilon)):
            # Generate adversarial test examples
            attack = getAttackMethod(attackName[atck], classifier, epsilon[i])
            x_test_adv.append(attack.generate(x=X_test))
        filename = "Adv_Images_AttackMethod_" + attackName[atck] +"_Epsilon_%s" % (epsilon) + ".pkl"
        guardar_datos(x_test_adv, filename)
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
last_conv_layer_name = "top_activation"

for atck in range(0, len(attackName)):
    for index in range(0, n):
        list_of_images, predicted = executeGradCam(A[index], classifier, epsilon, atck)
        save_and_plot_results(A[index], list_of_images, predicted, epsilon, attackName[atck])

#https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network
