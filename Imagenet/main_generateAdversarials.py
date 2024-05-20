from __future__ import print_function

import gradCamInterface
import auxiliarFunctions as aux

import os
import errno
import tensorflow as tf
from tensorflow import keras
#Mas ataques: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#fast-gradient-method-fgm
from art.estimators.classification import TensorFlowV2Classifier

# ------------------------ Funciones auxiliares ---------------------------------
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# DEPRECATED: executeGradCam(num, classifier, epsilon, n_iter):
# https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network

def executeGradCam(orig, adv) :
    # Prepare image
    list_img = []  # Orig, adversaria
    list_img.append(orig)
    list_img.append(adv)
    plot_img = []

    last_conv_layer_name = aux.getLastConvLayerName(NetworkModelName)
    # Generate class activation heatmap
    for ind in range(0, len(list_img)):
        img_array = gradCamInterface.get_img_array(list_img[ind].data)
        heatmap = gradCamInterface.make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        list_img[ind].addHeatmap(heatmap)

        # Display heatmap. Ya esta entre 0-255
        gradCam_img = gradCamInterface.display_gradcam(list_img[ind].data, heatmap)

        plot_img.append(keras.preprocessing.image.array_to_img(list_img[ind].data))
        plot_img.append(gradCam_img)

    orig.addHeatmap(list_img[0].heatmap)
    adv.addHeatmap(list_img[1].heatmap)
    aux.printResultsPerImage(orig, adv)
    print("     ------------------")
    return plot_img, list_img

# ------------------------ Constantes ---------------------------------------
NUM_CLASSES = 1000 #imagenet=1000

IMG_SIZE = (299, 299)
IMG_SHAPE = (299, 299, 3)
LR = 0.01 #Learning Rate usado en el optimizador
EPSILON = 125
ATTACK_NAME = 'ProjectedGradientDescent'
NetworkModelName = 'VGG16'
IMG_PATH = "C:/Users/User/TFG-repository/webcam_gradcam/ImageNetWebcam/luz_validos/waterBottle_%s_5000luz/selectedOrigImages/selected/" % (NetworkModelName)
realID='n04557648' #botella de agua

if NetworkModelName == "vgg16" or NetworkModelName == "VGG16" or NetworkModelName == "EfficientNetB0" or NetworkModelName == "mobileNet" or NetworkModelName == "MobileNet" :
    IMG_SIZE = (224, 224)
    IMG_SHAPE = (224, 224, 3)
# ------------------------ Código principal ---------------------------------
# Load model: CNN -> EfficientNetB0
model = aux.getNetworkModel(NetworkModelName)
model.trainable = False
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
classifier = TensorFlowV2Classifier(model=model, clip_values=(0, 1), nb_classes=NUM_CLASSES, input_shape=IMG_SHAPE, loss_object=loss_object, train_step=train_step)

#Load Images
files_names = os.listdir(IMG_PATH)
randomVector = []
for index in range(0, len(files_names)):
    files_names[index] = files_names[index].replace('.png', '')
    randomVector.append(files_names[index].replace('imageFrame_', ''))

x_test, img_test = aux.loadImages(IMG_PATH, randomVector, size= IMG_SIZE, unclassified_images=True, realID=realID, networkName=NetworkModelName)# Quitar unclassified_images y realID para imagenet
#Si createImages = True: cargará las imagenes originales desde la carpeta y generará las adversarias de cero
#Si unclassified_images = True: cargará las imagenes que no son de imagenet y por tanto no estan dentro de una carpeta con el valor de su ID

#Generate Adversarials
variablePath = "C:/Users/User/TFG-repository/Imagenet/recalculateVariables_%s/" % (NetworkModelName)
aux.generateAllAdversarialImagesAtOnce(img_test, x_test, ATTACK_NAME, EPSILON, classifier, variablePath, isImagenet=False)


