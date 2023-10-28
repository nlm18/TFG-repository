from __future__ import print_function

import gradCamInterface
import auxiliarFunctions as aux
from selectOrigImages import searchCloserImageByID

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
NUM_IMG = 500 #Cantidad de imagenes de test
TOTAL_IMG = 500 #Cantidad de imagenes de las que se disponen, imagenet=50000
ATTACK_NAME = 'FastGradientMethod'
NetworkModelName = 'EfficientNetB0'
#cambiar parametros de entrada de loadImages segun si son de imagenet o no
IMG_PATH = "C:/Users/User/TFG-repository/webcam_gradcam/ImageNetWebcam/luz_validos/waterBottle_%s_5000luz/" % (NetworkModelName)
folder = 'frames_naturalAdv/'#'frames_naturalAdv/' 'selectedOrigImages/'
EXECUTION_ID = "Test_%s" % (NetworkModelName)#Se usará para no sustituir variables de distintas ejecuciones
realID='n04557648'

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

aux.createDirs(EXECUTION_ID, onebyone=True)
num_AdvNaturales = 0
startIndex = 0
#Load Images
for num_img in range(startIndex, NUM_IMG) :
    x_test, img_test = aux.loadImage(IMG_PATH+folder, num_img, size= IMG_SIZE, unclassified_images=True, realID=realID, networkName=NetworkModelName)# Quitar unclassified_images y realID para imagenet
#Si createImages = True: cargará las imagenes originales desde la carpeta y generará las adversarias de cero
#Si unclassified_images = True: cargará las imagenes que no son de imagenet y por tanto no estan dentro de una carpeta con el valor de su ID
    print("Image name: %s" %(img_test[0].name))
    if folder == 'frames_naturalAdv/':
        closerImageName = searchCloserImageByID(img_test[0].name, IMG_PATH+'selectedOrigImages/')
        img_test[0].addCloserOriginalImageName(closerImageName)
    #Generate Adversarial
    img_adv=[]
    img_adv.append(aux.generateAnAdversarialImage(img_test[0], x_test[0], ATTACK_NAME, classifier, isImagenet=False))
#Hasta aqui tenemos un objeto imagen para la original y otro para la adversaria, en ambas se ha predicho ya la clase

#GRAD CAM
    if num_img == startIndex: #Solo se hace una vez
        # Remove last layer's softmax
        model.layers[-1].activation = None #efficientnetb0
        #print(model.summary())

    img_figure, list_img_data = executeGradCam(img_test[0], img_adv[0])
    aux.saveResults(img_figure, list_img_data, EXECUTION_ID)
    aux.plotDifferenceBetweenImages(img_test[0], img_adv[0], EXECUTION_ID)

    img_id = img_test[0].name
    img_id = img_id.replace('.png', '')
    # Save variables
    if img_test[0].advNatural == False:
        aux.saveVariable(img_test[0], "variablesIndividuales_%s/ArtificialAdversarial/%s_testImage.pkl" % (EXECUTION_ID, img_id))
        aux.saveVariable(img_adv[0], "variablesIndividuales_%s/ArtificialAdversarial/%s_adversarialImage_atck_%s" % (EXECUTION_ID, img_id, ATTACK_NAME) + ".pkl")
    else:
        aux.saveVariable(img_test[0], "variablesIndividuales_%s/NaturalAdversarial/%s_testImage.pkl" % (EXECUTION_ID, img_id))
        num_AdvNaturales += 1
    #Borramos variables
    img_test.pop()
    img_adv.pop()
    img_figure.pop()
    list_img_data.pop()
    x_test = []
print("- Percentage of natural adversarial images: {}%".format(num_AdvNaturales/NUM_IMG * 100))