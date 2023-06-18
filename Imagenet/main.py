from __future__ import print_function

import gradCamInterface
import auxiliarFunctions as aux

import os
import errno
import tensorflow as tf
from tensorflow import keras
from keras.applications.efficientnet import preprocess_input, decode_predictions
#Mas ataques: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#fast-gradient-method-fgm
from art.estimators.classification import TensorFlowV2Classifier

# ------------------------ Funciones auxiliares ---------------------------------
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def executeGradCam(num, classifier, epsilon, n_iter):
    # https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network
    # Prepare image
    list_img = []
    list_img.append(img_test[num])
    for j in range(0, len(epsilon)):
        list_img.append(img_adv[num+NUM_IMG*n_iter*len(epsilon)+NUM_IMG*j])#Guarda todas las imagenes seguidas
    img_array = []
    predicted = []
    heatmap = []
    gradCam_img = []
    plot_img = []
    # Para la lista de imagenes que tendra la forma: [imagOriginal, adv_eps1, adv_eps2...]
    for ind in range(0, len(list_img)): #ind == 0 es la imagen sin modificar
        img_array.append(gradCamInterface.get_img_array(list_img[ind].data))
        preds = classifier.predict(img_array[ind])
        p = decode_predictions(preds, top=1) #El resultado es del tipo [[('n03814906', 'necklace', 9.88)]]
        predicted.append(p[0][0])
        list_img[ind].addPrediction(p[0][0][0])

        # Generate class activation heatmap
        heatmap.append(gradCamInterface.make_gradcam_heatmap(img_array[ind], model, last_conv_layer_name))
        list_img[ind].addHeatmap(heatmap[ind])

        # Display heatmap
        #Ya esta entre 0-255 img_255.append(list_img[ind] * 255)
        gradCam_img.append(gradCamInterface.display_gradcam(list_img[ind].data, heatmap[ind]))

        if ind == 0:
            print("Real value: ", list_img[ind].idName)
            print("Predicted benign example: ", list_img[ind].predictionName)
        else:
            print("AttackMethod: %s with epsilon = %s" % (ATTACK_NAME[atck], epsilon[ind-1]))
            print("Predicted adversarial example: ", list_img[ind].predictionName)

        plot_img.append(keras.preprocessing.image.array_to_img(list_img[ind].data))
        plot_img.append(gradCam_img[ind])

    #Se actualiza los valores de prediccion y heatmap de las imagenes:
    img_test[num].addPrediction(list_img[0].predictionId)
    img_test[num].addHeatmap(list_img[0].heatmap)
    for j in range(0, len(epsilon)):
        index = num + NUM_IMG*n_iter*len(epsilon) + NUM_IMG*j
        img_adv[index].addPrediction(list_img[j+1].predictionId)
        img_adv[index].addHeatmap(list_img[j+1].heatmap)
    print("     ------------------")
    return plot_img, list_img

# ------------------------ Constantes ---------------------------------------
NUM_CLASSES = 1000
IMG_SIZE = (224, 224)
IMG_SHAPE = (224, 224, 3)
LR = 0.01 #Learning Rate usado en el optimizador
NUM_IMG = 800 #Cantidad de imagenes de test
TOTAL_IMG = 810
IMG_PATH = "C:/Users/User/TFG-repository/webcam_gradcam/ImageNetWebcam/water_bottle_efficientNetB0/frames_raw/"
#IMG_PATH = "C:/Users/User/TFG-repository/Imagenet/val_classes/"
EXECUTION_ID = "WebcamData_01" #Se usará para no sustituir variables de distintas ejecuciones
realID='n04557648'

EPSILON = [20000, 30000]
ATTACK_NAME = ['FastGradientMethod']

# ------------------------ Código principal ---------------------------------
# Load model: CNN -> EfficientNetB0
model = tf.keras.applications.EfficientNetB0(weights="imagenet", include_top=True, classes=NUM_CLASSES, input_shape=IMG_SHAPE)
model.trainable = False
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
classifier = TensorFlowV2Classifier(model=model, clip_values=(0, 1), nb_classes=NUM_CLASSES, input_shape=IMG_SHAPE, loss_object=loss_object, train_step=train_step)

#Load Images
randomVector = aux.generateRandomVector(NUM_IMG, TOTAL_IMG)
x_test, img_test = aux.loadImages(IMG_PATH, randomVector, unclassified_images=True, realID=realID)
#Si createImages = True: cargará las imagenes originales desde la carpeta y generará las adversarias de cero

#Generate Adversarials
img_adv = aux.generateAdversarialImages(img_test, x_test, ATTACK_NAME, EPSILON, classifier)

#GRAD CAM
# Remove last layer's softmax
model.layers[-1].activation = None
print(model.summary())
last_conv_layer_name = "top_activation"
for atck in range(0, len(ATTACK_NAME)):
    for num in range(0, NUM_IMG):
        list_of_images, list_img_data = executeGradCam(num, classifier, EPSILON, atck)
        isSuccesfulExample = aux.isValidExample(num, img_test, img_adv, atck, EPSILON, filter=False)
        if isSuccesfulExample:
            aux.saveResults(list_of_images, list_img_data, EXECUTION_ID)
            aux.plotDifference(num, img_test, img_adv, atck, EPSILON, EXECUTION_ID)
aux.calculateAccuracy(img_test, img_adv, ATTACK_NAME, EPSILON)

# Save variables
try :
    os.mkdir('variables')
except OSError as e :
    if e.errno != errno.EEXIST :
        raise
aux.saveVariable(img_test, "variables/%s_testImages_efficientnetB0_random%simages.pkl" % (EXECUTION_ID, NUM_IMG))
aux.saveVariable(img_adv, "variables/%s_Adversarials_images_atcks_%s" % (ATTACK_NAME) + "_Epsilon_%s" % (EXECUTION_ID, EPSILON) + ".pkl")
#https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network