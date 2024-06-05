from __future__ import print_function
import gradCamInterface
import auxiliarFunctions as aux
from selectOrigImages import searchCloserImageByID
import copy
import tensorflow as tf
from tensorflow import keras
from art.estimators.classification import TensorFlowV2Classifier

# -------------------- Funciones auxiliares --------------------
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

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
        auxi = aux.preprocess_input(list_img[0].networkModelName, copy.deepcopy(img_array))#test
        heatmap = gradCamInterface.make_gradcam_heatmap(auxi, model, last_conv_layer_name)
        list_img[ind].addHeatmap(heatmap)

        # Display heatmap. Ya esta entre 0-255
        gradCam_img = gradCamInterface.display_gradcam(list_img[ind].data, heatmap, alpha=0.4)

        plot_img.append(keras.preprocessing.image.array_to_img(list_img[ind].data))
        plot_img.append(gradCam_img)

    orig.addHeatmap(list_img[0].heatmap)
    adv.addHeatmap(list_img[1].heatmap)
    aux.printResultsPerImage(orig, adv)
    print("     ------------------")
    return plot_img, list_img

# --------------------- Constantes ------------------------------
NUM_CLASSES = 1000 #imagenet=1000
IMG_SIZE = (299, 299)
IMG_SHAPE = (299, 299, 3)
LR = 0.01 #Learning Rate usado en el optimizador
NUM_IMG = 500 #Cantidad de imagenes de test
TOTAL_IMG = 500 #Cantidad de imagenes de las que se disponen
ATTACK_NAME = 'ProjectedGradientDescent'
NetworkModelName = 'VGG16'
IMG_PATH = "C:/Users/User/TFG-repository/webcam_gradcam/ImageNetWebcam/waterBottle_%s/" % (NetworkModelName)
folder = 'selectedOrigImages/'#'frames_naturalAdv/' or 'selectedOrigImages/'
EXECUTION_ID = "Test_%s" % (NetworkModelName)#Se usa para no sustituir variables de distintas ejecuciones
realID='n04557648'#'n04557648' botella de agua ------ n03388183 fountain pen

if NetworkModelName == "VGG16" or NetworkModelName == "EfficientNetB0":
    IMG_SIZE = (224, 224)
    IMG_SHAPE = (224, 224, 3)

# -------------------- Codigo principal ------------------------
# Load model: CNN
model = aux.getNetworkModel(NetworkModelName)
model.trainable = False
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
classifier = TensorFlowV2Classifier(model=model, clip_values=(0, 255), nb_classes=NUM_CLASSES, input_shape=IMG_SHAPE, loss_object=loss_object, train_step=train_step)

aux.createDirs(EXECUTION_ID, onebyone=True)
num_AdvNaturales = 0
startIndex = 0
createAdvImages=False
#Load Images
for num_img in range(startIndex, NUM_IMG) :
    #Si falla al cargar la imagen comprueba que hay num_img suficientes en la carpeta
    x_test, img_test = aux.loadImage(IMG_PATH+folder, num_img, size= IMG_SIZE, unclassified_images=True, realID=realID, networkName=NetworkModelName)# Quitar unclassified_images y realID para imagenet
#Si createImages = True: carga las imagenes originales desde la carpeta y genera las adversarias de cero
#Si unclassified_images = True: carga las imagenes que no son de imagenet y por tanto no estan dentro de una carpeta con el valor de su ID
    print("Image name: %s" %(img_test[0].name))
    img_adv = []
    if createAdvImages:
        if folder == 'frames_naturalAdv/':
            closerImageName = searchCloserImageByID(img_test[0].name, IMG_PATH+'selectedOrigImages/')
            img_test[0].addCloserOriginalImageName(closerImageName)
        #Generate Adversarial
        img_adv.append(aux.generateAnAdversarialImage(img_test[0], x_test[0], ATTACK_NAME, classifier, isImagenet=False))
    else:
        variablePath = "C:/Users/User/TFG-repository/Imagenet/recalculateVariables_%s/" % (NetworkModelName)
        orig_img_name = copy.deepcopy(img_test[0].name)
        orig_img_name = orig_img_name.replace('.png','')
        file_name = "%s_adversarialImage_atck_%s.pkl" % (orig_img_name, ATTACK_NAME)
        img_adv.append(aux.loadVariable(variablePath+file_name))
#Hasta aqui tenemos un objeto imagen para la original y otro para la adversaria, en ambas se ha predicho ya la clase

#GRAD CAM
    if num_img == startIndex: #Solo se hace una vez
        # Remove last layer's softmax
        model.layers[-1].activation = None

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