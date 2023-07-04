import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
import gradCamInterface
import auxiliarFunctions as aux
import os
# ------------------------ Constantes ---------------------------------------
DATA_PATH = "C:/Users/User/TFG-repository/Imagenet/variables/"
DATA_ID = "execution_02"
NUM_ATCKS = 1 #Numero de ataques distintos que se usaron cuando se guardaron las imagenes
NUM_EPS = 2  #Numero de epsilon distintos que se usaron cuando se guardaron las imagenes

img_orig, img_adv = aux.loadImagesByID(DATA_PATH, DATA_ID)
NUM_IMG = len(img_orig)

#Se ordenan las imagenes como [img_orig, img_adv1eps1, img_adv1eps2, img_adv2eps1, img_adv2eps2, ...]
for num in range(0, NUM_IMG):
    sorted_list = []
    list_img_to_plot = []
    sorted_list.append(img_orig[num])
    for atck_iter in range(0, NUM_ATCKS):
        for eps_iter in range(0, NUM_EPS):
            pos = num+NUM_IMG*atck_iter*NUM_EPS+NUM_IMG*eps_iter
            sorted_list.append(img_adv[pos])#Guarda todas las imagenes seguidas

    # Si la red no ha acertado en la predicción de la imagen original, no se guarda la imagen
    if aux.isValidExample(sorted_list):
        for ind in range(0, len(sorted_list)):
            list_img_to_plot.append(keras.preprocessing.image.array_to_img(sorted_list[ind].data))
            list_img_to_plot.append(gradCamInterface.display_superimposed_gradcam(sorted_list[ind].data, sorted_list[ind].heatmap))
        aux.saveResults(list_img_to_plot, sorted_list, DATA_ID, "superimposed")