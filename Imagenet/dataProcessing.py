import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
#NO FUNSIONA: from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from cv2 import PSNR

import gradCamInterface
import auxiliarFunctions as aux
import os
def writeDataImageInCSV(data, img):
    data.append(img.name)
    if img.attackName == "":
        if img.predictionId == img.id:
            data.append("Original")
            data.append("-")
        else:
            data.append("Adv. Natural")
            data.append("-")
    else:
        data.append(img.attackName)
        data.append(img.epsilon)
    return data

# ------------------------ Constantes ---------------------------------------
DATA_PATH = "C:/Users/User/TFG-repository/Imagenet/variables/"
DATA_ID = "execution_02"
NUM_ATCKS = 1 #Numero de ataques distintos que se usaron cuando se guardaron las imagenes
NUM_EPS = 2  #Numero de epsilon distintos que se usaron cuando se guardaron las imagenes

img_orig, img_adv = aux.loadImagesByID(DATA_PATH, DATA_ID)
NUM_IMG = len(img_orig)

# ------------------------ Operaciones --------------------------------------
execute_gray_gradcam=False
calculate_metrics=True
if calculate_metrics == True:
    metricsName = ["Nombre Imagen", "Ataque", "Epsilon", "Media", "Media/255*100 (%)", "Varianza", "Desviación Típica", "Norma Mascara", "Norma Imagen", "MSE", "PSNR", "SSIM"]
    aux.createCsvFile(DATA_ID+"_metrics.csv", metricsName)
#Se ordenan las imagenes como [img_orig, img_adv1eps1, img_adv1eps2, img_adv2eps1, img_adv2eps2, ...]
for num in range(0, NUM_IMG):
    sorted_list = []
    list_img_to_plot = []
    sorted_list.append(img_orig[num])
    for atck_iter in range(0, NUM_ATCKS):
        for eps_iter in range(0, NUM_EPS):
            pos = num+NUM_IMG*atck_iter*NUM_EPS+NUM_IMG*eps_iter
            sorted_list.append(img_adv[pos])#Guarda todas las imagenes seguidas


    if execute_gray_gradcam == True:
        if aux.isValidExample_sortedList(sorted_list) :# Si la red no ha acertado en la predicción de la imagen original, no se guarda la imagen
            for ind in range(0, len(sorted_list)):
                list_img_to_plot.append(keras.preprocessing.image.array_to_img(sorted_list[ind].data))
                list_img_to_plot.append(gradCamInterface.display_gray_gradcam(sorted_list[ind].data, sorted_list[ind].heatmap))
            aux.saveResults(list_img_to_plot, sorted_list, DATA_ID, "superimposed")

    if calculate_metrics == True:
        for ind in range(0, len(sorted_list)) :
            metricsValue = []
            metricsValue = writeDataImageInCSV(metricsValue, sorted_list[ind])
            gray_heatmap = gradCamInterface.display_gray_gradcam(sorted_list[ind].data, sorted_list[ind].heatmap, superimposed=False)
            gray_heatmap_orig = gradCamInterface.display_gray_gradcam(sorted_list[0].data, sorted_list[0].heatmap, superimposed=False)
            list_img_to_plot.append(gray_heatmap)
            metricsValue.append(round(gray_heatmap.mean(), 2))
            metricsValue.append(round(gray_heatmap.mean()/255*100, 2))
            metricsValue.append(round(gray_heatmap.var(), 2))
            metricsValue.append(round(gray_heatmap.std(), 2))
            if metricsValue[1] != "Original": #Si no es la imagen original
                # La norma es la distancia euclidea
                metricsValue.append(round(np.linalg.norm(gray_heatmap - gray_heatmap_orig), 2))
                metricsValue.append(round(np.linalg.norm(sorted_list[ind].data - sorted_list[0].data),2))
                metricsValue.append(round(np.square(np.subtract(gray_heatmap, gray_heatmap_orig)).mean(), 2))#MSE con mean_squared_error da error Found array with dim 3. None expected <= 2.
                metricsValue.append(round(PSNR(gray_heatmap, gray_heatmap_orig), 2))
                metricsValue.append(round(ssim(gray_heatmap, gray_heatmap_orig, data_range=255, channel_axis=-1), 2))#SSIM. The higher the value, the more "similar" the two images are.
                #DUDA: Le quito el tercer canal? no se supone que es gris?
            else:
                metricsValue.append("-")
                metricsValue.append("-")
                metricsValue.append("-")
                metricsValue.append("-")
                metricsValue.append("-")

            aux.addRowToCsvFile(DATA_ID+"_metrics.csv", metricsName, metricsValue)

