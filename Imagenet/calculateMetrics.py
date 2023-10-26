import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.metrics import mean_squared_error #con 3chnn no funciona
from skimage.metrics import structural_similarity as ssim
import cv2
import xlwt #Para escribir en excel

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

def searchCloserOriginalImage(img_test, name_list, num):
#Se parte de que el objeto imagen tiene un parametro que dice el nombre de su imagen original mas cercana
    pos = name_list.index(img_test[num].closerOriginalImageName)
    if img_test[pos].name == img_test[num].closerOriginalImageName:
        return img_test[pos]
    else:
        import sys
        print('No se ha encontrado una imagen original cercana a la adversaria natural, revisa los datos')
        sys.exit(1)

# ------------------------ Constantes ---------------------------------------
DATA_PATH = "C:/Users/User/TFG-repository/Imagenet/variablesIndividuales_WebcamData_luzTest_InceptionResNetV2/"
DATA_ID = ""
NUM_ATCKS = 1 #Numero de ataques distintos que se usaron cuando se guardaron las imagenes

sorted_data, name_list = aux.loadImageOneByOne(DATA_PATH)
NUM_IMG = len(sorted_data)

# ------------------------ Operaciones --------------------------------------
calculate_metrics = True #tarda 'poco'
execute_Histogram = False #tarda mucho

if calculate_metrics == True:
    metricsName = ["Nombre Imagen", "Ataque", "Epsilon", "Media", "Media normalizada", "Dif de medias", "Valor máximo",
                   "Dif de máximo", "Posición del máximo", "Distancia entre máximos", "Valor mínimo", "Dif de mínimo",
                   "Posición del mínimo", "Distancia entre mínimos", "Varianza", "Desviación Típica", "Norma Mascara",
                   "Norma Imagen", "MSE", "PSNR", "SSIM"]
    aux.createCsvFile(DATA_ID+"_metrics.csv", metricsName)

for num in range(0, NUM_IMG):
    list_img_to_plot = []
    if calculate_metrics == True:
        closer_orig_img = searchCloserOriginalImage(sorted_data, name_list, num)

        metricsValue = []
        metricsValue = writeDataImageInCSV(metricsValue, sorted_data[num])

        gray_heatmap = gradCamInterface.display_gray_gradcam(sorted_data[num].data, sorted_data[num].heatmap, superimposed=False)
        gray_heatmap = cv2.cvtColor(gray_heatmap, cv2.COLOR_RGB2GRAY)  # plt.imshow(gray, cmap='gray')
        gray_heatmap_orig = gradCamInterface.display_gray_gradcam(sorted_data[num].data, closer_orig_img.heatmap, superimposed=False)
        gray_heatmap_orig = cv2.cvtColor(gray_heatmap_orig, cv2.COLOR_RGB2GRAY)
        list_img_to_plot.append(gray_heatmap)
        metricsValue.append(round(gray_heatmap.mean(), 2))
        metricsValue.append(round(gray_heatmap.mean()/255, 2))
        metricsValue.append(round(gray_heatmap.var(), 2))
        metricsValue.append(round(gray_heatmap.std(), 2))
        if metricsValue[1] != "Original" and metricsValue[1] != "Adv. Natural": #Si no es la imagen original
            # La norma es la distancia euclidea
            metricsValue.append(round(np.linalg.norm(gray_heatmap - gray_heatmap_orig), 2))
            metricsValue.append(round(np.linalg.norm(sorted_data[num].data - closer_orig_img.data),2))
            metricsValue.append(round(mean_squared_error(gray_heatmap, gray_heatmap_orig),2))#3ch; round(np.square(np.subtract(gray_heatmap, gray_heatmap_orig)).mean(), 2))#MSE con mean_squared_error da error Found array with dim 3. None expected <= 2.
            metricsValue.append(round(cv2.PSNR(gray_heatmap, gray_heatmap_orig), 2)) #Peak Signal-to-Noise Ratio (PSNR)
            metricsValue.append(round(ssim(gray_heatmap, gray_heatmap_orig, data_range=255, channel_axis=-1), 2))#SSIM. The higher the value, the more "similar" the two images are.
            #DUDA: Le quito el tercer canal? no se supone que es gris?
        else:
            metricsValue.append("-")
            metricsValue.append("-")
            metricsValue.append("-")
            metricsValue.append("-")
            metricsValue.append("-")

        aux.addRowToCsvFile(DATA_ID+"_metrics.csv", metricsName, metricsValue)

    if execute_Histogram == True:
        aux.saveHistogram(sorted_data[num], DATA_ID)