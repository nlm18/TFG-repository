import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.metrics import mean_squared_error #con 3chnn no funciona
from skimage.metrics import structural_similarity as ssim
from statistics import median
import cv2
from scipy.stats import shapiro
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
#y la lista esta ordenada de la forma image_1_testImage,image_1_adversarial,image_2_testImage
    if img_test[num].closerOriginalImageName != '':
        pos = name_list.index(img_test[num].closerOriginalImageName)
        if img_test[pos].name == img_test[num].closerOriginalImageName and img_test[pos].attackName == '':
            return img_test[pos]
        else:
            import sys
            print('No se ha encontrado una imagen original cercana a la adversaria natural, revisa los datos')
            sys.exit(1)
    else:
        pos = name_list.index(img_test[num].name)
        return img_test[pos]


# ------------------------ Constantes ---------------------------------------
DATA_ID = "EfficientNetB0"
DATA_PATH = "C:/Users/User/TFG-repository/Imagenet/variablesIndividuales_Test_%s/" % (DATA_ID)
NUM_ATCKS = 1 #Numero de ataques distintos que se usaron cuando se guardaron las imagenes

sorted_data, name_list = aux.loadImagesSorted(DATA_PATH)
NUM_IMG = len(sorted_data)

# ------------------------ Operaciones --------------------------------------
calculate_metrics = False #tarda 'poco'
execute_Histogram = False #tarda mucho
execute_BoxPlot = True

if calculate_metrics == True:
    metricsName = ["Nombre Imagen", "Ataque", "Epsilon", "Media", "Media normalizada", "Varianza", "Desviación Típica",
                   "Mediana", "Dif de medias", "Norma Mascara", "Norma Imagen", "MSE", "PSNR", "SSIM"]
    aux.createCsvFile(DATA_ID+"_metrics.csv", metricsName)

for num in range(0, NUM_IMG):
    gray_heatmap = gradCamInterface.display_gray_gradcam(sorted_data[num].data, sorted_data[num].heatmap,
                                                         superimposed=False)
    gray_heatmap = cv2.cvtColor(gray_heatmap, cv2.COLOR_RGB2GRAY)  # plt.imshow(gray, cmap='gray')
    heatmap_array = gray_heatmap.flatten()
    """result = shapiro(heatmap_array)
    if result[1] < 0.05:
        print('La muestra no sigue una distribucion normal')
    else:
        print('No se tienen suficientes evidencias para asegurar que es una distribución normal')"""
    if calculate_metrics == True:
        metricsValue = []
        metricsValue = writeDataImageInCSV(metricsValue, sorted_data[num]) #Se escribe "Nombre Imagen", "Ataque", "Epsilon"
        if metricsValue[1] != "Original" :
            closer_orig_img = searchCloserOriginalImage(sorted_data, name_list, num)
            heatmap_ref = gradCamInterface.display_gray_gradcam(closer_orig_img.data, closer_orig_img.heatmap,
                                                                superimposed=False)
            heatmap_ref = cv2.cvtColor(heatmap_ref, cv2.COLOR_RGB2GRAY)

        metricsValue.append(round(gray_heatmap.mean(), 2)) #"Media"
        metricsValue.append(round(gray_heatmap.mean()/255, 2)) #"Media normalizada"
        metricsValue.append(round(gray_heatmap.var(), 2)) #"Varianza"
        metricsValue.append(round(gray_heatmap.std(), 2)) #"Desviación Típica"
        metricsValue.append(median(heatmap_array)) #"Mediana"
        #No tiene sentido calcular lo siguiente: "Valor máximo" #"Posición del máximo" #"Valor mínimo"
        #"Posición del mínimo" #"Dif de máximo" #"Distancia entre máximos", #"Dif de mínimo" #"Distancia entre mínimos"
        #Pues el valor maximo es 255 el minimo 0 y hay varios...
        if metricsValue[1] != "Original": #Si no es la imagen original
            # La norma es la distancia euclidea
            metricsValue.append(round(gray_heatmap.mean(), 2)-round(heatmap_ref.mean(), 2)) #"Dif de medias"
            metricsValue.append(round(np.linalg.norm(gray_heatmap - heatmap_ref), 2)) #"Norma Mascara"
            metricsValue.append(round(np.linalg.norm(sorted_data[num].data - closer_orig_img.data),2)) #"Norma Imagen"
            metricsValue.append(round(mean_squared_error(gray_heatmap, heatmap_ref),2)) #Mean Squared Error (MSE)
            metricsValue.append(round(cv2.PSNR(gray_heatmap, heatmap_ref), 2)) #Peak Signal-to-Noise Ratio (PSNR)
            metricsValue.append(round(ssim(gray_heatmap, heatmap_ref, data_range=255, channel_axis=-1), 2))#SSIM. The higher the value, the more "similar" the two images are.
        else:
            param=len(metricsName)-len(metricsValue)
            for index in range(0, param):
                metricsValue.append("-")

        aux.addRowToCsvFile(DATA_ID+"_metrics.csv", metricsName, metricsValue)

    heatmap_array_sinMenorQue25 = [x for x in heatmap_array if x > 25]
    if execute_Histogram == True:
        aux.saveHistogram(heatmap_array, sorted_data[num], DATA_ID)#Parece distribucion geometrica ?
    if execute_BoxPlot == True:
        aux.saveBoxPlot(heatmap_array_sinMenorQue25, sorted_data[num], DATA_ID)
