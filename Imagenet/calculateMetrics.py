import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.metrics import mean_squared_error #con 3chnn no funciona
from skimage.metrics import structural_similarity as ssim
from statistics import median
import cv2
import pandas as pd
import plotly.graph_objects as go
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
    data.append(img.predictionName)
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

def meanFreqPerBin(bins, array):
    interval = 255.0/bins
    sorted_list = np.array(sorted(array))
    mean_array = []
    std_array = []
    freq_array = []
    inf = 0
    for i in range(0, bins):
        if interval*(i+1) != 255:
            sup = np.where(sorted_list >interval*(i+1))[0][0]
        else:
            sup = len(sorted_list)-1
        mean_array.append(sorted_list[inf:sup].mean())
        std_array.append(sorted_list[inf:sup].std())
        freq_array.append(len(sorted_list[inf:sup]))
        inf = sup+1
    return mean_array, np.array(freq_array), std_array

def createDataFrameToPlot(freq_orig, freq_nat, freq_art, std_orig, std_nat, std_art, violin=False):
    #https://www.codigopiton.com/como-crear-un-dataframe-con-pandas-y-python/#5-c%C3%B3mo-crear-un-dataframe-a-partir-de-un-diccionario-de-listas
    columns = ['Original','Adv. Natural','Adv. Artificial']

    df = pd.DataFrame(columns=columns)
    dfe = pd.DataFrame(columns=columns)

    # añadimos filas por su nombre de fila
    df['Original'] = list(freq_orig)
    df['Adv. Natural'] = list(freq_nat)
    df['Adv. Artificial'] = list(freq_art)
    # añadimos filas por su nombre de fila
    dfe['Original'] = std_orig
    dfe['Adv. Natural'] = std_nat
    dfe['Adv. Artificial'] = std_art
    if violin != True:
        df.plot(kind='bar', yerr=dfe, ecolor="#FF5733", width=0.8)
        plt.legend()
        plt.title('Histograma comparativo del mapa de activación,\nresumen de las 500 imágenes de cada tipo')
        plt.xlabel('Intervalos de intensidad del mapa de activación')
        plt.ylabel('Frecuencia')
        plt.ylim(0, 20000)
        plt.xticks([])
        plt.subplots_adjust(bottom=0.1, right=0.97)
        plt.savefig("graficas-%s/comparacionFreqHistograma" % (DATA_ID))
        plt.clf()
    if violin:
        fig = go.Figure(go.Violin(y=df, box_visible=True, line_color="#ACCBF3", meanline_visible=True,
                              fillcolor="#ACCBF3", opacity=0.6))
        fig.write_image("j.png")
#https://joserzapata.github.io/courses/python-ciencia-datos/visualizacion/

def combineMeanValueWithFreq(mean, freq):
    for i in range(0, len(freq)):
        freq[i]=round(freq[i])
    result = np.zeros(shape=int(sum(freq)))
    inf = 0
    for i in range(0, len(mean)):
        sup = inf+int(freq[i])
        result[inf:sup] = mean[i]
        inf = sup
    return result
def calculateMaxCentroid(gray_heatmap, threshold):#115
    max_values_filtered=cv2.threshold(gray_heatmap,threshold,255,cv2.THRESH_TOZERO)
    for i in range(0, len(gray_heatmap)):
        if (sum(max_values_filtered[1][i,:]) != 0):
            row_max = i
        if (sum(max_values_filtered[1][:,i]) != 0):
            col_max = i
    for j in reversed(range(len(gray_heatmap))):
        if (sum(max_values_filtered[1][j,:]) != 0):
            row_min = j
        if (sum(max_values_filtered[1][:,j]) != 0):
            col_min = j
    row_centroid = round((row_max-row_min)/2)+row_min
    col_centroid = round((col_max-col_min)/2)+col_min
    return col_centroid, row_centroid #(x,y)
def calculateMaxInInterval(input):
    data = []
    juntos = 0
    data_array = cv2.threshold(np.array(input), 0.9, 1, cv2.THRESH_TOZERO)
    for i in range(0, len(data_array[1])) :
        if data_array[1][i] != 0 :
            if juntos == 0 :
                ini = i
            juntos += 1
        else :
            if juntos != 0 :
                data.append((ini, i))
            juntos = 0
    data.append((ini, i))
    juntos = 0
    for i in range(0, len(data)):
        aux = data[i][1]-data[i][0]
        if juntos < aux:
            juntos = aux
            interval = data[i]
    position = interval[0]+data_array[1][interval[0]:interval[1]].argmax()
    return position
def calculateMinCentroid(gray_heatmap):
    gray_heatmap_inv = abs(gray_heatmap-255)/255
    columns_weights = []
    rows_weights = []
    for i in range(0, len(gray_heatmap)):
        columns_weights.append( sum(gray_heatmap_inv[:,i])/len(gray_heatmap) )
        rows_weights.append( sum(gray_heatmap_inv[i,:])/len(gray_heatmap) )
    col_centroid = calculateMaxInInterval(columns_weights)
    row_centroid = calculateMaxInInterval(rows_weights)

    return col_centroid, row_centroid #(x,y)

# ------------------------ Constantes ---------------------------------------
DATA_ID = "EfficientNetB0"
DATA_PATH = "C:/Users/User/TFG-repository/Imagenet/variablesIndividuales_Test_%s/" % (DATA_ID)
NUM_ATCKS = 1 #Numero de ataques distintos que se usaron cuando se guardaron las imagenes

sorted_data, name_list = aux.loadImagesSorted(DATA_PATH)
NUM_IMG = len(sorted_data)

# ------------------------ Operaciones --------------------------------------
calculate_metrics = False #tarda 'poco'
execute_Histogram = False #tarda mucho
execute_BoxPlot = False

if calculate_metrics == True:
    metricsName = ["Nombre Imagen", "Ataque", "Epsilon", "Predicción", "Media", "Media normalizada", "Mediana",
                   "Varianza", "Desviación Típica", "Dif de medias", "Norma Mascara", "Norma Imagen", "MSE",
                   "PSNR", "SSIM"]
    aux.createCsvFile(DATA_ID+"_metrics.csv", metricsName)

bins = 24
mean_heatmap_array_advNatural = []
mean_heatmap_array_advArtificial = []
mean_heatmap_array_original = []
freq_heatmap_array_advNatural = np.zeros(shape=bins)
freq_heatmap_array_advArtificial = np.zeros(shape=bins)
freq_heatmap_array_original = np.zeros(shape=bins)
for num in range(0, NUM_IMG):
    gray_heatmap = gradCamInterface.display_gray_gradcam(sorted_data[num].data, sorted_data[num].heatmap,
                                                         superimposed=False)
    gray_heatmap = cv2.cvtColor(gray_heatmap, cv2.COLOR_RGB2GRAY)  # plt.imshow(gray, cmap='gray')
    heatmap_array = gray_heatmap.flatten()
    metricsValue = []
    metricsValue = writeDataImageInCSV(metricsValue, sorted_data[num]) #Se escribe "Nombre Imagen", "Ataque", "Epsilon", "Prediccion"
    """result = shapiro(heatmap_array)
    if result[1] < 0.05:
        print('La muestra no sigue una distribucion normal')
    else:
        print('No se tienen suficientes evidencias para asegurar que es una distribución normal')"""
    if calculate_metrics == True:
        if metricsValue[1] != "Original" :
            closer_orig_img = searchCloserOriginalImage(sorted_data, name_list, num)
            heatmap_ref = gradCamInterface.display_gray_gradcam(closer_orig_img.data, closer_orig_img.heatmap,
                                                                superimposed=False)
            heatmap_ref = cv2.cvtColor(heatmap_ref, cv2.COLOR_RGB2GRAY)

        metricsValue.append(round(gray_heatmap.mean(), 2)) #"Media"
        metricsValue.append(round(gray_heatmap.mean()/255, 2)) #"Media normalizada"
        metricsValue.append(median(heatmap_array)) #"Mediana"
        metricsValue.append(round(gray_heatmap.var(), 2)) #"Varianza"
        metricsValue.append(round(gray_heatmap.std(), 2)) #"Desviación Típica"
        #No tiene sentido calcular lo siguiente: "Valor máximo" #"Posición del máximo" #"Valor mínimo"
        #"Posición del mínimo" #"Dif de máximo" #"Distancia entre máximos", #"Dif de mínimo" #"Distancia entre mínimos"
        #Pues el valor maximo es 255 el minimo 0 y hay varios...
        if metricsValue[1] != "Original": #Si no es la imagen original
            # La norma es la distancia euclidea
            metricsValue.append(round(gray_heatmap.mean(), 2)-round(heatmap_ref.mean(), 2)) #"Dif de medias"
            metricsValue.append(round(np.linalg.norm(gray_heatmap - heatmap_ref), 2)) #"Norma Mascara"
            metricsValue.append(round(np.linalg.norm(sorted_data[num].data - closer_orig_img.data), 2)) #"Norma Imagen"
            metricsValue.append(round(mean_squared_error(gray_heatmap, heatmap_ref), 2)) #Mean Squared Error (MSE)
            metricsValue.append(round(cv2.PSNR(gray_heatmap, heatmap_ref), 2)) #Peak Signal-to-Noise Ratio (PSNR)
            metricsValue.append(round(ssim(gray_heatmap, heatmap_ref, data_range=255, channel_axis=-1), 2))#SSIM. The higher the value, the more "similar" the two images are.
            #Desviacion media absoluta? MAD stadistica xd
            #VIF (Visual Information Fidelity - ?Factor de inflación de la varianza)https://github.com/pavancm/Visual-Information-Fidelity---Python https://github.com/baidut/matLIVE
            #IFC (Information Fidelity Criterion)
            #MAD (Most Apparent Distortion) https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/MAD.py
            #L2
        else:
            param=len(metricsName)-len(metricsValue)
            for index in range(0, param):
                metricsValue.append("-")

        aux.addRowToCsvFile(DATA_ID+"_metrics.csv", metricsName, metricsValue)

    #heatmap_array_sinMenorQue25 = [x for x in heatmap_array if x > 25]
    #https://joserzapata.github.io/courses/python-ciencia-datos/visualizacion/
    #se ejecuta por imagen
    if execute_Histogram == True:
        aux.saveHistogram(heatmap_array, sorted_data[num], DATA_ID)#Parece distribucion geometrica/exponencial xd
    if execute_BoxPlot == True:
        aux.saveBoxPlot(heatmap_array, sorted_data[num], DATA_ID)

    meanPerBin, freqPerBin, x = meanFreqPerBin(bins, heatmap_array)
    if metricsValue[1] != "Original":
        if metricsValue[1] == "Adv. Natural" :
            mean_heatmap_array_advNatural += meanPerBin
            for i in range(0, bins):
                freq_heatmap_array_advNatural[i] += freqPerBin[i]

        else :
            mean_heatmap_array_advArtificial += meanPerBin
            for i in range(0, bins):
                freq_heatmap_array_advArtificial[i] += freqPerBin[i]
    else:
        mean_heatmap_array_original += meanPerBin
        for i in range(0, bins):
            freq_heatmap_array_original[i] += freqPerBin[i]

freq_heatmap_array_original = freq_heatmap_array_original/500  #total imagenes
freq_heatmap_array_advNatural = freq_heatmap_array_advNatural/500
freq_heatmap_array_advArtificial = freq_heatmap_array_advArtificial/500
mean500_Orig, x, std_orig = meanFreqPerBin(bins, mean_heatmap_array_original)
mean500_AdvNat, x, std_nat = meanFreqPerBin(bins, mean_heatmap_array_advNatural)
mean500_AdvArt, x, std_art = meanFreqPerBin(bins, mean_heatmap_array_advArtificial)
aux.saveBarWithError(mean500_Orig, freq_heatmap_array_original, std_orig, "originales", DATA_ID)
aux.saveBarWithError(mean500_AdvNat, freq_heatmap_array_advNatural, std_nat, "adv. naturales", DATA_ID)
aux.saveBarWithError(mean500_AdvArt, freq_heatmap_array_advArtificial, std_art, "adv. artificiales, ", DATA_ID,
                     "FastGradientMethod")
createDataFrameToPlot(freq_heatmap_array_original, freq_heatmap_array_advNatural, freq_heatmap_array_advArtificial,
                      std_orig, std_nat, std_art)
aux.saveMeanLineWithError(mean500_Orig, mean500_AdvNat, mean500_AdvArt, freq_heatmap_array_original,
                      freq_heatmap_array_advNatural, freq_heatmap_array_advArtificial, std_orig, std_nat, std_art,
                      DATA_ID, "FastGradientMethod")

mean_freq_Orig = combineMeanValueWithFreq(mean500_Orig, freq_heatmap_array_original)
mean_freq_AdvNat = combineMeanValueWithFreq(mean500_AdvNat, freq_heatmap_array_advNatural)
mean_freq_AdvArt = combineMeanValueWithFreq(mean500_AdvArt, freq_heatmap_array_advArtificial)
summary_boxplot=[]
summary_boxplot.append(mean_freq_Orig)
summary_boxplot.append(mean_freq_AdvNat)
summary_boxplot.append(mean_freq_AdvArt)
#aux.saveBoxPlot(summary_boxplot, "", DATA_ID)
#aux.saveBoxPlot(summary_boxplot, "", DATA_ID, violin=True)
createDataFrameToPlot(mean_freq_Orig[0:50151], mean_freq_AdvNat[0:50151], mean_freq_AdvArt[0:50151], std_orig, std_nat, std_art, violin=True)