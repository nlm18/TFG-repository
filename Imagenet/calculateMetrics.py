import numpy as np

from tensorflow import keras
from sklearn.metrics import mean_squared_error #con 3chnn no funciona
from skimage.metrics import structural_similarity as ssim
from statistics import median
import cv2
from scipy.stats import shapiro

import gradCamInterface
import auxiliarFunctions as aux
import auxiliarMetricsFunctions as mf

# ------------------------ Constantes ---------------------------------------
DATA_ID = "EfficientNetB0"
DATA_PATH = "C:/Users/User/TFG-repository/Imagenet/variablesIndividuales_Test_%s/" % (DATA_ID)
NUM_ATCKS = 1 #Numero de ataques distintos que se usaron cuando se guardaron las imagenes

sorted_data, name_list = aux.loadImagesSorted(DATA_PATH)
NUM_IMG = len(sorted_data)

# ------------------------ Operaciones --------------------------------------
calculate_metrics = True #tarda 'poco'
execute_Histogram = False #tarda mucho
execute_BoxPlot = False

if calculate_metrics == True:
    metricsName = ["Nombre Imagen", "Ataque", "Epsilon", "Predicción", "Media", "Media normalizada", "Mediana",
                   "Varianza", "Desviación Típica", "Centroide Máximo", "Distancia centroides máximos",
                   "Centroide Mínimo", "Distancia centroides mínimos" "Dif de medias", "Norma Mascara",
                   "Norma Imagen", "MSE", "PSNR", "SSIM"]
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
    metricsValue = mf.writeDataImageInCSV(metricsValue, sorted_data[num]) #Se escribe "Nombre Imagen", "Ataque", "Epsilon", "Prediccion"
    """result = shapiro(heatmap_array)
    if result[1] < 0.05:
        print('La muestra no sigue una distribucion normal')
    else:
        print('No se tienen suficientes evidencias para asegurar que es una distribución normal')"""
    if calculate_metrics == True:
        if metricsValue[1] != "Original" :
            closer_orig_img = mf.searchCloserOriginalImage(sorted_data, name_list, num)
            heatmap_ref = gradCamInterface.display_gray_gradcam(closer_orig_img.data, closer_orig_img.heatmap,
                                                                superimposed=False)
            heatmap_ref = cv2.cvtColor(heatmap_ref, cv2.COLOR_RGB2GRAY)
        else:
            heatmap_ref = []

        metricsValue.append(round(gray_heatmap.mean(), 2)) #"Media"
        metricsValue.append(round(gray_heatmap.mean()/255, 2)) #"Media normalizada"
        metricsValue.append(round(median(heatmap_array), 2)) #"Mediana"
        metricsValue.append(round(gray_heatmap.var(), 2)) #"Varianza"
        metricsValue.append(round(gray_heatmap.std(), 2)) #"Desviación Típica"
        mf.writeCentroidsInCSV(metricsValue, gray_heatmap, 215, heatmap_ref) #"Centroide Maximo","Distancia centroides maximos","Centroide Minimo","Distancia centroides minimos"

        if metricsValue[1] != "Original": #Si no es la imagen original
            # La norma es la distancia euclidea
            metricsValue.append(round(gray_heatmap.mean(), 2)-round(heatmap_ref.mean(), 2)) #"Dif de medias"
            metricsValue.append(round(np.linalg.norm(gray_heatmap - heatmap_ref), 2)) #"Norma Mascara"
            metricsValue.append(round(np.linalg.norm(sorted_data[num].data - closer_orig_img.data), 2)) #"Norma Imagen"
            metricsValue.append(round(mean_squared_error(gray_heatmap, heatmap_ref), 2)) #Mean Squared Error (MSE)
            metricsValue.append(round(cv2.PSNR(gray_heatmap, heatmap_ref), 2)) #Peak Signal-to-Noise Ratio (PSNR)
            metricsValue.append(round(ssim(gray_heatmap, heatmap_ref, data_range=255, channel_axis=-1), 2))#SSIM. The higher the value, the more "similar" the two images are.
            #Desviacion media absoluta?
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

    meanPerBin, freqPerBin, x = mf.meanFreqPerBin(bins, heatmap_array)
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
'''
freq_heatmap_array_original = freq_heatmap_array_original/500  #total imagenes
freq_heatmap_array_advNatural = freq_heatmap_array_advNatural/500
freq_heatmap_array_advArtificial = freq_heatmap_array_advArtificial/500
mean500_Orig, x, std_orig = mf.meanFreqPerBin(bins, mean_heatmap_array_original)
mean500_AdvNat, x, std_nat = mf.meanFreqPerBin(bins, mean_heatmap_array_advNatural)
mean500_AdvArt, x, std_art = mf.meanFreqPerBin(bins, mean_heatmap_array_advArtificial)
aux.saveBarWithError(mean500_Orig, freq_heatmap_array_original, std_orig, "originales", DATA_ID)
aux.saveBarWithError(mean500_AdvNat, freq_heatmap_array_advNatural, std_nat, "adv. naturales", DATA_ID)
aux.saveBarWithError(mean500_AdvArt, freq_heatmap_array_advArtificial, std_art, "adv. artificiales, ", DATA_ID,
                     "FastGradientMethod")
mf.createDataFrameToPlot(freq_heatmap_array_original, freq_heatmap_array_advNatural, freq_heatmap_array_advArtificial,
                      std_orig, std_nat, std_art)
aux.saveMeanLineWithError(mean500_Orig, mean500_AdvNat, mean500_AdvArt, freq_heatmap_array_original,
                      freq_heatmap_array_advNatural, freq_heatmap_array_advArtificial, std_orig, std_nat, std_art,
                      DATA_ID, "FastGradientMethod")

mean_freq_Orig = mf.combineMeanValueWithFreq(mean500_Orig, freq_heatmap_array_original)
mean_freq_AdvNat = mf.combineMeanValueWithFreq(mean500_AdvNat, freq_heatmap_array_advNatural)
mean_freq_AdvArt = mf.combineMeanValueWithFreq(mean500_AdvArt, freq_heatmap_array_advArtificial)
summary_boxplot=[]
summary_boxplot.append(mean_freq_Orig)
summary_boxplot.append(mean_freq_AdvNat)
summary_boxplot.append(mean_freq_AdvArt)
#aux.saveBoxPlot(summary_boxplot, "", DATA_ID)
#aux.saveBoxPlot(summary_boxplot, "", DATA_ID, violin=True)
mf.createDataFrameToPlot(mean_freq_Orig[0:50151], mean_freq_AdvNat[0:50151], mean_freq_AdvArt[0:50151], std_orig, std_nat, std_art, violin=True)'''