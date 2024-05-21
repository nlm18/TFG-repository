from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from statistics import median
import cv2
import numpy as np
import gradCamInterface
import auxiliarFunctions as aux
import auxiliarMetricsFunctions as mf

# ------------------------ Constantes ---------------------------------------
DATA_ID = "Xception"
DATA_PATH = "C:/Users/User/TFG-repository/Imagenet/case1/variablesIndividuales_%s/" % (DATA_ID)
ATCK_NAME = ["BoundaryAttack", "FastGradientMethod", "ProjectedGradientDescent"]
NUM_ATCKS = len(ATCK_NAME) #Numero de ataques distintos que se usaron cuando se guardaron las imagenes
sorted_data, name_list = aux.loadImagesSorted(DATA_PATH, NUM_ATCKS)
NUM_IMG = len(sorted_data)

# ------------------------ Operaciones --------------------------------------
calculate_metrics = True
execute_Histogram = False
execute_BoxPlot = False

if calculate_metrics == True:
    metricsName = ["Nombre Imagen", "Ataque", "Epsilon", "Predicción", "Media", "Media normalizada", "Mediana",
                   "Varianza", "Desviación Típica", "Centroide Máximo", "Distancia centroides máximos",
                   "Centroide Mínimo", "Distancia centroides mínimos", "Dif de medias", "Diferencia Norma Mascara",
                   "Diferencia Norma Imagen", "MSE", "PSNR", "SSIM"]
    aux.createCsvFile(DATA_ID+"_metrics.csv", metricsName)

bins = 24
mean_heatmap_array_advNatural = []
mean_heatmap_array_advArtificial = []
mean_heatmap_array_original = []
freq_heatmap_array_advNatural = []
freq_heatmap_array_advArtificial = []
freq_heatmap_array_original = []
if len(ATCK_NAME) != 1 :
    valoresMetricas = mf.initializeVariablesToSave(ATCK_NAME, sorted_data[0]) #Dentro del vector ira Orig - AdvNatural - Adv1 - Adv2...

for num in range(0, NUM_IMG):
    gray_heatmap = gradCamInterface.display_gray_gradcam(sorted_data[num].data, sorted_data[num].heatmap,
                                                         superimposed=False)
    gray_heatmap = cv2.cvtColor(gray_heatmap, cv2.COLOR_RGB2GRAY)
    heatmap_array = gray_heatmap.flatten()
    metricsValue = []
    metricsValue = mf.writeDataImageInCSV(metricsValue, sorted_data[num]) #Se escribe "Nombre Imagen", "Ataque", "Epsilon", "Prediccion"

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
            metricsValue.append(abs(round(gray_heatmap.mean()-heatmap_ref.mean(), 2))) #"Dif de medias"
            metricsValue.append(round(np.linalg.norm(gray_heatmap - heatmap_ref), 2)) #"Norma Mascara"
            metricsValue.append(round(np.linalg.norm(sorted_data[num].data - closer_orig_img.data), 2)) #"Norma Imagen"
            metricsValue.append(round(mean_squared_error(gray_heatmap, heatmap_ref), 2)) #Mean Squared Error (MSE)
            metricsValue.append(round(cv2.PSNR(gray_heatmap, heatmap_ref), 2)) #Peak Signal-to-Noise Ratio (PSNR)
            metricsValue.append(round(ssim(gray_heatmap, heatmap_ref, data_range=255, channel_axis=-1), 2))#Structure Similarity Index Method (SSIM)
        else:
            param=len(metricsName)-len(metricsValue)
            for index in range(0, param):
                metricsValue.append("-")

        aux.addRowToCsvFile(DATA_ID+"_metrics.csv", metricsName, metricsValue)
        if len(ATCK_NAME) != 1 :
            mf.saveMetricsInVariable(ATCK_NAME, metricsValue, valoresMetricas)

    #se ejecuta por imagen
    if execute_Histogram == True:
        aux.saveHistogram(heatmap_array, sorted_data[num], DATA_ID)
    if execute_BoxPlot == True:
        aux.saveBoxPlot(heatmap_array, sorted_data[num], DATA_ID)

    if len(ATCK_NAME) == 1 :  # No es un vector por lo que puedo generar las graficas del ataque correspondiente
        meanPerBin, freqPerBin = mf.meanFreqPerBin(bins, heatmap_array)
        if metricsValue[1] != "Original":
            if metricsValue[1] == "Adv. Natural":
                mean_heatmap_array_advNatural += meanPerBin
                freq_heatmap_array_advNatural.append(freqPerBin)

            else :
                mean_heatmap_array_advArtificial += meanPerBin
                freq_heatmap_array_advArtificial.append(freqPerBin)
        else:
            mean_heatmap_array_original += meanPerBin
            freq_heatmap_array_original.append(freqPerBin)
if len(ATCK_NAME) == 1 :  # No es un vector por lo que puedo generar las graficas del ataque correspondiente
    freq500_Orig, std_orig = mf.meanFreqTotalImgPerBin(freq_heatmap_array_original)
    freq500_AdvNat, std_nat = mf.meanFreqTotalImgPerBin(freq_heatmap_array_advNatural)
    freq500_AdvArt, std_art = mf.meanFreqTotalImgPerBin(freq_heatmap_array_advArtificial)
    mean500_Orig, x = mf.meanFreqPerBin(bins, mean_heatmap_array_original)
    mean500_AdvNat, x = mf.meanFreqPerBin(bins, mean_heatmap_array_advNatural)
    mean500_AdvArt, x = mf.meanFreqPerBin(bins, mean_heatmap_array_advArtificial)
    aux.saveBarWithError(mean500_Orig, freq500_Orig, std_orig, "originales", DATA_ID)
    aux.saveBarWithError(mean500_AdvNat, freq500_AdvNat, std_nat, "adv. naturales", DATA_ID)
    aux.saveBarWithError(mean500_AdvArt, freq500_AdvArt, std_art, "adv. artificiales,", DATA_ID,
                         ATCK_NAME[0])
    mf.createDataFrameToPlot(freq500_Orig, freq500_AdvNat, freq500_AdvArt,
                          std_orig, std_nat, std_art, DATA_ID, atck=ATCK_NAME[0])
    aux.saveMeanLineWithError(mean500_Orig, mean500_AdvNat, mean500_AdvArt, freq500_Orig,
                          freq500_AdvNat, freq500_AdvArt, std_orig, std_nat, std_art,
                          DATA_ID, ATCK_NAME[0])

    mean_freq_Orig = mf.combineMeanValueWithFreq(mean500_Orig, freq500_Orig)
    mean_freq_AdvNat = mf.combineMeanValueWithFreq(mean500_AdvNat, freq500_AdvNat)
    mean_freq_AdvArt = mf.combineMeanValueWithFreq(mean500_AdvArt, freq500_AdvArt)
    summary_boxplot=[]
    summary_boxplot.append(mean_freq_Orig)
    summary_boxplot.append(mean_freq_AdvNat)
    summary_boxplot.append(mean_freq_AdvArt)
    aux.saveBoxPlot(summary_boxplot, "", DATA_ID, atck=ATCK_NAME[0])
    aux.saveBoxPlot(summary_boxplot, "", DATA_ID, violin=True, atck=ATCK_NAME[0])
else:
    aux.saveVariable(valoresMetricas, "vectoresMetricas_%s.pkl" % (DATA_ID))