import numpy as np
import os
import gc

import auxiliarFunctions as aux
import auxiliarMetricsFunctions as mf
from scipy.stats import ttest_ind

def initializeData2Csv(list_data):
    data2csv = [["%s" % (list_data[0].networkModelName),"Original - Adv. Natural"]]
    for atck in range(0, len(list_data)-2) :
        data2csv.append(["%s" % (list_data[0].networkModelName),
                         "Original - Adv. %s" % (list_data[atck+2].imageType)])
        data2csv.append(["%s" % (list_data[0].networkModelName),
                         "Adv. Natural - Adv. %s" % (list_data[atck+2].imageType)])
    return data2csv

def obtainListFromObjectMetricsData(metricsData, original = False):
    resultList = []
    resultList.append(metricsData.MediaIntensidadPixeles)
    resultList.append(metricsData.Mediana)
    resultList.append(metricsData.VarianzaPixeles)
    resultList.append(metricsData.DesviacionTipicaPixeles)
    if original:
        for i in range(0,8):
            resultList.append("-")
    else:
        resultList.append(metricsData.DistanciaCentroideMax)
        resultList.append(metricsData.DistanciaCentroideMin)
        resultList.append(metricsData.DifMedias)
        resultList.append(metricsData.NormaMascara)
        resultList.append(metricsData.NormaImagen)
        resultList.append(metricsData.MSE)
        resultList.append(metricsData.PSNR)
        resultList.append(metricsData.SSIM)
    return resultList

#https://www.jmp.com/es_es/statistics-knowledge-portal/t-test/two-sample-t-test.html
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
# ------------------------ Constantes ---------------------------------------
DATA_PATH = "C:/Users/User/TFG-repository/Imagenet/case2/"
DATA_ID = "case2_test"
list_files_names = os.listdir(DATA_PATH)
list_data = []
var = []

# ------------------------ Operaciones --------------------------------------
header = ["-1", "-2", "Media de la intensidad de los pixeles", "-3",
          "Mediana de la intensidad de los pixeles", "-4",
          "Varianza de la intensidad de los pixeles", "-5",
          "Desviación típica de la intensidad de los pixeles", "-6",
          "Distancia al centroide máximo del heatmap", "-7",
          "Distancia al centroide mínimo del heatmap", "-8",
          "Diferencia entre las medias de la intensidad de pixel con respecto a la original", "-9",
          "Norma máscara", "1-",
          "Norma imagen", "2-",
          "MSE", "3-",
          "PSNR", "4-",
          "SSIM", "5-"]
metricsName = ["Nombre de la red", "Imágenes a comparar",
               "t-statistic1", "p-valor1",
               "t-statistic2", "p-valor2",
               "t-statistic3", "p-valor3",
               "t-statistic4", "p-valor4",
               "t-statistic5", "p-valor5",
               "t-statistic6", "p-valor6",
               "t-statistic7", "p-valor7",
               "t-statistic8", "p-valor8",
               "t-statistic9", "p-valor9",
               "t-statistic10", "p-valor10",
               "t-statistic11", "p-valor11",
               "t-statistic12", "p-valor12"]
aux.createCsvFile(DATA_ID + "_tstatistic_metrics.csv", header)
aux.addRowToCsvFile(DATA_ID + "_tstatistic_metrics.csv", header, metricsName)

for network in range (0, len(list_files_names)):
    list_data.append(aux.loadVariable(DATA_PATH+list_files_names[network]))
    num_atcks = len(list_data[network])-2
    data2csv = initializeData2Csv(list_data[network]) #quiero intentar crear una lista de listas para guardar por filas lo que habra en el csv
    '''for type in range(0, len(list_data[network])): #para ver si se puede aplicar el t-statistic
        var.append(round(np.array(list_data[network][type].MediaIntensidadPixeles).var(), 2))
        mf.saveHistogram(list_data[network][type].MediaIntensidadPixeles, list_data[network][type].imageType)#, False)
'''
    ''' Se escribira en la excel:
        Nombre de la red | Imagenes a comparar | t-statistic1 | p-valor1 | grados de libertad1 | ...
        EfficientNetB0 | Original - Adv. Natural | t1       | p1       | df1 | ...
        EfficientNetB0 | Original - Adv Art1     | t1       | p1       | df1 | ...
        EfficientNetB0 | Adv. Natural - Adv Art1 | t1       | p1       | df1 | ...
        ...
        EfficientNetB0 | Original - Adv Art2     | t1       | p1       | df1 | ...
        EfficientNetB0 | Adv. Natural - Adv Art2 | t1       | p1       | df1 | ...
        ... '''
    org = obtainListFromObjectMetricsData(list_data[network][0])
    nat = obtainListFromObjectMetricsData(list_data[network][1])
    atcks = []
    for atck in range(0, num_atcks):#1 2,  3  4 +1 y +2 ej 2*0+1  2*0+2  2*1+1 2*1+2 2*2+1 2*2+2
        atcks.append(obtainListFromObjectMetricsData(list_data[network][atck + 2]))

    for metrics in range(0, len(org)):
        if org[metrics][0] != "-": #original - natural
            org_nat = ttest_ind(org[metrics], nat[metrics])
            data2csv[0] += [round(org_nat.statistic,3), round(org_nat.pvalue,3)]
        else:
            data2csv[0] += ["-", "-"]

        for atck in range(0, num_atcks):
            if org[metrics][0] != "-": #original - adv art
                org_adv = ttest_ind(org[metrics], atcks[atck][metrics])
                data2csv[2*atck+1] += [round(org_adv.statistic,3), round(org_adv.pvalue,3)]
            else:
                data2csv[2*atck+1] += ["-", "-"]
            nat_adv = ttest_ind(nat[metrics], atcks[atck][metrics])
            data2csv[2*atck+2] += [round(nat_adv.statistic,3), round(nat_adv.pvalue,3)]

    for i in range(0,len(data2csv)):
        aux.addRowToCsvFile(DATA_ID + "_tstatistic_metrics.csv", header, data2csv[i])
    del nat_adv, org_nat, org_adv, data2csv
    gc.collect()
    print("y se borró y a su barco le llamó libertaaad")
    #si no es significativa en muchos casos a lo mejor esa metrica no nos vale para distinguir entre art y natural
