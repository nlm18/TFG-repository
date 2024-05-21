import numpy as np
import os
import gc
import auxiliarFunctions as aux
from scipy.stats import ttest_ind, wilcoxon

# ------------------------ Funciones auxiliares ---------------------------------
def initializeData2Csv(list_data):
    data2csv = [["%s" % (list_data[0].networkModelName),"Original - Adv. Natural"]]
    for atck in range(2, len(list_data)) :
        data2csv.append(["%s" % (list_data[0].networkModelName),
                         "Original - Adv. %s" % (list_data[atck].imageType)])
        data2csv.append(["%s" % (list_data[0].networkModelName),
                         "Adv. Natural - Adv. %s" % (list_data[atck].imageType)])
    for atck in range(2, len(list_data)):
        for index in range(atck+1, len(list_data)):
            data2csv.append(["%s" % (list_data[0].networkModelName),
                             "Adv. %s - Adv. %s" % (list_data[atck].imageType, list_data[index].imageType)])


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

# ------------------------ Constantes ---------------------------------------
DATA_PATH = "C:/Users/User/TFG-repository/Imagenet/case1/variables/"
DATA_ID = "case1_test_"
list_files_names = os.listdir(DATA_PATH)
list_data = []
var = []
test = 't' #'t' o 'Wilcoxon'

# ------------------------ Operaciones --------------------------------------
header = ["-1", "-2", "Media de la intensidad de los pixeles", "-3",
          "Mediana de la intensidad de los pixeles", "-4",
          "Varianza de la intensidad de los pixeles", "-5",
          "Desviación típica de la intensidad de los pixeles", "-6",
          "Distancia al centroide máximo del heatmap", "-7",
          "Distancia al centroide mínimo del heatmap", "-8",
          "Diferencia entre las medias de la intensidad de pixel con respecto a la original", "-9",
          "Norma de la diferencia entre la máscara con respecto a la original", "1-",
          "Norma de la diferencia entre la imagen adversaria con respecto a la original", "2-",
          "MSE", "3-",
          "PSNR", "4-",
          "SSIM", "5-"]
metricsName = ["Nombre de la red", "Imágenes a comparar",
               "%s-statistic1" % (test), "p-valor1",
               "%s-statistic2" % (test), "p-valor2",
               "%s-statistic3" % (test), "p-valor3",
               "%s-statistic4" % (test), "p-valor4",
               "%s-statistic5" % (test), "p-valor5",
               "%s-statistic6" % (test), "p-valor6",
               "%s-statistic7" % (test), "p-valor7",
               "%s-statistic8" % (test), "p-valor8",
               "%s-statistic9" % (test), "p-valor9",
               "%s-statistic10" % (test), "p-valor10",
               "%s-statistic11" % (test), "p-valor11",
               "%s-statistic12" % (test), "p-valor12",
               "%s-statistic13" % (test), "p-valor13",
               "%s-statistic14" % (test), "p-valor14"]
aux.createCsvFile(DATA_ID + "%s-statistic_metrics.csv" % (test), header)
aux.addRowToCsvFile(DATA_ID + "%s-statistic_metrics.csv" % (test), header, metricsName)

for network in range (0, len(list_files_names)):
    list_data.append(aux.loadVariable(DATA_PATH+list_files_names[network]))
    num_atcks = len(list_data[network])-2
    data2csv = initializeData2Csv(list_data[network])
    var_aux = []
    for type in range(0, len(list_data[network])):
        var_aux.append(round(np.array(list_data[network][type].MediaIntensidadPixeles).var(), 2))

    var.append(var_aux)
    ''' Se escribira en un excel:
        Nombre de la red | Imagenes a comparar | %s-statistic1 | p-valor1 | ...
        EfficientNetB0 | Original - Adv. Natural | t1       | p1       | ...
        EfficientNetB0 | Original - Adv Art1     | t1       | p1       | ...
        EfficientNetB0 | Adv. Natural - Adv Art1 | t1       | p1       | ...
        ...
        EfficientNetB0 | Original - Adv Art2     | t1       | p1       | ...
        EfficientNetB0 | Adv. Natural - Adv Art2 | t1       | p1       | ...
        ... '''
    org = obtainListFromObjectMetricsData(list_data[network][0], original=True)
    nat = obtainListFromObjectMetricsData(list_data[network][1])
    atcks = []
    for atck in range(0, num_atcks):
        atcks.append(obtainListFromObjectMetricsData(list_data[network][atck + 2]))

    for metrics in range(0, len(org)):
        if org[metrics][0] != "-": #original - natural
            if test == "Wilcoxon":
                org_nat = wilcoxon(org[metrics], nat[metrics], method='approx')
                data2csv[0] += [round(org_nat.statistic, 5), round(org_nat.pvalue, 5)]
            else:#t test
                org_nat = ttest_ind(org[metrics], nat[metrics], equal_var=False)
                data2csv[0] += [round(org_nat.statistic, 5), round(org_nat.pvalue, 5)]
        else:
            data2csv[0] += ["-", "-"]

        for atck in range(0, num_atcks):
            if org[metrics][0] != "-": #original - adv art
                if test == "Wilcoxon" :
                    org_adv = wilcoxon(org[metrics], atcks[atck][metrics], method='approx')
                    data2csv[2*atck+1] += [round(org_adv.statistic, 5), round(org_adv.pvalue, 5)]
                else:#t test
                    org_adv = ttest_ind(org[metrics], atcks[atck][metrics], equal_var=False)
                    data2csv[2*atck+1] += [round(org_adv.statistic, 5), round(org_adv.pvalue, 5)]

            else:
                data2csv[2*atck+1] += ["-", "-"]
            if test == "Wilcoxon" :
                nat_adv = wilcoxon(nat[metrics], atcks[atck][metrics], method='approx')
                data2csv[2*atck+2] += [round(nat_adv.statistic, 5), round(nat_adv.pvalue, 5)]
            else:#t test
                nat_adv = ttest_ind(nat[metrics], atcks[atck][metrics], equal_var=False)
                data2csv[2*atck+2] += [round(nat_adv.statistic, 5), round(nat_adv.pvalue, 5)]
        atck_comparision_index = 2*num_atcks
        for atck in range(0, num_atcks):
            for index in range(atck+1, num_atcks):
                if test == "Wilcoxon" :
                    adv_adv = wilcoxon(atcks[atck][metrics], atcks[index][metrics], method='approx')
                    data2csv[atck_comparision_index+atck+index] += [round(adv_adv.statistic, 5), round(adv_adv.pvalue, 5)]
                else :  # t test
                    adv_adv = ttest_ind(atcks[atck][metrics], atcks[index][metrics], equal_var=False)
                    data2csv[atck_comparision_index+atck+index] += [round(adv_adv.statistic, 5), round(adv_adv.pvalue, 5)]

    for i in range(0,len(data2csv)):
        aux.addRowToCsvFile(DATA_ID + "%s-statistic_metrics.csv" % (test), header, data2csv[i])
    del nat_adv, org_nat, org_adv, data2csv
    gc.collect()

print("se ejecutó %s test" % (test))