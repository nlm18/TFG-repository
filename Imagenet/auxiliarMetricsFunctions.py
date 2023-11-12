import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import plotly.graph_objects as go
import math

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
def calculateMaxCentroid(gray_heatmap, threshold):#215
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
    ini = 0
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

def writeCentroidsInCSV(data, gray_heatmap, threshold, gray_heatmap_orig):
    x2, y2 = calculateMaxCentroid(gray_heatmap, threshold)
    data.append("(%s,%s)" % (x2, y2))
    if gray_heatmap_orig != []:
        x1, y1 = calculateMaxCentroid(gray_heatmap_orig, threshold)
        dist_between_max = round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 2)
        data.append(dist_between_max)
    else:
        data.append("-")

    x2, y2 = calculateMinCentroid(gray_heatmap)
    data.append("(%s,%s)" % (x2, y2))
    if gray_heatmap_orig != []:
        x1, y1 = calculateMinCentroid(gray_heatmap_orig)
        dist_between_min = round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 2)
        data.append(dist_between_min)
    else:
        data.append("-")
    return data