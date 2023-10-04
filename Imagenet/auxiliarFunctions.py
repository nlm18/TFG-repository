from __future__ import print_function

from Imagen import Imagen
import gradCamInterface

import cv2
import csv
import os
import math
import errno
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
#Mas ataques: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#fast-gradient-method-fgm
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, CarliniLInfMethod, HopSkipJump
from keras.applications.efficientnet import decode_predictions
# ------------------------ Funciones auxiliares ---------------------------------
def generateRandomVector(n, total_img):
    A = []
    i = 0
    while (i < n) :
        A.append(random.randint(0, total_img - 1))
        i += 1
    return A
def searchDirectory(num, path):
    img_per_directory = 50
    index = math.ceil(num/img_per_directory)-1
    list_dir = os.listdir(path)
    dir_name = list_dir[index]
    return dir_name
def searchImageInDirectory(num, path, img_per_directory):
    if num < img_per_directory:
        index = num-1
    else:
        index = (num - (math.ceil(num/img_per_directory)-1)*img_per_directory)-1
    list_img = os.listdir(path)
    img_name = list_img[index]
    return img_name
def loadImages(path, index_vector, size=(224,224), createImages=True, unclassified_images=False, realID =''):
    X_test = np.ndarray(shape=(len(index_vector), 224, 224, 3), dtype='float32')
    img_test = []
    if createImages == True :
        img_path = ""
        for index in range(0, len(index_vector)):
            if unclassified_images:
                img_path=path
                files_names = os.listdir(path)
                file_name=files_names[index]
                ID = realID
            else:#Imagenet
                dir_name = searchDirectory(index_vector[index], path)
                img_path = path + dir_name + "/"
                file_name = searchImageInDirectory(index_vector[index], img_path, 50)
                ID = dir_name
            img_path += file_name
            # Preprocess data
            X_test[index] = gradCamInterface.get_img_array_path(img_path, size)
            imagen = Imagen(file_name, X_test[index], size, ID, '')
            img_test.append(imagen)
            # preprocess_input(gradCamInterface.get_img_array_path(img_path[index], size))
    else :
        img_test = loadVariable(path+"testImages_efficientnetB0_random%simages.pkl" % (len(index_vector)))
    return X_test, img_test

def loadImagesByID(data_path, data_ID):
    list_files_names = os.listdir(data_path)
    img_adv_name = [x for x in list_files_names if data_ID+"_Adv" in x]
    img_orig_name = [x for x in list_files_names if data_ID+"_test" in x]

    img_adv = loadVariable(data_path+img_adv_name[0])
    img_orig = loadVariable(data_path+img_orig_name[0])
    return img_orig, img_adv

def createAdvImagenFromOriginal(original, adv_data, attackName, epsilon, predictionID=0):
    imagen = original.copyImage()
    if original.advNatural:
        imagen.modifyData(adv_data*0)
    else:
        imagen.modifyData(adv_data)
        imagen.addAdversarialData(attackName, epsilon)
        if predictionID  != 0:
            imagen.addPrediction(predictionID)
    return imagen
def isValidToCreateAdversarialExample(originalImage, classifier, isImagenet):
    img_array = gradCamInterface.get_img_array(originalImage.data)
    preds = classifier.model.predict(img_array)
    p = decode_predictions(preds, top=1)
    originalImage.addPrediction(p[0][0][0])
    if p[0][0][0] != originalImage.id:
        if isImagenet == False:
            originalImage.addAdvNatural(True)
        return False
    else:
        return True

def isAnAdversarialExample(originalImage, adv_img, classifier):
    isValid = False
    adv_array = gradCamInterface.get_img_array(adv_img)
    preds = classifier.predict(adv_array)
    p = decode_predictions(preds, top=1)
    if p[0][0][0] != originalImage.id:
        isValid = True
    return isValid, p[0][0][0]

def generateAdversarialImages(originalImages, x_test, attackName, epsilon, classifier, createImages=True, saveIndividualAttack=False, isImagenet=True):
    # Para distintos valores de epsilon
    img_array_test = np.ndarray(shape=(len(x_test), 224, 224, 3), dtype='float32')
    img_adv = [] # Guarda todas las imagenes seguidas, para recorrerlo:
                 # num+NUM_IMG*(indiceAtaque)*len(epsilon)+NUM_IMG*(indiceEpsilon)
    if createImages == True :
        for atck in range(0, len(attackName)) :
            individual_atck = []
            for i in range(0, len(epsilon)) :
                if attackName == 'HopSkipJump' and i > 0 and saveIndividualAttack:
                    # Guardo el hopskipjump porque no depende de epsilon
                    filename = "Adv_Images_AttackMethod_" + attackName[atck] + "max_iter_50.pkl"
                    saveVariable(individual_atck, filename)
                else :
                    # Si es un adversario natural o no acierta la imagen original de imagenet no hace falta que genere imagenes con ataques
                    for img in range(0, len(originalImages)) :
                        if isValidToCreateAdversarialExample(originalImages[img], classifier, isImagenet) == False:
                            img_array_test[img] = x_test[img]*0
                        else:
                            img_array_test[img] = x_test[img]
                    # Generate adversarial test examples
                    attack = getAttackMethod(attackName[atck], classifier, epsilon[i])
                    x_test_adv = attack.generate(x=img_array_test)
                    for img in range(0, len(originalImages)):
                        adv_imagen = createAdvImagenFromOriginal(originalImages[img], x_test_adv[img], attackName[atck], epsilon[i])
                        img_adv.append(adv_imagen)
                        individual_atck.append(adv_imagen)
            if saveIndividualAttack:
                filename = "Adv_Images_AttackMethod_" + attackName[atck] + "_Epsilon_%s" % (epsilon) + ".pkl"
                saveVariable(individual_atck, filename)
    else :
        for atck in range(0, len(attackName)) :
            filename = "Adv_Images_AttackMethod_" + attackName[atck] + "_Epsilon_%s" % (epsilon) + ".pkl"
            if atck == 0 :
                img_adv = loadVariable(filename)
            else :
                img_adv.append(loadVariable(filename))
    return img_adv

def generateAnAdversarialImage(originalImage, x_test, attackName, classifier, isImagenet=True):
    # Para distintos valores de epsilon
    img_array_test = np.ndarray(shape=(1, 224, 224, 3), dtype='float32')
    epsilon = 2500
    # Si es un adversario natural o no acierta la imagen original de imagenet no hace falta que genere imagenes con ataques
    if isValidToCreateAdversarialExample(originalImage, classifier, isImagenet) == False:
        img_array_test[0] = x_test*0
    else:
        img_array_test[0] = x_test
    # Generate adversarial test examples
    while True:
        attack = getAttackMethod(attackName, classifier, epsilon)
        x_test_adv = attack.generate(x=img_array_test)
        isValidAdversarial, predictionID = isAnAdversarialExample(originalImage, x_test_adv[0], classifier)
        if isValidAdversarial:
            break
        else:
            epsilon += 2500

    adv_imagen = createAdvImagenFromOriginal(originalImage, x_test_adv[0], attackName, epsilon, predictionID)
    return adv_imagen

def saveVariable(datos, filename):
    with open(filename, "wb") as f:
        pickle.dump(datos, f)

def loadVariable(filename):
     with open(filename, "rb") as f:
         return pickle.load(f)

def getAttackMethod(name, classifier, epsilon):
#attackName = ['FastGradientMethod', 'BasicIterativeMethod', 'ProjectedGradientDescent', 'CarliniLInfMethod', 'HopSkipJump']
    if name == 'FastGradientMethod':
        return FastGradientMethod(estimator=classifier, eps=epsilon, norm=2, batch_size=4)
    elif name == 'BasicIterativeMethod':
        return BasicIterativeMethod(estimator=classifier, eps=epsilon, max_iter=100, batch_size=4)
    elif name == 'ProjectedGradientDescent':
        return ProjectedGradientDescent(estimator=classifier, eps=epsilon, max_iter=100, batch_size=4)
    elif name == 'CarliniLInfMethod':
        return CarliniLInfMethod(classifier=classifier, confidence=epsilon, learning_rate=0.2, max_iter=10, batch_size=4)
    elif name == 'HopSkipJump':
        return HopSkipJump(classifier=classifier, max_iter=50, batch_size=4)

def createFigure(list_of_images, imagen_data, resultColumn='GradCam'):
    if imagen_data[0].advNatural:
        num_rows = 1
    else:
        num_rows = len(imagen_data)
    fig, axs = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, 15), subplot_kw={'xticks':[], 'yticks':[]},
                            layout='compressed')
    ind = 0;
    ind_pred = 0;
    for ax in axs.flat :
        if ind > 1 and imagen_data[0].advNatural:
            break
        ax.imshow(list_of_images[ind])
        # Ponemos titulos y nombre de los ejes
        if ind == 0 :
            ax.set_ylabel('Original')

        if ind % 2 == 0 :  # Los pares tendran el valor predecido
            if ind_pred == 0 :
                predText = 'Predicted: %s' % (imagen_data[ind_pred].predictionName)
            else :
                predText = 'Predicted: %s' % (imagen_data[ind_pred].predictionName)

            ax.set_title(predText)
            if ind > 1 :
                ax.set_ylabel('Adversarial, $\epsilon$=%s' % (imagen_data[ind_pred].epsilon))
        else :  # Los impares seran las imagenes con gradCam
            ax.set_title(resultColumn)
            ind_pred += 1
        ind += 1
    if imagen_data[0].advNatural :
        suptitle = 'Real value: %s, Natural adversarial example' % (imagen_data[1].idName)
    else:
        suptitle = 'Real value: %s, attack method used: %s' % (imagen_data[1].idName, imagen_data[1].attackName)
    # Cogemos el valor de la primera imagen adversaria pues todas tienen el mismo attackName menos la original(posicion 0)
    fig.suptitle(suptitle)
    return fig

def saveResults(list_of_images, imagen_data, exec_ID='', type=''):
    fig = createFigure(list_of_images, imagen_data, resultColumn='GradCam ')
    try :
        os.mkdir('gradCam_examples_%s' % (exec_ID))
    except OSError as e :
        if e.errno != errno.EEXIST :
            raise
    try :
        os.mkdir('gradCam_examples_%s/NaturalAdversarial%s' % (exec_ID, type) )
        os.mkdir('gradCam_examples_%s/ArtificialAdversarial%s' % (exec_ID, type) )
    except OSError as e :
        if e.errno != errno.EEXIST :
            raise
    if imagen_data[0].advNatural:
        file_name = 'gradCam_examples_%s/NaturalAdversarial%s/gradCam_example_image-%s.jpg' % ( exec_ID, type, imagen_data[1].name)
    else:
        file_name = 'gradCam_examples_%s/ArtificialAdversarial%s/gradCam_example_image-%s_attack_method-%s.jpg' % ( exec_ID, type, imagen_data[1].name, imagen_data[1].attackName)

    fig.savefig(file_name)
    plt.close()

def plotDifferenceBetweenImages(original_img, adv_img, exec_ID=''):
    resultado = (abs(original_img.data-adv_img.data))*10000
    #Ponemos titulo
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Adv-Orig, $\epsilon$=%s'% (adv_img.epsilon))
    plt.imshow(resultado.astype(np.uint8))
    suptitle = 'Attack method used: %s' % (adv_img.attackName)
    plt.suptitle(suptitle)
    if original_img.advNatural == False :
        try :
            os.mkdir('gradCam_examples_%s/Difference_between_orig_adv_method-%s' % (exec_ID, adv_img.attackName))
        except OSError as e :
            if e.errno != errno.EEXIST :
                raise

        File_name = 'gradCam_examples_%s/Difference_between_orig_adv_method-%s/Difference_image-%s.jpg' % (exec_ID, adv_img.attackName, original_img.name)
        plt.savefig(File_name)
    plt.close()

def plotDifference(num, original_img, adversarial_img, n_iter, epsilon, exec_ID=''):
    div_entera = (len(epsilon) % 2 == 0)
    num_col = 1
    num_rows = len(epsilon)
    if div_entera:
        num_col = 2
        num_rows = math.ceil(len(epsilon) / 2)
    for j in range(0, len(epsilon)):
        NUM_IMG = len(original_img)
        adv_img = adversarial_img[num+NUM_IMG*n_iter*len(epsilon)+NUM_IMG*j]
        resultado = (abs(original_img[num].data-adv_img.data))*10000
        #Ponemos titulo
        plt.subplot(num_rows, num_col, j + 1)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('Adv-Orig, $\epsilon$=%s'% (epsilon[j]))
        plt.imshow(resultado.astype(np.uint8))
    suptitle = 'Attack method used: %s' % (adv_img.attackName)
    plt.suptitle(suptitle)
    try :
        os.mkdir('Difference_between_orig_adv_method-%s' % (adv_img.attackName))
    except OSError as e :
        if e.errno != errno.EEXIST :
            raise
    File_name = 'Difference_between_orig_adv_method-%s/%s_Difference_image-%s_attack_method-%s.jpg' % (adv_img.attackName, exec_ID, original_img[num].name, adv_img.attackName)
    plt.savefig(File_name)
    plt.close()

def isValidExample(num, original_img, adversarial_img, n_iter, epsilon, filter=True, isImagenet=True):
    if filter == False:
        return True
    saveSuccesfulExample = False
    total_img = len(original_img)
    # Si la red no ha acertado en la predicción de la imagen original, no se guarda la imagen
    if original_img[num].predictionId == original_img[num].id:
        for j in range(0, len(epsilon)) :
            index = num + total_img * n_iter * len(epsilon) + total_img * j
            # Si el adversario ha conseguido confundir a la red, se guarda la imagen
            if adversarial_img[index].predictionId != adversarial_img[index].id:
                saveSuccesfulExample = True
    # Si no es de imagenet se guarda si ha fallado la predicción de la imagen original
    if original_img[num].advNatural and (isImagenet == False):
        saveSuccesfulExample = True

    return saveSuccesfulExample

def isValidExample_sortedList(sorted_list):
    saveSuccesfulExample = False
    # Si la red no ha acertado en la predicción de la imagen original, no se guarda la imagen
    if sorted_list[0].predictionId == sorted_list[0].id:
        for ind in range(1, len(sorted_list)):
            # Si el adversario ha conseguido confundir a la red, se guarda la imagen
            if sorted_list[ind].predictionId != sorted_list[ind].id:
                saveSuccesfulExample = True
    # Si no es de imagenet se guarda si ha fallado la predicción de la imagen original
    if sorted_list[0].advNatural:
        saveSuccesfulExample = True
    return saveSuccesfulExample
def calculateAccuracy(img_test, img_adv, attackName, epsilon):
    total_img = len(img_test)
    # Porcentaje de acierto para las imagenes originales:
    hits = 0
    for ind in range(0, total_img):
        if img_test[ind].id == img_test[ind].predictionId:
            hits+=1
    accuracy = hits / total_img
    print("- Accuracy on benign test examples: {}%".format(accuracy * 100))

    # Porcentaje de acierto para las imagenes adversarias:
    for atck in range(0, len(attackName)):
        for eps in range(0, len(epsilon)):
            hits = 0
            for num in range(0, total_img):
                index = num + total_img*atck*len(epsilon) + total_img*eps
                if img_adv[index].id == img_adv[index].predictionId:
                    hits+=1
            accuracy = hits / total_img
            print("- Accuracy on adversarial test examples: {}%".format(accuracy * 100))
            print("\twith AttackMethod: %s with epsilon = %s" % (attackName[atck], epsilon[eps]))

def calculatePercentageNaturalAdversarial(img_test):
    total_img = len(img_test)
    # Porcentaje de adversarias naturales para las imagenes originales:
    hits = 0
    for ind in range(0, total_img):
        if img_test[ind].advNatural == True:
            hits+=1
    percentage = hits / total_img
    print("- Percentage of natural adversarial images: {}%".format(percentage * 100))

def createCsvFile(filename, fieldnames):
    #Comprobamos si existe el archivo
    list_files_names = os.listdir("C:/Users/User/TFG-repository/Imagenet/")
    csvName = [x for x in list_files_names if filename+".csv" in x]
    if csvName == []:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames,)
            writer.writeheader()
        return ""
    else:
        return "error"
def addRowToCsvFile(filename, fieldnames, data):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        row={}
        for i in range(0,len(fieldnames)):
            row[fieldnames[i]]=data[i]
        writer.writerow(row)

def saveHistogram(sorted_list, DATA_ID):
    try :
        os.mkdir('histogram-%s' % (DATA_ID))
    except OSError as e :
        if e.errno != errno.EEXIST :
            raise
    for ind in range(0, len(sorted_list)) :
        gray_heatmap = gradCamInterface.display_gray_gradcam(sorted_list[ind].data, sorted_list[ind].heatmap,
                                                             superimposed=False)
        gray_1channel = cv2.cvtColor(gray_heatmap, cv2.COLOR_RGB2GRAY) #plt.imshow(gray_1channel, cmap='gray')
        type = defineTypeOfAdversarial(sorted_list[ind])

        intervalos = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255]  # indicamos los extremos de los intervalos

        plt.hist(x=gray_1channel, bins=intervalos, rwidth=0.85 )
        plt.title('Histograma del mapa de activación, imagen %s' %(type))
        plt.xlabel('Intensidad del mapa de activación')
        plt.ylabel('Frecuencia')
        plt.xticks(intervalos)

        #plt.show()  # dibujamos el histograma
        plt.savefig("histogram-%s/histogram_" % (DATA_ID) + type + "_" + sorted_list[ind].name)

def defineTypeOfAdversarial(img):
    if img.attackName == "":
        if img.predictionId == img.id:
            result = "Original"
        else:
            result = "AdvNatural"
    else:
        result = img.attackName+"_Eps_%s" %(img.epsilon)
    return result

def printResultsPerImage(orig, adv):
    print("Real value: ", orig.idName)
    if orig.advNatural :
        print("Predicted benign example: ", orig.predictionName, " NATURAL ADVERSARIAL EXAMPLE")
    else :
        print("Predicted benign example: ", orig.predictionName)
    if orig.advNatural == False :
        print("AttackMethod: %s with epsilon = %s" % (adv.attackName, adv.epsilon))
        print("Predicted adversarial example: ", adv.predictionName)