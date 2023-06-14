from __future__ import print_function

from Imagen import Imagen
import gradCamInterface

import os
import math
import errno
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
#Mas ataques: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#fast-gradient-method-fgm
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, CarliniLInfMethod, HopSkipJump

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
def loadImages(path, index_vector, size=(224,224), createImages=True):
    X_test = np.ndarray(shape=(len(index_vector), 224, 224, 3), dtype='float32')
    img_test = []
    if createImages == True :
        img_path = ""
        for index in range(0, len(index_vector)):
            dir_name = searchDirectory(index_vector[index], path)
            img_path = path + dir_name + "/"
            file_name = searchImageInDirectory(index_vector[index], img_path, 50)
            img_path += file_name
            # Preprocess data
            X_test[index] = gradCamInterface.get_img_array_path(img_path, size)
            imagen = Imagen(file_name, X_test[index], size, dir_name, '')
            img_test.append(imagen)
            # preprocess_input(gradCamInterface.get_img_array_path(img_path[index], size))
    else :
        img_test = loadVariable("testImages_efficientnetB0_random%simages.pkl" % (len(index_vector)))
    return X_test, img_test
def createAdvImagenFromOriginal(original, adv_data, attackName, epsilon):
    imagen = original.copyImage()
    imagen.modifyData(adv_data)
    imagen.addAdversarialData(attackName, epsilon)
    return imagen
def generateAdversarialImages(originalImages, x_test, attackName, epsilon, classifier, createImages=True, saveIndividualAttack=False):
    # Para distintos valores de epsilon
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
                    # Generate adversarial test examples
                    attack = getAttackMethod(attackName[atck], classifier, epsilon[i])
                    x_test_adv = attack.generate(x=x_test)
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

def saveResults(list_of_images, imagen_data, exec_ID=''):
    num_rows=len(imagen_data)
    fig, axs = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, 15), subplot_kw={'xticks': [], 'yticks': []}, layout='compressed')
    ind = 0;
    ind_pred = 0;
    for ax in axs.flat:
        ax.imshow(list_of_images[ind])
        #Ponemos titulos y nombre de los ejes
        if ind == 0:
            ax.set_ylabel('Original')

        if ind % 2 == 0: #Los pares tendran el valor predecido
            if ind_pred==0:
                predText = 'Predicted: %s'% (imagen_data[ind_pred].predictionName)
            else:
                predText = 'Predicted: %s' % (imagen_data[ind_pred].predictionName)

            ax.set_title(predText)
            if ind > 1:
                ax.set_ylabel('Adversarial, $\epsilon$=%s'% (imagen_data[ind_pred].epsilon))
        else: #Los impares seran las imagenes con gradCam
            ax.set_title('GradCam')
            ind_pred+=1
        ind+=1
    suptitle = 'Real value: %s, attack method used: %s' % (imagen_data[1].idName, imagen_data[1].attackName)
    # Cogemos el valor de la primera imagen adversaria pues todas tienen el mismo attackName menos la original(posicion 0)
    fig.suptitle(suptitle)
    try :
        os.mkdir('gradCam_examples_attack_method-%s' % (imagen_data[1].attackName))
    except OSError as e :
        if e.errno != errno.EEXIST :
            raise
    File_name = 'gradCam_examples_attack_method-%s/gradCam_example_image-%s_attack_method-%s%s.jpg' % (imagen_data[1].attackName, imagen_data[1].name, imagen_data[1].attackName, exec_ID)
    fig.savefig(File_name)

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
        resultado = (abs(original_img[num].data-adv_img.data))*100000
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
    File_name = 'Difference_between_orig_adv_method-%s/Difference_image-%s_attack_method-%s%s.jpg' % (adv_img.attackName, original_img[num].name, adv_img.attackName, exec_ID)
    plt.savefig(File_name)

def isValidExample(num, original_img, adversarial_img, n_iter, epsilon):
    saveSuccesfulExample = False
    total_img = len(original_img)
    # Si la red no ha acertado en la predicción de la imagen original, no se guarda la imagen
    if original_img[num].predictionId == original_img[num].id :
        for j in range(0, len(epsilon)) :
            index = num + total_img * n_iter * len(epsilon) + total_img * j
            # Si el adversario ha conseguido confundir a la red, se guarda la imagen
            if adversarial_img[index].predictionId != adversarial_img[index].id:
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