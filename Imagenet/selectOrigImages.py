import os
import errno
import shutil

# ------------------------ Funciones auxiliares ---------------------------------
def obtainImageNumber(imageName):
    num = imageName.replace('.png', '')
    num = num.replace('imageFrame_', '')
    return int(num)

def findCloserImage(num_ref, superior=True):
    j, aux, result = 0, 0, 0
    imgFinded = False
    while imgFinded == False :
        result = aux
        aux = original_sorted_list[j]
        if aux > num_ref and superior != True : #la imagen es la mas cercana por debajo
            if result != 0: #La numeracion de las imagenes empieza en uno
                imgFinded = True
            else:
                superior = True
        elif aux > num_ref and superior:
            imgFinded=True
            result = aux
        j+=1
    if j != 0 and result  == 0:
        print('No se ha encontrado una imagen original cercana, comprueba la lista de fotos de la carpeta frames_raw')
    return result

def sortImgList(files_advNatural_names, files_orig_names):
    nat = []
    orig = []
    for i in range(0,len(files_advNatural_names)):
        nat.append(obtainImageNumber(files_advNatural_names[i]))
    for i in range(0,len(files_orig_names)):
        orig.append(obtainImageNumber(files_orig_names[i]))
    return sorted(nat), sorted(orig)

# -------------------------------------------------------------------------------
#Este script pretende sacar las imagenes originales mas cercanas a las imagenes adversarias naturales, no se filtra la
#cantidad de imagenes adversarias naturales se cogen, si se quiere hacer puede generarse un vector aleatorio con usando
#generateRandomVector de auxiliarFunction.py
DATA_PATH = "C:/Users/User/TFG-repository/webcam_gradcam/ImageNetWebcam/luz_validos/waterBottle_Xception_5000luz/"
files_advNatural_names = os.listdir(DATA_PATH+"frames_naturalAdv")
#files_orig_names = os.listdir(DATA_PATH+"frames_raw")
DISCO_DURO_PATH = "D:/TFG_VISILAB_FOTOS/luz_validos/waterBottle_Xception_5000luz/frames_raw/"
files_orig_names = os.listdir(DISCO_DURO_PATH)

natural_sorted_list, original_sorted_list = sortImgList(files_advNatural_names, files_orig_names)

try :
    os.mkdir(DATA_PATH+'selectedOrigImages')
except OSError as e :
    if e.errno != errno.EEXIST :
        raise

selected_orig_names = []
for i in range(0, len(files_advNatural_names)):
    num_ref = natural_sorted_list[i]
    inf = findCloserImage(num_ref, superior=False)
    sup = findCloserImage(num_ref, superior=True)
    if abs(num_ref-inf) > abs(num_ref-sup): #La imagen mas cercana es la superior
        selected_orig_names.append('imageFrame_%s.png' % (sup))
    else:
        selected_orig_names.append('imageFrame_%s.png' % (inf))

for i in range(0,len(selected_orig_names)):
    #shutil.copy( DATA_PATH + 'frames_raw/' + selected_orig_names[i], DATA_PATH + "selectedOrigImages/" + selected_orig_names[i])
    shutil.copy(DISCO_DURO_PATH + selected_orig_names[i],
                DATA_PATH + "selectedOrigImages/" + selected_orig_names[i])


