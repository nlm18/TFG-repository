import os
import errno
import shutil
import csv

# ------------------------ Funciones auxiliares ---------------------------------
def obtainNames(csvFile):
    imageNames=[]
    with open(csvFile, 'r') as archivo :
        lector = csv.reader(archivo)
        for fila in lector :
            imageNames.append(fila[0])
    return imageNames


# -------------------------------------------------------------------------------

NetworkModelName = 'Xception'
DATA_PATH = "C:/Users/User/TFG-repository/webcam_gradcam/ImageNetWebcam/luz_validos/waterBottle_%s_5000luz/selectedOrigImages/" % (
    NetworkModelName)
csvFile = "images_to_recalculate_%s.csv" % (NetworkModelName)
files_orig_names = obtainNames(csvFile)

try :
    os.mkdir(DATA_PATH+'selected')
except OSError as e :
    if e.errno != errno.EEXIST :
        raise

for i in range(0, len(files_orig_names)):
    shutil.copy( DATA_PATH + files_orig_names[i], DATA_PATH + "selected/" + files_orig_names[i])
