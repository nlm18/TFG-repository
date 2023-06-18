from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg16
from keras.applications.xception import Xception, preprocess_input as preprocess_xception
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_inceptionv3
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as preprocess_inceptionresnetv2
from keras.applications.vgg16 import decode_predictions as decode_vgg16
from keras.applications.xception import decode_predictions as decode_xception
from keras.applications.inception_v3 import decode_predictions as decode_inceptionv3
from keras.applications.inception_resnet_v2 import decode_predictions as decode_inceptionresnetv2
from keras.applications.efficientnet import EfficientNetB0, decode_predictions as decode_efficientNetB0

import numpy as np
import cv2
import errno
import gradCamInterface

NETWORK_NAME = "EfficientNetB0"
OBJECT = "water_bottle_efficientNetB0"

import os
import keras
from keras.layers import Input

def example():

    imagePath=os.path.join('unifome.jpg')

    # load the original image via OpenCV so we can draw on it and display
    # it to our screen later

    #orig = cv2.imread(args["image"])
    orig = cv2.imread(imagePath)

    # load the input image using the Keras helper utility while ensuring
    # that the image is resized to 224x224 pixels, the required input
    # dimensions for the network -- then convert the PIL image to a
    # NumPy array
    print("[INFO] loading and preprocessing image...")
    #image = image_utils.load_img(args["image"], target_size=(224, 224))
    image = image_utils.load_img(imagePath, target_size=(299,299))
    image = image_utils.img_to_array(image)

    # our image is now represented by a NumPy array of shape (224, 224, 3),
    # assuming TensorFlow "channels last" ordering of course, but we need
    # to expand the dimensions to be (1, 3, 224, 224) so we can pass it
    # through the network -- we'll also preprocess the image by subtracting
    # the mean RGB pixel intensity from the ImageNet dataset
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # load the VGG16 network pre-trained on the ImageNet dataset
    print("[INFO] loading network...")
    #model = VGG16(weights="imagenet")
    model = Xception(weights="imagenet")

    # classify the image
    print("[INFO] classifying image...")
    preds = model.predict(image)
    P = decode_predictions(preds)

    # loop over the predictions and display the rank-5 predictions +
    # probabilities to our terminal
    for (i, (imagenetID, label, prob)) in enumerate(P[0]):
        print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

    # load the image via OpenCV, draw the top prediction on the image,
    # and display the image to our screen
    #orig = cv2.imread(args["image"])
    orig = cv2.imread(imagePath)
    (imagenetID, label, prob) = P[0][0]
    cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Classification", orig)
    cv2.waitKey(0)

def classifyImage(frame,network,network_name):
    frameLabeled=np.copy(frame)
    if network_name == "vgg16" or network_name == "EfficientNetB0" :
        img_shape = (224,224)
    else:
        img_shape = (299,299)
    frame_resized=cv2.resize(frame,img_shape)
    #frame_resized = frame_resized.reshape(1,224,224,3)
    #image = image_utils.img_to_array(frame_resized)
    image = np.expand_dims(frame_resized, axis=0)
    if network_name == "xception":
        image = preprocess_xception(image)
        preds = network.predict(image)#.reshape(1, 299, 299, 3))
        P = decode_xception(preds)
        last_conv_layer_name = "conv2d_3"
    elif network_name == "inceptionv3":
        image = preprocess_inceptionv3(image)
        preds = network.predict(image)
        P = decode_inceptionv3(preds)
        last_conv_layer_name = "conv2d_97"
    elif network_name == "inceptionresnetv2":
        image = preprocess_inceptionresnetv2(image)
        preds = network.predict(image)
        P = decode_inceptionresnetv2(preds)
        last_conv_layer_name = "conv_7b"
    elif network_name == "vgg16":
        image = preprocess_vgg16(image)
        preds = network.predict(image)
        P = decode_vgg16(preds)
        last_conv_layer_name = "block5_conv3"
    elif network_name == "EfficientNetB0":
        preds = network.predict(image)
        P = decode_efficientNetB0(preds)
        last_conv_layer_name = "top_activation"
    (imagenetID, label, prob) = P[0][0]
    #gradCam = executeGradCam(image, network, last_conv_layer_name)
    #frame_resized = cv2.resize(gradCam, (640,480))
    cv2.putText(frameLabeled, "Label: {}, {:.2f}%".format(label, prob * 100),#poner frameLabeled si no se quiere mostrar el gradcam
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frameLabeled

def webcamShow():

    print("[INFO] loading network...")
    model = EfficientNetB0(weights="imagenet", include_top=True, classes=1000, input_shape=(224, 224, 3))
        #Xception(include_top=True, weights="imagenet") #, input_tensor=Input(shape=(299, 299, 3))

    cv2.namedWindow("webcam")
    vc = cv2.VideoCapture(0)

    N_FRAMES = 100

    while True:
        # Capture the frame
        next, frame = vc.read()
        #cv2.imshow("webcam", frame)
        # Classify and show the image
        frameDetection=classifyImage(frame,model,NETWORK_NAME)
        cv2.imshow("webcam", frameDetection)
        # When a key is pressed, start recording
        if cv2.waitKey(50) >= 0:#Enter
            break

    for i in range(N_FRAMES):
        # Capture the frame
        next, frame = vc.read()
        # cv2.imshow("webcam", frame)
        # Classify and show the image
        frameDetection = classifyImage(frame, model,NETWORK_NAME)
        #cv2.imshow("webcam", frameDetection)
        # Crea los directorios
        createDirs(OBJECT)
        # Save the frames
        list = os.listdir("ImageNetWebcam/%s/frames_%s/" % (OBJECT, "raw"))
        num = len(list)
        path="ImageNetWebcam/%s/frames_%s/imageFrame_%s.png"
        cv2.imwrite(path % (OBJECT, "raw", num), frame)
        cv2.imwrite(path % (OBJECT, "detected", num), frameDetection)
        num+=1

    vc.release()
    cv2.destroyAllWindows()

def networksTest():

    # FIIIIINNNNNNNNNNNNNNIIIIIIIIIIIIIIIIIIIISSSSSSSSSSSSSSSHHHHHHHHHHHHHHHHH

    networkModels={"xception":Xception(include_top=True, weights="imagenet", input_tensor=Input(shape=(299, 299, 3))),
                   "inceptionv3": InceptionV3(include_top=True, weights="imagenet", input_tensor=Input(shape=(299, 299, 3))),
                   "inceptionresnetv2": InceptionResNetV2(include_top=True, weights="imagenet",input_tensor=Input(shape=(299, 299, 3))),
                   "vgg16": VGG16(include_top=True, weights="imagenet",input_tensor=Input(shape=(224, 224, 3)))}

    for network in ["xception","inceptionv3","inceptionresnetv2","vgg16"]:
        keras.backend.clear_session()
        model=networkModels[network]

def executeGradCam(img_array, model, last_conv_layer_name):
    # https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network

    # Generate class activation heatmap
    heatmap = gradCamInterface.make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    #img = keras.preprocessing.image.array_to_img(img_array)
    # Display heatmap
    #Ya esta entre 0-255 img_255.append(list_img[ind] * 255)
    gradCam_img = gradCamInterface.display_gradcam(img_array[0], heatmap, return_image=False,alpha=0.05)

    return gradCam_img

def createDirs(OBJECT):
    try :
        os.makedirs('ImageNetWebcam/%s/frames_raw' % (OBJECT))
    except OSError as e :
        if e.errno != errno.EEXIST :
            raise
    try :
        os.makedirs('ImageNetWebcam/%s/frames_detected' % (OBJECT))
    except OSError as e :
        if e.errno != errno.EEXIST :
            raise

def rename():
    path_org="ImageNetWebcam/water_bottle_efficientNet_soapDispenser/frames_%s/"
    list_origin_raw = os.listdir( path_org % ("raw"))
    list_origin_detected = os.listdir(path_org % ("detected"))
    list_destination = os.listdir("ImageNetWebcam/%s/frames_%s" % (OBJECT, "raw"))
    num = len(list_destination)
    #nombre_nuevo = "ImageNetWebcam/%s/frames_%s/imageFrame_%s.png"
    nombre_nuevo = path_org+"imageFrame_%s.png"
    for i in range(len(list_origin_raw)):
        os.rename(path_org%("raw")+list_origin_raw[i], nombre_nuevo % ("raw", num))
        os.rename(path_org%("detected")+list_origin_detected[i], nombre_nuevo % ("detected", num))
        num+=1

if __name__ == '__main__':
    #example()
#    webcamShow()
    rename()
    #networksTest()
