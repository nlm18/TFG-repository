from keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg16, decode_predictions as decode_vgg16
from keras.applications.xception import Xception, preprocess_input as preprocess_xception, decode_predictions as decode_xception
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_inceptionv3, decode_predictions as decode_inceptionv3
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as preprocess_inceptionresnetv2, decode_predictions as decode_inceptionresnetv2
from keras.applications.efficientnet import EfficientNetB0, decode_predictions as decode_efficientNetB0
from keras.layers import Input
import os
import numpy as np
import cv2
import errno

NETWORK_NAME = "Xception"
OBJECT = "FountainPen_%s" % (NETWORK_NAME)
realID='n03388183'

def classifyImage(frame, network, network_name):
    frameLabeled=np.copy(frame)
    advNatural = False
    if network_name == "VGG16" or network_name == "EfficientNetB0":
        img_shape = (224,224)
    else:
        img_shape = (299,299)
    frame_resized=cv2.resize(frame,img_shape)
    image = np.expand_dims(frame_resized, axis=0)
    if network_name == "EfficientNetB0":
        preds = network.predict(image)
        P = decode_efficientNetB0(preds)
    elif network_name == 'InceptionResNetV2':
        image = preprocess_inceptionresnetv2(image)
        preds = network.predict(image)
        P = decode_inceptionresnetv2(preds)
    elif network_name == "InceptionV3":
        image = preprocess_inceptionv3(image)
        preds = network.predict(image)
        P = decode_inceptionv3(preds)
    elif network_name == 'VGG16':
        image = preprocess_vgg16(image)
        preds = network.predict(image)
        P = decode_vgg16(preds)
    elif network_name == "Xception":
        image = preprocess_xception(image)
        preds = network.predict(image)
        P = decode_xception(preds)
    (imagenetID, label, prob) = P[0][0]

    if imagenetID != realID:
        advNatural=True
    cv2.putText(frameLabeled, "Label: {}, {:.2f}%".format(label, prob * 100),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frameLabeled, advNatural

def getNetworkModel(NetworkModelName):
    if NetworkModelName == 'EfficientNetB0':
        return EfficientNetB0(weights="imagenet", include_top=True, classes=1000, input_shape=(224, 224, 3))
    elif NetworkModelName == 'Xception':
        return Xception(include_top=True, weights="imagenet", input_tensor=Input(shape=(299, 299, 3)))
    elif NetworkModelName == 'InceptionV3':
        return InceptionV3(include_top=True, weights="imagenet", input_tensor=Input(shape=(299, 299, 3)))
    elif NetworkModelName == 'InceptionResNetV2':
        return InceptionResNetV2(include_top=True, weights="imagenet", input_tensor=Input(shape=(299, 299, 3)))
    elif NetworkModelName == 'VGG16':
        return VGG16(include_top=True, weights="imagenet", input_tensor=Input(shape=(224, 224, 3)))

def webcamShow():

    print("[INFO] loading network...")
    model = getNetworkModel(NETWORK_NAME)

    cv2.namedWindow("webcam")
    vc = cv2.VideoCapture(0)

    N_FRAMES = 5000

    while True:
        # Capture the frame
        next, frame = vc.read()

        # Classify and show the image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameDetection,x = classifyImage(frame_rgb, model, NETWORK_NAME)
        frame_bgr = cv2.cvtColor(frameDetection, cv2.COLOR_RGB2BGR)
        cv2.imshow("webcam", frame_bgr)
        # When a key is pressed, start recording
        if cv2.waitKey(50) >= 0:#Enter
            break
    num_AdvNaturales = 0
    # Create Directories
    createDirs(OBJECT)
    list = os.listdir("ImageNetWebcam/%s/frames_%s/" % (OBJECT, "raw"))
    num = len(list)
    for i in range(N_FRAMES):
        # Capture the frame
        next, frame = vc.read()

        # Classify and show the image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameDetection, advNatural = classifyImage(frame_rgb, model, NETWORK_NAME)
        frame_bgr = cv2.cvtColor(frameDetection, cv2.COLOR_RGB2BGR)

        # Save the frames
        path="ImageNetWebcam/%s/frames_%s/imageFrame_%s.png"
        num+=1
        if advNatural:
            num_AdvNaturales += 1
            cv2.imwrite(path % (OBJECT, "naturalAdvDetected", num), frame_bgr)
            cv2.imwrite(path % (OBJECT, "naturalAdv", num), frame)
        else:
            cv2.imwrite(path % (OBJECT, "detected", num), frame_bgr)
            cv2.imwrite(path % (OBJECT, "raw", num), frame)


    vc.release()
    cv2.destroyAllWindows()
    print("- Percentage of natural adversarial images: {}%".format(num_AdvNaturales / N_FRAMES * 100))


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
    try :
        os.makedirs('ImageNetWebcam/%s/frames_naturalAdvDetected' % (OBJECT))
    except OSError as e :
        if e.errno != errno.EEXIST :
            raise
    try :
        os.makedirs('ImageNetWebcam/%s/frames_naturalAdv' % (OBJECT))
    except OSError as e :
        if e.errno != errno.EEXIST :
            raise

if __name__ == '__main__':
    webcamShow()
