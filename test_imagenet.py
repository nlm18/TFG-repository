from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
#from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg16
from keras.applications.xception import Xception, preprocess_input as preprocess_xception
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_inceptionv3
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as preprocess_inceptionresnetv2
import numpy as np
import cv2

NETWORK_NAME = "xception"

import os
import keras
from keras.layers import Input

def example():

    imagePath=os.path.join('images','brown_bear.png')

    # load the original image via OpenCV so we can draw on it and display
    # it to our screen later

    #orig = cv2.imread(args["image"])
    orig = cv2.imread(imagePath)

    # load the input image using the Keras helper utility while ensuring
    # that the image is resized to 224x224 pxiels, the required input
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
    image = image_utils.img_to_array(frame)
    image = np.expand_dims(image, axis=0)
    if network_name == "xception":
        image = preprocess_xception(image)
    elif network_name == "inceptionv3":
        image = preprocess_inceptionv3(image)
    elif network_name == "inceptionresnetv2":
        image = preprocess_inceptionresnetv2(image)
    elif network_name == "vgg16":
        image = preprocess_vgg16(image)
    preds = network.predict(image)
    P = decode_predictions(preds)
    (imagenetID, label, prob) = P[0][0]
    cv2.putText(frameLabeled, "Label: {}, {:.2f}%".format(label, prob * 100),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frameLabeled

def webcamShow():

    print("[INFO] loading network...")
    model = Xception(weights="imagenet")

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
        if cv2.waitKey(50) >= 0:
            break

    for i in range(N_FRAMES):
        # Capture the frame
        next, frame = vc.read()
        # cv2.imshow("webcam", frame)
        # Classify and show the image
        frameDetection = classifyImage(frame, model,NETWORK_NAME)
        #cv2.imshow("webcam", frameDetection)
        # Save the frames
        cv2.imwrite(os.path.join('ImageNetWebcam','frames_raw','imageFrame_{:02d}'.format(i)+'.png'),frame)
        cv2.imwrite(os.path.join('ImageNetWebcam','frames_detected', 'imageFrame_{:02d}'.format(i) + '.png'), frameDetection)

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
        model=networkModels(network)



if __name__ == '__main__':
    #example()
    #webcamShow()
    networksTest()
