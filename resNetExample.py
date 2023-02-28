import tensorflow as tf

tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10 as cifar10
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
#las x son las imagenes y las 'y' los label de las imagenes 'x'
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1 :]
num_classes=10


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Print Training and Test Samples
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

model = Sequential()
pretrained_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    classes=num_classes
)
for layer in pretrained_model.layers:
    layer.trainable = False
model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
if x_train.min() < x_test.min():
    min_pixel_value = x_train.min()
else:
    min_pixel_value = x_test.min()
if x_train.max() > x_test.max():
    max_pixel_value = x_train.max()
else:
    max_pixel_value = x_test.max()

classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)
epochs = 3
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=epochs)

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

'''w =tf.shape(x_train)
model.fit(tf.reshape(x_train,(w[0],32,32,3)), y_train, epochs=5, steps_per_epoch=100 )
#model.fit(x_train, y_train, epochs=5)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


history = model.fit(x_train, y_train, validation_split=0.3, epochs=epochs, batch_size= 32, shuffle=True)#validation_data=(x,y)
#lista de datos del historial
#print(history.history.keys())

scores = model.evaluate(x_test, y_test, verbose = 1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

fig1= plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.grid()
plt.title('ResNet model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

fig2= plt.gcf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('ResNet model loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[y_test[i][0]])
plt.show()'''