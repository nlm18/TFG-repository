import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import numpy as np
from templateModel import createModel, CnnType

# Step 1: Load the MNIST dataset, este data set contiene imagenes de dígitos del 0 al 9 manuscritos.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Step 2: Create the model, el modelo secuencial significa que se va haciendo stacking de capas
model = createModel(CnnType.SEQ)

# Step 3: Configure and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
