# Import Keras modules and its important APIs
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K


# Computed depth of
def calculate_depth(version, n):
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2
    # Model name, depth and version
    model_type = 'ResNet % dv % d' % (depth, version)
    return depth


# Setting LR for different number of Epochs
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# Basic ResNet Building Block
def resnet_layer(inputs,
				num_filters=16,
				kernel_size=3,
				strides=1,
				activation='relu',
				batch_normalization=True,
				conv_first=False):
	conv=Conv2D(num_filters,
				kernel_size=kernel_size,
				strides=strides,
				padding='same',
				kernel_initializer='he_normal',
				kernel_regularizer=l2(1e-4))

	x=inputs
	if conv_first:
		x = conv(x)
		if batch_normalization:
			x = BatchNormalization()(x)
		if activation is not None:
			x = Activation(activation)(x)
	else:
		if batch_normalization:
			x = BatchNormalization()(x)
		if activation is not None:
			x = Activation(activation)(x)
		x = conv(x)
	return x

#ResNet V1 architecture
def resnet_v1(input_shape, depth, num_classes=10):

	if (depth - 2) % 6 != 0:
		raise ValueError('depth should be 6n + 2 (eg 20, 32, 44 in [a])')
	# Start model definition.
	num_filters = 16
	num_res_blocks = int((depth - 2) / 6)

	inputs = Input(shape=input_shape)
	x = resnet_layer(inputs=inputs)
	# Instantiate the stack of residual units
	for stack in range(3):
		for res_block in range(num_res_blocks):
			strides = 1
			if stack & gt == 0 and res_block == 0: # first layer but not first stack
				strides = 2 # downsample
			y = resnet_layer(inputs=x,
							num_filters=num_filters,
							strides=strides)
			y = resnet_layer(inputs=y,
							num_filters=num_filters,
							activation=None)
			if stack & gt == 0 and res_block == 0: # first layer but not first stack
				# linear projection residual shortcut connection to match
				# changed dims
				x = resnet_layer(inputs=x,
								num_filters=num_filters,
								kernel_size=1,
								strides=strides,
								activation=None,
								batch_normalization=False)
			x = keras.layers.add([x, y])
			x = Activation('relu')(x)
		num_filters *= 2

	# Add classifier on top.
	# v1 does not use BN after last shortcut connection-ReLU
	x = AveragePooling2D(pool_size=8)(x)
	y = Flatten()(x)
	outputs = Dense(num_classes,
					activation='softmax',
					kernel_initializer='he_normal')(y)

	# Instantiate model.
	model = Model(inputs=inputs, outputs=outputs)
	return model

# ResNet V2 architecture
def resnet_v2(input_shape, depth, num_classes=10):
	if (depth - 2) % 9 != 0:
		raise ValueError('depth should be 9n + 2 (eg 56 or 110 in [b])')
	# Start model definition.
	num_filters_in = 16
	num_res_blocks = int((depth - 2) / 9)

	inputs = Input(shape=input_shape)
	# v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
	x = resnet_layer(inputs=inputs,
					num_filters=num_filters_in,
					conv_first=True)

	# Instantiate the stack of residual units
	for stage in range(3):
		for res_block in range(num_res_blocks):
			activation = 'relu'
			batch_normalization = True
			strides = 1
			if stage == 0:
				num_filters_out = num_filters_in * 4
				if res_block == 0: # first layer and first stage
					activation = None
					batch_normalization = False
			else:
				num_filters_out = num_filters_in * 2
				if res_block == 0: # first layer but not first stage
					strides = 2 # downsample

			# bottleneck residual unit
			y = resnet_layer(inputs=x,
							num_filters=num_filters_in,
							kernel_size=1,
							strides=strides,
							activation=activation,
							batch_normalization=batch_normalization,
							conv_first=False)
			y = resnet_layer(inputs=y,
							num_filters=num_filters_in,
							conv_first=False)
			y = resnet_layer(inputs=y,
							num_filters=num_filters_out,
							kernel_size=1,
							conv_first=False)
			if res_block == 0:
				# linear projection residual shortcut connection to match
				# changed dims
				x = resnet_layer(inputs=x,
								num_filters=num_filters_out,
								kernel_size=1,
								strides=strides,
								activation=None,
								batch_normalization=False)
			x = keras.layers.add([x, y])

		num_filters_in = num_filters_out

	# Add classifier on top.
	# v2 has BN-ReLU before Pooling
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = AveragePooling2D(pool_size=8)(x)
	y = Flatten()(x)
	outputs = Dense(num_classes,
					activation='softmax',
					kernel_initializer='he_normal')(y)

	# Instantiate model.
	model = Model(inputs=inputs, outputs=outputs)
	return model
