# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class vgg16net:
	@staticmethod
	def build(width, height, depth, classes, finalAct="softmax"):

		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1


		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# CONV => RELU => POOL
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
		model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
		model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
		model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
		model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
		model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
		model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
		model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
		model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
		model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
		model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
		model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
		model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
		model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
		model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
		model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
		model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

		# softmax classifier
		model.add(Flatten())
		model.add(Dense(units=4096,activation="relu"))
		model.add(Dense(units=4096,activation="relu"))
		model.add(Dense(classes))
		model.add(Activation(finalAct))

		return model