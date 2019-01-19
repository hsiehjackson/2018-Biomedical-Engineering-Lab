import numpy as np
import tensorflow as tf
import os
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D, Concatenate
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from scipy.fftpack import fft, ifft

from scipy import signal
import pywt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
PATH="/home/mlpjb04/bioproject/deepsleepnet/data/"


def get_model():

	input1 = Input(shape=(3000,1))
	cnn1 = Conv1D(64, 50,strides=6, activation='relu', input_shape=(3000,1),padding='same')(input1)
	cnn1 = BatchNormalization()(cnn1)
	cnn1 = MaxPooling1D(pool_size=8,strides=8)(cnn1)
	cnn1 = Dropout(0.5)(cnn1)
	cnn1 = Conv1D(128,8,strides=1, activation='relu',padding='same')(cnn1)
	cnn1 = BatchNormalization()(cnn1)
	cnn1 = Conv1D(128,8,strides=1, activation='relu',padding='same')(cnn1)
	cnn1 = BatchNormalization()(cnn1)
	cnn1 = Conv1D(128,8,strides=1, activation='relu',padding='same')(cnn1)
	cnn1 = BatchNormalization()(cnn1)
	cnn1 = MaxPooling1D(pool_size=4,strides=4)(cnn1)
	cnn1 = Flatten()(cnn1)

	cnn2 = Conv1D(64, 400,strides=50, activation='relu', input_shape=(3000,1),padding='same')(input1)
	cnn2 = BatchNormalization()(cnn2)
	cnn2 = MaxPooling1D(pool_size=4,strides=4)(cnn2)
	cnn2 = Dropout(0.5)(cnn2)
	cnn2 = Conv1D(128,6,strides=1, activation='relu',padding='same')(cnn2)
	cnn2 = BatchNormalization()(cnn2)
	cnn2 = Conv1D(128,6,strides=1, activation='relu',padding='same')(cnn2)
	cnn2 = BatchNormalization()(cnn2)
	cnn2 = Conv1D(128,6,strides=1, activation='relu',padding='same')(cnn2)
	cnn2 = BatchNormalization()(cnn2)
	cnn2 = MaxPooling1D(pool_size=2,strides=2)(cnn2)
	cnn2 = Flatten()(cnn2)

	concat = Concatenate()([cnn1,cnn2])
	concat = Dropout(0.5)(concat)

	out = Dense(units=1024, activation='relu')(concat)
	out = BatchNormalization()(out)
	out = Dropout(0.5)(out)
	out = Dense(units=5, activation='softmax')(out)
	out = BatchNormalization()(out)

	model = Model(inputs=input1, outputs=out)

	return model


def main():
	get_session(0.5)
	train_x = np.load(PATH+"data.npy")#.reshape(42308, 3000, 1)
	train_y = np.load(PATH+'label.npy')#.reshape(42308, 3000, 1)
	print(train_x.shape)
	print(train_y.shape)

	#train_x = np.expand_dims(train_x, axis=2)
	#train_x2 = np.expand_dims(train_x2, axis=2)
	
	for i, label in enumerate(train_y):
		if label == int(4):
			train_y[i] = 3
		elif label == int(5):
			train_y[i] = 4

	train_y = np_utils.to_categorical(train_y,5)
	

	model = get_model()
	opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=10) #initial:lr=0.001
	model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()


	#checkpoint = ModelCheckpoint('save_models/model-{epoch:03d}.h5', verbose=1, monitor='val_acc',save_best_only=True, mode='max')
	model.fit(train_x, train_y, epochs=100, shuffle=True, batch_size=100, verbose=1, validation_split=0.05)


def get_session(gpu_fraction):
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
	return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  

if __name__ == '__main__':
	get_session(0.5)
	main()


