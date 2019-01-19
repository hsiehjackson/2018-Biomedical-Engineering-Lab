import numpy as np
import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
from keras.utils import np_utils
from scipy.fftpack import fft, ifft

from scipy import signal
import pywt
from pyhht.emd import EMD
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt



def read_csv_data(filepath):
	csv_path_list = os.listdir(filepath)
	csv_path_list.sort()
	data = []
	train_x1 = np.empty([0,3000])
	train_x2 = np.empty([0,3000])
	train_y = []
	for i, file in enumerate(csv_path_list):
		f = open(os.path.join(filepath, file),'r',encoding='utf8')
		next(f)
		for row in f :
			tmp = row.split(',')
			tmp_x1 = []
			tmp_x2 = []
			if int(tmp[0])>=10 and int(tmp[0])<len(tmp)-10:
				train_y.append(int(tmp[1]))
				for i in range(2,len(tmp)-1):
					if len(tmp_x1)<3000:
						tmp_x1.append(float(tmp[i]))
					else:
						tmp_x2.append(float(tmp[i]))

				tmp_x1 = np.array(tmp_x1).reshape(-1,1)
				tmp_x2 = np.array(tmp_x2).reshape(-1,1)
				train_x1 = np.concatenate((train_x1,np.array(tmp_x1).T),axis=0)
				train_x2 = np.concatenate((train_x2,np.array(tmp_x2).T),axis=0)
				#xxx = np.concatenate((tmp_xft1,tmp_xft2),axis=1)
				#train_x.append(xxx.T)

			if int(tmp[0])>969: break
		print(np.array(train_x1).shape)
		print(np.array(train_x2).shape)
		f.close()

	train_x1 = np.array(train_x1)
	train_x2 = np.array(train_x2)
	train_y = np.array(train_y)
	return train_x1,train_x2, train_y

def get_model():
	model = Sequential()
	model.add(Conv1D(300, 50, strides=10,activation='relu', input_shape=(3000,1),padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling1D(pool_size=10, stride=5))
	model.add(Conv1D(100, 30, strides=1,activation='relu',padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling1D(pool_size=5, stride=1))
	model.add(Conv1D(100, 20, strides=1,activation='relu',padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling1D(pool_size=5, stride=1))
	"""
	model.add(Conv2D(100, (3,5), activation='relu',padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(1,5), stride=5))
	
	model.add(Conv2D(128, , activation='relu',padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(1,5)))
	model.add(Conv2D(64,  3, activation='relu',padding='same'))
	model.add(BatchNormalization())
	"""
	model.add(Flatten())
	#model.add(Dense(units=1024, activation='relu'))
	#model.add(BatchNormalization())
	#model.add(Dropout(0.3))

	model.add(Dense(units=375, activation='relu'))
	model.add(BatchNormalization())
	#model.add(Dropout(0.3))
	
	model.add(Dense(units=6, activation='softmax'))
	model.add(BatchNormalization())
	#model.add(Dropout(0.3))

	return model

def DFT(data):
	f,t,z=signal.stft(data,fs=100,window='hamming',nperseg=600,noverlap=0,nfft=999)
	z = np.reshape(z.T,(-1,1))
	z = abs(z).flatten()
	return z

def WT(data):
	sig = pywt.wavedec(data, 'db4', mode='symmetric', level=4, axis=-1)
	x = np.arange(0,len(sig[0]))
	y = sig[0]
	f = interp1d(x,y,kind='cubic')
	xnew = np.arange(0,len(sig[0])-1,(len(sig[0])-1)/3000)
	ynew = f(xnew)
	return ynew

def EMD_anly(data):
	decomposer = EMD(data)
	imfs = decomposer.decompose()
	return imfs[4].flatten(), imfs[5].flatten()

def main():
	#get_session(0.4)
	train_data = np.load('train_x1.npy')
	train_y = np.load('train_y.npy')	

	
	train_x = []
	for i in range(len(train_data)):
		w = WT(train_data[i]).flatten()	
		#d = DFT(train_data[i]).flatten()
		train_x.append([w])
		#train_x.append([train_data[i],d,w])

	train_x = np.array(train_x)
	print(train_x.shape)
	
	train_x = train_x.reshape(-1,3000)
	mean = np.mean(train_x, axis=0)
	std = np.std(train_x, axis=0)
	train_x = (train_x-mean) / std
	train_x = train_x.reshape(-1,3000,1)

	train_y = np_utils.to_categorical(train_y,6)
	print(train_x.shape)


	model = get_model()
	opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #initial:lr=0.001
	#opt = RMSprop(lr=0.0002, rho=0.9, epsilon=None, decay=0.0)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()


	#checkpoint = ModelCheckpoint('save_models/model-{epoch:03d}.h5', verbose=1, monitor='val_acc',save_best_only=True, mode='max')
	model.fit(train_x, train_y, epochs=100, shuffle=True, batch_size=256, verbose=1, validation_split=0.05)


def get_session(gpu_fraction):
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
	return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  

if __name__ == '__main__':
	main()


