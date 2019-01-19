import numpy as np
import os
from keras.utils import np_utils
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch import optim
from torch.utils import data as Data
import torchvision
import time
import argparse

PATH="/home/mlpjb04/bioproject/deepsleepnet/data/"
def to_var(x):
	x = Variable(x)
	if torch.cuda.is_available():
		x = x.cuda()
	return x

def timeSince(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

class CNN_Classifier(nn.Module):
	def __init__(self, ngf=64):
		super(CNN_Classifier, self).__init__()
		

		self.cnn1 = nn.Sequential(
				nn.Conv1d(1, 64, kernel_size=50, stride=6, padding=1, bias=False),
				nn.BatchNorm1d(64),
				nn.LeakyReLU(0.2, inplace=True),
				nn.MaxPool1d(kernel_size=8, stride=8),
				nn.Dropout(0.5),

				nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=1, bias=False),
				nn.BatchNorm1d(128),
				nn.LeakyReLU(0.2, inplace=True),

				nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=1, bias=False),
				nn.BatchNorm1d(128),
				nn.LeakyReLU(0.2, inplace=True),

				nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=1, bias=False),
				nn.BatchNorm1d(128),
				nn.LeakyReLU(0.2, inplace=True),

				nn.MaxPool1d(kernel_size=4, stride=4)
			)
		self.cnn2 = nn.Sequential(
				nn.Conv1d(1, 64, kernel_size=400, stride=50, padding=1, bias=False),
				nn.BatchNorm1d(64),
				nn.LeakyReLU(0.2, inplace=True),
				nn.MaxPool1d(kernel_size=4, stride=4),
				nn.Dropout(0.5),
				
				nn.Conv1d(64, 128, kernel_size=6, stride=1, padding=1, bias=False),
				nn.BatchNorm1d(128),
				nn.LeakyReLU(0.2, inplace=True),

				nn.Conv1d(128, 128, kernel_size=6, stride=1, padding=1, bias=False),
				nn.BatchNorm1d(128),
				nn.LeakyReLU(0.2, inplace=True),

				nn.Conv1d(128, 128, kernel_size=6, stride=1, padding=1, bias=False),
				nn.BatchNorm1d(128),
				nn.LeakyReLU(0.2, inplace=True),

				nn.MaxPool1d(kernel_size=2, stride=2)
			)

		self.dnn = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(11121, 1024),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(1024,5),
			nn.Softmax(1)
		)

	def forward(self, x):
		out1 = self.cnn1(x).view(-1,)
		out2 = self.cnn2(x).view(-1,)
		dnn_in = torch.cat((out1,out2), 1)
		print(dnn_in.shape)
		output = self.dnn(dnn_in)
		return output

def train(n_epochs, train_loader, x_val, y_val):
	model = CNN_Classifier()
	optimizer = torch.optim.Adam(model.parameters(),lr=0.0001, betas=(0.9, 0.999))
	loss_function = nn.CrossEntropyLoss()
	x_val = to_var(x_val)
	y_val = to_var(y_val)
	if torch.cuda.is_available():
		model.cuda()

	train_loss_list, train_acc_list = [],[]
	val_loss_list, val_acc_list = [],[]
	best_accuracy = 0.0
	print("1")
	for epoch in range(n_epochs):
		start = time.time()
		CE_loss = 0.0
		Train_Acc = 0.0
		for batch_idx, (x,y) in enumerate(train_loader):
			batch_size = x.size(0)
			x = to_var(x)
			y = to_var(y)
			print("2")
			print(x.shape)

			model.train()
			optimizer.zero_grad()

			output = model(x)
			print("3")
			err = loss_function(output, y)
			#print (output.size())	#[64,11]
			#print (y.size())		#[64]
			print("4")
			err.backward()
			optimizer.step()
			
			CE_loss += err.item()
			output_label = torch.argmax(output,1).cpu()
			Acc = np.mean((output_label == y.cpu()).numpy())
			Train_Acc += Acc



			if batch_idx % 2 == 0:		
				print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]| CE_Loss: {:.6f} | Acc: {:.6f} | Time: {}  '.format(
					epoch+1, (batch_idx+1) * len(x), len(train_loader.dataset),
					100. * batch_idx * len(x)/ len(train_loader.dataset),
					CE_loss, Acc,
					timeSince(start, (batch_idx+1)*len(x)/ len(train_loader.dataset))),end='')
		
		
		#Validation
		with torch.no_grad():
			model.eval()
			output = model(x_val)
			val_loss = loss_function(output, y_val)
			predict_label = torch.argmax(output,1).cpu()
			val_acc = np.mean((predict_label == y_val.cpu()).numpy())

		print('\n====> Epoch: {} \nTrain:\nCE_Loss: {:.6f} | Accuracy: {:.6f} \nValidation:\nCE_loss: {:.6f} | Accuracy: {:.6f}'.format(
			epoch+1, CE_loss/len(train_loader), Train_Acc/len(train_loader), val_loss, val_acc))

		train_loss_list.append(CE_loss/len(train_loader))
		train_acc_list.append(Train_Acc/len(train_loader))
		val_loss_list.append(val_loss)
		val_acc_list.append(val_acc)
		#Checkpoint
        	#torch.save(model.state_dict(), "./models/CNN_FC_model.pkt")
		"""
		if (val_acc > best_accuracy):
			best_accuracy = val_acc
			torch.save(model, 'save_models/CNN/CNN_'+str(epoch)+'.pkl')
			print ('Saving Improved Model(val_acc = %.6f)...' % (val_acc))
		"""
		print('-'*88)
	"""
	with open('./checkpoint/Q1/train_loss.pkl', 'wb') as fp:
		pickle.dump(train_loss_list, fp)
	with open('./checkpoint/Q1/train_acc.pkl', 'wb') as fp:
		pickle.dump(train_acc_list, fp)
	with open('./checkpoint/Q1/val_loss.pkl', 'wb') as fp:
		pickle.dump(val_loss_list, fp)
	with open('./checkpoint/Q1/val_acc.pkl', 'wb') as fp:
		pickle.dump(val_acc_list, fp)
	"""


def main():

	train_x = np.load(PATH+"data.npy")#.reshape(42308, 3000, 1)
	train_y = np.load(PATH+'label.npy')#.reshape(42308, 3000, 1)
	for i, label in enumerate(train_y):
		if label == int(4):
			train_y[i] = 3
		elif label == int(5):
			train_y[i] = 4
	train_y = np_utils.to_categorical(train_y,5)
	train_x = train_x.transpose(0,2,1)
	print("train_x shape :", train_x.shape)		#(42308, 3000, 1)
	print("train_y shape :", train_y.shape)		#(42308, 5)

	x_train = train_x[:40000]
	x_val = train_x[40000:]
	y_train = train_y[:40000]
	y_val = train_y[40000:]

	x_train = torch.from_numpy(x_train)
	y_train = torch.from_numpy(y_train)
	x_val = torch.from_numpy(x_val)
	y_val = torch.from_numpy(y_val)
	dataset = Data.TensorDataset(x_train, y_train)
	train_loader = Data.DataLoader(dataset, batch_size=100, shuffle=True, num_workers=8)
	print("data load done")
	train(100, train_loader, x_val, y_val)
	
	

if __name__ == '__main__':
	main()


