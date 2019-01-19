import csv
import random
from sys import argv
import numpy as np 
import pandas as pd 
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score,accuracy_score,recall_score,confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings  
import pickle
import itertools 

def main():
	
	dataX = np.load('xgb/testX.npy').astype(float)
	dataY = np.load('xgb/testY.npy').astype(int)

	print(dataX.shape)
	mean = np.load('xgb/mean.npy')
	std = np.load('xgb/std.npy')
	dataX = (dataX - mean)/std
	test_epo = calepoch(dataY)
	print('test size: ', np.sum(test_epo).astype(int),' test epoch: ', test_epo)
	
	name = argv[1]
	thresh = np.load('xgb/model/thresh/npy'+str(name)+'h.npy')
	thresh_model = pickle.load(open('xgb/model/thresh/model'+str(name)+'.model','rb'))
	selection = SelectFromModel(thresh_model, threshold=thresh, prefit=True)
	dataX = selection.transform(dataX)
	
	print('feature size: ', dataX.shape[1])
	model = pickle.load(open('xgb/model/'+str(name)+'.model','rb'))
	y_pred = model.predict(dataX)
	acc = calacc(dataY,y_pred)
	f1 = f1_score(dataY,y_pred,average='macro')
	sense =	recall_score(dataY,y_pred,average='macro')
	print("test acc: %.3g, f1 score: %.3f, sensitivity: %.3f" %(acc, f1 ,sense))
	plot_conf_matrx(dataY, y_pred)



def calacc(ans,predict):
	correct = 0
	for i in range(len(ans)):
		if predict[i] == ans[i]:
			correct+=1
	acc = correct/len(ans)
	return acc

def calepoch(label):
	epoch = np.zeros(5)
	for i in label:
		for j in range(5):
			if int(i)==int(j):
				epoch[j]+=1
	return epoch


def plot_conf_matrx(ans, predict):
	classes = ['w','s1','s2','s3','R']
	cm = confusion_matrix(ans, predict, labels=[0,1,2,3,4])
	np.set_printoptions(precision=2)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Confusion matrix')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' 
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig('xgb/confusion_matrix.png')
	plt.show()

if __name__ == '__main__':
	warnings.filterwarnings("ignore")
	main()
