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
matplotlib.use("agg")
import matplotlib.pyplot as plt
import warnings  
import pickle

#train size:  60030  train epoch:  [ 9281.  4626. 26236.  8470. 11417.]
#val size:  1225  val epoch:  [211. 105. 499. 187. 223.]
#test size: 1000  test epoch:  [151.  78. 452. 148. 171.]




def main():
	dataX = np.load('f_122.npy')[1:].astype(float)
	dataY = np.load('npy/train_y.npy').astype(int)
	
	for i, label in enumerate(dataY):
		if label == int(4):
			dataY[i] = 3
		elif label == int(5):
			dataY[i] = 4
	
	print(dataX.shape)
	print(dataY.shape)
	pair = list(zip(dataX,dataY))
	random.shuffle(pair)
	dataX, dataY = zip(*pair)			
	testdataX = np.array(dataX[:1000])
	testdataY = np.array(dataY[:1000])
	dataX = np.array(dataX[1000:])
	dataY = np.array(dataY[1000:])
	np.save('xgb/testX_nor.npy', testdataX)
	np.save('xgb/testY_nor.npy', testdataY)

	accuracy = []
	for threshold in range(1):

		pair = list(zip(dataX,dataY))
		random.shuffle(pair)
		dataX, dataY = zip(*pair)
		trainX = np.array(dataX[int(len(dataX)*0.02):])
		trainY = np.array(dataY[int(len(dataY)*0.02):])
		testX = np.array(dataX[:int(len(dataX)*0.02)])
		testY = np.array(dataY[:int(len(dataY)*0.02)])
		train_epo = calepoch(trainY)
		val_epo = calepoch(testY)
		test_epo = calepoch(testdataY)
		print('train size: ', np.sum(train_epo).astype(int),' train epoch: ', train_epo)
		print('val size: ', np.sum(val_epo).astype(int),' val epoch: ', val_epo)
		print('test size:', np.sum(test_epo).astype(int),' test epoch: ', test_epo)

		#mean = np.mean(trainX,axis=0)
		#std = np.std(trainX,axis=0) 
		#np.save('xgb/mean.npy',mean)
		#np.save('xgb/std.npy',std)
		#trainX = (trainX-mean)/std
		#testX = (testX-mean)/std
		tn_acc, f1_scr_t, ts_acc, f1_scr, sense, feature_size = train(trainX, trainY, testX, testY, threshold)

		print("===============================")
		print('feature select  = '+str(feature_size))
		print('train accuracy = '+str(tn_acc))
		print('train f1 score = '+str(f1_scr_t))
		print('test accuracy = '+str(ts_acc))
		print('test f1 score = '+str(f1_scr))
		print('sensitivity = '+str(sense))
		print("===============================")
		accuracy.append([feature_size, tn_acc, f1_scr_t, ts_acc, f1_scr, sense])

	writefile = open('xgb/xgb_nor.csv','w+',encoding = 'big5')
	writefile.write('fearure_size, train_acc, train_f1, test_acc, test_f1, sensitivity\n')
	for i in range(len(accuracy)):
		writefile.write(','.join(repr(accuracy[i][j]) for j in range(len(accuracy[i]))))
		writefile.write('\n')
	writefile.close()



def train(X_train, Y_train, X_test, Y_test, num):

	feature_size = 0	
	print('traing...')
	xgb1 = XGBClassifier(
	learning_rate =0.1,
	n_estimators=1000,
	max_depth=5,
	min_child_weight=1,
	gamma=0,
	subsample=0.8,
	colsample_bytree=0.8,
	objective= 'multi:softmax',
	nthread=4,
	scale_pos_weight=1,
	n_jobs=24)

	xgb1new, accuracy_t,f1score_t,accuracy, f1score, sense, cm = modelfit(xgb1, X_train, Y_train, X_test, Y_test)

	if num==0:
		pickle.dump(xgb1new, open('xgb/model/'+str(num)+'_nor.model','wb'))
		feature_size = X_train.shape[1]
		print("Thresh=%.6f, n=%d" %(0, X_train.shape[1]))
		print("train acc: %.3g, f1 score: %.3f" %(accuracy_t, f1score_t))
		print("test acc: %.3g, f1 score: %.3f" %(accuracy, f1score))
		print("sensitivity: ",sense)
		print(cm)
	
	print ( 'Ploting...')
	fig, ax = plt.subplots(1,1)
	xgb.plot_importance(xgb1new, max_num_features=30, ax=ax)
	fig.savefig('xgb/'+str(num)+'.png')
	plt.close()
	
	thresholds = list(set(xgb1new.feature_importances_))
	thresholds.sort(reverse=True)

	if num!=0: 
		
		thresh = thresholds[num-1]
		
		np.save('xgb/model/thresh/npy'+str(num)+'h_nor.npy',thresh)
		pickle.dump(xgb1new, open('xgb/model/thresh/model'+str(num)+'_nor.model','wb'))

		selection = SelectFromModel(xgb1new, threshold=thresh, prefit=True)
		select_X_train = selection.transform(X_train)
		select_X_test = selection.transform(X_test)
		feature_size = select_X_train.shape[1]
		xgb2 = XGBClassifier(
		learning_rate =0.1,
		n_estimators=1000,
		max_depth=5,
		min_child_weight=1,
		gamma=0,
		subsample=0.8,
		colsample_bytree=0.8,
		objective= 'multi:softmax',
		nthread=4,
		scale_pos_weight=1,
		n_jobs=24)

		temp,accuracy_t,f1score_t,accuracy,f1score,sense,cm = modelfit(xgb2, select_X_train, Y_train, select_X_test, Y_test)
		pickle.dump(temp, open('xgb/model/'+str(num)+'_nor.model','wb'))
		print("Thresh=%.6f, n=%d" %(thresh, select_X_train.shape[1]))
		print("train acc: %.3g, f1 score: %.3f" %(accuracy_t, f1score_t))
		print("test acc: %.3g, f1 score: %.3f" %(accuracy, f1score))
		print("sensitivity: ",sense)
		print(cm)

	
	return accuracy_t,f1score_t,accuracy, f1score, sense, feature_size

def modelfit(alg, X_train, Y_train, X_test, Y_test, cv_folds=10, early_stopping_rounds=10):

	xgb_param = alg.get_xgb_params()
	xgb_param['num_class'] = 5
	dtrain = xgb.DMatrix(X_train, label=Y_train)
	cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
	metrics='mlogloss', early_stopping_rounds=early_stopping_rounds,verbose_eval=True)

	alg.set_params(n_estimators=cvresult.shape[0])

	#Fit the algorithm on the data
	alg.fit(X_train, Y_train ,eval_metric='mlogloss')
	
	Ytrain_pred = alg.predict(X_train)
	Ytest_pred = alg.predict(X_test)

	
	acc_t = calacc(Y_train,Ytrain_pred) 
	f1_t = f1_score(Y_train,Ytrain_pred,average='macro')
	acc = calacc(Y_test,Ytest_pred)
	f1 = f1_score(Y_test,Ytest_pred,average='macro')
	sense =	recall_score(Y_test,Ytest_pred,average='macro')
	cm = confusion_matrix(Y_test, Ytest_pred)
	return alg,acc_t,f1_t,acc,f1,sense,cm

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
	classes = ['w','st_1','st_2','st_3','st_4','R']
	cm = confusion_matrix(ans, predict, labels=[0,1,2,3,4,5])
	np.set_printoptions(precision=2)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Confusion matrix')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig('confusion_matrix.png')
	plt.show()

if __name__ == '__main__':
	warnings.filterwarnings("ignore")
	main()


