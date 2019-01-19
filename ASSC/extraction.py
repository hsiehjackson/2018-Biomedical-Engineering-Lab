import os
from utils import *
import warnings




stage = [0,0,0,0,0,0]
stage = np.array(stage)
#stage = [W, 1, 2, 3, 4 ,R]
# base = [ 9643  4809 27187  5043  3762 11811] 62255
# base nor [ 9644  4810 27194  5043  3762 11820] 62273
# overlap = [9643  4809 27187  5043  3762 11811] 62133
# overlap nor = 
# ST = [2067 2005 9388 1673 1429 4094] 20656



def read_csv_data(filepath, first, last, stop):
	csv_path_list = os.listdir(filepath)
	csv_path_list.sort()
	train_x1 = []
	train_x2 = []
	train_y = []
	for i, file in enumerate(csv_path_list):
		f = open(os.path.join(filepath, file),'r',encoding='utf8')
		next(f)
		t_x1 = []
		t_x2 = []
		t_y = []
		for j, row in enumerate(f) :
			tmp = row.split(',')
			tmp_x1 = []
			tmp_x2 = []
			if int(tmp[0])>stop: break
			if int(tmp[0])>=first and int(tmp[0])<len(tmp)+last:
				if int(tmp[1]) == 6:
					continue
				stage[int(tmp[1])] +=  int(1)
				t_y.append(int(tmp[1]))
				for i in range(2,len(tmp)-1):
					if len(tmp_x1)<3000:
						tmp_x1.append(float(tmp[i]))
					else:
						tmp_x2.append(float(tmp[i]))
				t_x1.append(tmp_x1)
				t_x2.append(tmp_x2)
		for i in range(len(t_x1)):
			train_x1.append(t_x1[i])
			train_x2.append(t_x2[i])
			train_y.append(t_y[i])
		print(np.array(train_x1).shape)
		print(np.array(train_x2).shape)
		print(np.array(train_y).shape)
		print(stage)
		f.close()
	train_x1 = np.array(train_x1)
	train_x2 = np.array(train_x2)
	train_y = np.array(train_y)
	return train_x1,train_x2, train_y


def save():
	train_STx1, train_STx2, train_STy = read_csv_data('data_ST_CSV', 10, -5, 990)
	#train_SCx1, train_SCx2, train_SCy = read_csv_data('new_SC_CSV', 0, 0, 10000) 
	#train_x1 =  np.concatenate((train_STx1,train_SCx1),axis=0)
	#train_x2 = np.concatenate((train_STx2,train_SCx2),axis=0)
	#train_y = np.concatenate((train_STy,train_SCy),axis=0)
	print(np.array(train_STx1).shape)
	print(np.array(train_STx2).shape)
	print(np.array(train_STy).shape)
	np.save('npy/trainST_x1.npy', train_STx1)
	np.save('npy/trainST_x2.npy', train_STx2)
	np.save('npy/trainST_y.npy', train_STy)

def main():
	#save()

	trainX = np.load('npy/train_x1.npy')
	name = []
	feature = []
	print(trainX.shape)
	for i in range(len(trainX)):
		#print('data number: ',i)
		#print('bandpass ...')
		trainband =  butter_bandpass_filter(trainX[i], 0.1, 45)
		#print('standard_statistics...')
		n_1, f_1 = standard_statistics(trainband)
		#print('time_analysis...')
		n_2, f_2 = time_analysis(trainband)
		#print('nonparametric_frequency...')
		n_3, f_3 = nonparametric_frequency(trainband)
		#print('parametric_frequency...')
		n_4, f_4 = parametric_frequency(trainband)
		#print('harmonic_parameter...')
		n_5, f_5 =harmonic_parameter(trainband)
		#print('DWT...')
		n_6, f_6 = DWT(trainband)
		#print('EMD_analysis...')
		n_7, f_7 = EMD_analysis(trainband)
		#print('complex_measure...')
		n_8, f_8 = complex_measure(trainband)
		#print('paper_2017...')
		n_9, f_9 = paper_2017(trainband)
		if not len(name):
			name = n_1+n_2+n_3+n_4+n_5+n_6+n_7+n_8+n_9
			print('name size ...', np.array(name).shape)
		f = f_1+f_2+f_3+f_4+f_5+f_6+f_7+f_8+f_9
		feature.append(f)
		print('feature size...', np.array(feature).shape)
		print("==================")

	feature = [name]+feature
	np.save('f_122.npy',feature)	
	


if __name__ == '__main__':
	warnings.filterwarnings("ignore")
	main()


