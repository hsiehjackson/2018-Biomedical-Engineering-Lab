import numpy as np
from sys import argv
import os

#/home/mlpjb04/bioproject/deepsleepnet/data/eeg_fpz_cz/
def load_npz_file(npz_file):
	"""Load data and labels from a npz file."""
	with np.load(npz_file) as f:
		data = f["x"]
		labels = f["y"]
		sampling_rate = f["fs"]
	return data, labels, sampling_rate
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
				t_y.append(int(tmp[1]))
				for i in range(2,len(tmp)-1):
					if len(tmp_x1)<3000:
						tmp_x1.append(float(tmp[i]))
					else:
						tmp_x2.append(float(tmp[i]))
				t_x1.append(tmp_x1)
				t_x2.append(tmp_x2)

		print(np.array(t_x1).shape)
		print(np.array(t_x2).shape)

		f.close()
	train_x1 = np.array(train_x1)
	train_x2 = np.array(train_x2)
	train_y = np.array(train_y)
	return train_x1,train_x2, train_y

def save():
	train_STx1, train_STx2, train_STy = read_csv_data('data_ST_CSV', 10, -5, 990)

	print(np.array(train_x1).shape)
	print(np.array(train_x2).shape)
	print(np.array(train_y).shape)
	np.save('npy/train_x1_overlap_nor.npy', train_x1)
	np.save('npy/train_x2_overlap_nor.npy', train_x2)
	np.save('npy/train_y_overlap_nor.npy', train_y)

#save()
PATH="/home/mlpjb04/bioproject/deepsleepnet/data/eeg_fpz_cz/"
folders_subjects = os.listdir(PATH)
folders_subjects.sort()
first=0
for s in folders_subjects:
	print("file name={}".format(s))
	file1 = str(PATH)+str(s)
	if(first==0):
		data,label,s_rate=load_npz_file(file1)
	else:
		data1,label2,s_rate=load_npz_file(file1)
		data=np.concatenate((data,data1),axis=0)
		label=np.concatenate((label,label2),axis=0)
	print(data.shape,label.shape)
	first+=1
np.save("/home/mlpjb04/bioproject/deepsleepnet/data/data.npy",data)
np.save("/home/mlpjb04/bioproject/deepsleepnet/data/label.npy",label)

'''
save_dict = {
	"x": x, 
	"y": y, 
	"fs": sampling_rate,
	"ch_label": select_ch,
	"header_raw": h_raw,
	"header_annotation": h_ann,
}
np.savez(os.path.join(args.output_dir, filename), **save_dict)
'''