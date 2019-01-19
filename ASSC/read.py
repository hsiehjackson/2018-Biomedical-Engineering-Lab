import numpy as np
import sys
import csv
import os
from sys import argv
from scipy.signal import butter, lfilter


def butter_bandpass(fh, fl, fs, order=5):
	nyq = 0.5 * fs
	low = fl / nyq
	high = fh / nyq
	b, a = butter(order, [low, high], btype='band', analog=False)
	return b, a

def butter_bandpass_filter(data, fh, fl, fs=100, order=4):
	b, a = butter_bandpass(fh, fl, fs, order=order)
	y = lfilter(b, a, data)
	return y


def read_X(filename,start):
	print("reading...{}".format(filename))
	data=[]
	a=[]
	b=[]
	signal1=[]
	signal2=[]
	with open(filename, 'r') as f:
		f.readline()
		linenum=0
		for row in f:
			every_row=row.split(',')
			time=float(every_row[0])

			if(time>=start):
				if(time==start):
					print(time)
				signal1.append(float(every_row[1]))
				signal2.append(float(every_row[2]))

	signal1 = np.array(signal1)
	signal1 = (signal1-np.mean(signal1))/np.std(signal1)
	signal2 = np.array(signal2)
	signal2 = (signal2-np.mean(signal2))/np.std(signal2)

	for i, sig in enumerate(signal1):
		a.append(signal1[i])
		b.append(signal2[i])
		if((i+1)%3000==0):
			data2 = []
			for x in a:
				data2.append(x)
			for y in b:
				data2.append(y)
			data.append(data2)
			a=[]
			b=[]
	return data
def read_Y(filename):
	print("reading...{}".format(filename))
	data=[]
	linenum=0
	with open(filename, 'r') as f:
		f.readline()
		for row in f:
			every_row=row.split(',')
			if(linenum==0):
				start=int(every_row[0][1:])
				print(start)
			linenum+=1
			stage=every_row[2][12]
			#+1560,90,Sleep stage 1
			head=int(every_row[0][1:])
			length=int(every_row[1])
			if(stage=="W"):
				stage=0
			elif(stage=="R"):
				stage=5
			elif(stage=="1" or stage=="2" or stage=="3" or stage=="4"):
				stage=int(stage)
			else:
				stage=6
			epoch=int(length/30)
			for i in range(epoch):
				data.append(stage)
	return data,start

def write_out(x,y,out):
	print("writing...{}".format(out))
	f = open((out),"w+")
	f.write("epoch,label,data\n")
	linenum=0
	temp=0
	for item in range(len(y)):
		if y[item] == 6:
			continue
		if item < len(x):
			if y[item]==temp:
				continue
			temp=100
			if y[item]==0 and y[item+1]==0 and y[item+2]==0 and y[item+3]==0 and y[item+4]==0 and linenum>1000: 
				continue
			data=np.array(x[item])
			f.write("{},{},".format(linenum,y[item]))
			for d in data:
				f.write("{},".format(d))
			f.write("\n")
			linenum+=1
	print("epoch= {}".format(linenum))
	f.close()



PATH="SC_CSV/"
folders_subjects = os.listdir(PATH)
folders_subjects.sort()
for s in folders_subjects: # 資料夾 s005 s10 ....
	if(s[9:16]=="Hypnogr"):
		file1 = str(PATH)+str(s)
		y,start = read_Y(file1)#"ST7011J0-PSG_data.txt" ST7241JO-Hypnogram_annotations.txt
		for b in folders_subjects:
			if(b[0:6]==s[0:6] and b[9]!="H"):
				#print('=======',str(b[0:6]))
				file2 = str(PATH)+str(b)
				x=read_X(file2,start)
				filename=s.split('_')[0]
				print('lable shape ',np.array(y).shape)
				print('data shape ',np.array(x).shape)
				write_out(x,y,"new_SC_CSV/"+filename+".csv")
				#write_out(x,y,"test.csv")








