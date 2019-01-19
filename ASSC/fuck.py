import matplotlib.pyplot as plt
import numpy as np
import pyeeg
import pywt
from scipy import signal
from scipy.signal import hilbert
from scipy.signal import butter, lfilter, hilbert
from biosppy.signals import eeg
from biosppy.signals import tools as st
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d
from sys import argv
from pyhht.visualization import plot_imfs
from pyhht.emd import EMD
from pyentrp import entropy as ent
from pyhht.utils import *
import matplotlib.pyplot as plt
#np.random.seed(1234)




train_data = np.load('npy/train_x1.npy')
train_y = np.load('npy/train_y.npy')
print(train_data.shape)


#select = argv[1]
for i in range(train_y.shape[0]):
	if train_y[i] == int(3):
		title = 'stage3'
		plt.title(title)
		plt.ylabel('Amplitude(uV)')
		plt.plot(train_data[i])
		plt.show()
input()





















def band_power(freqs=None, power=None, frequency=None):
	try:
		f1, f2 = frequency
	except ValueError:
		raise ValueError("Input 'frequency' must be a pair of frequencies.")
	if f1 < freqs[0]:
		f1 = freqs[0]
	if f2 > freqs[-1]:
		f2 = freqs[-1]
	sel = np.nonzero(np.logical_and(f1 <= freqs, freqs <= f2))[0]
	avg = (power[sel])
	return avg

def sef(feq, power, ratio):
	for i, f in enumerate(feq):
		if np.sum(power[0:i]) > (np.sum(power)*ratio):
			return (feq[i]*(np.sum(power)*ratio-np.sum(power[0:i-1]))+feq[i-1]*(np.sum(power[0:i])-np.sum(power)*ratio))/power[i]

def paper_2017(data):
	d = 0
	for j in range(30):
		maxY = np.max(data[j*100:j*100+100])
		minY = np.min(data[j*100:j*100+100])
		maxX = np.argmax(data[j*100:j*100+100])
		minX = np.argmin(data[j*100:j*100+100])
		d += np.sqrt((maxY-minY)**2 + (maxX-minX)**2)
	return d

def butter_bandpass_filter(data, fl, fh, fs=100, order=4):
	nyq = 0.5 * fs
	low = fl / nyq
	high = fh / nyq
	b, a = butter(order, [low, high], btype='band', analog=False)
	y = lfilter(b, a, data)
	return y


def order_diff(data, order):
	D=[]
	for i in range(0,(len(data)-int(order))):
		D.append(abs(data[i+order]-data[i]))
	return D

def crossing(signal, ref):
	count = 0
	test = signal - ref
	for i in range(1, len(signal)):
		if test[i] * test[i-1] < 0:
			count += 1
	return count / (len(signal)-1)

dataX = np.load('f_122.npy')[0]
print(dataX.shape)
file = open('xgb/feature.csv','w+')
for i, feature in enumerate(dataX):
	file.write(str(i)+','+str(feature)+'\n')
file.close()


















'''
fs = 100
N = 3000

amp = 10
freq = 30.0
noise_power = 10 * fs / 2
time = np.arange(N) / fs
x = amp*np.sin(2*np.pi*freq*time)
x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
x1 = x
train_data = np.load('train_x1.npy')
x = train_data[int(argv[1])]
data =  butter_bandpass_filter(x, 0.1, 45)


print(crossing(data,0))
print(np.sum(abs(data)))
print(np.var(data))
Hj_mob = np.std(order_diff(data,1))/np.std(data)
print(Hj_mob)
print(np.std(order_diff(order_diff(data,1),1))/(np.std(order_diff(data,1))*Hj_mob))
print(pyeeg.dfa(data))
#ts_entpy = tsallis_entropy(data)
print(ent.shannon_entropy(data))
print(pyeeg.hfd(data, 1500))










delta = butter_bandpass_filter(x, 0.4, 4.5)
theta =  butter_bandpass_filter(x, 4.5, 8.5)
low_alpha =  butter_bandpass_filter(x, 8.5, 10)
high_alpha =  butter_bandpass_filter(x, 10, 13)
beta =  butter_bandpass_filter(x, 13, 25)
print(paper_2017(delta), paper_2017(theta), paper_2017(low_alpha))
print(paper_2017(high_alpha), paper_2017(beta))
print("===")
print(np.sum(delta**2)*100*(0.4+4.5)*0.5)
print(np.sum(theta**2)*100*(4.5+8.5)*0.5)
print(np.sum(low_alpha**2)*100*(8.5+10)*0.5)
print(np.sum(high_alpha**2)*100*(10+13)*0.5)
print(np.sum(beta**2)*100*(13+25)*0.5)
#x1 = x

#print(x.shape[0])
#print("===")

print(pyeeg.pfd(x))
print(pyeeg.ap_entropy(x,1,0.15*np.std(x)))
print(ent.multiscale_entropy(x,2,0.2*np.std(x)))

print(pyeeg.hurst(x))
print(ent.multiscale_permutation_entropy(x, 5, 1, 20))


t = np.linspace(0, 1, N)
#modes = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
#x = modes + t

decomposer = EMD(x)
imfs = decomposer.decompose()

z = hilbert(imfs) 
a = np.abs(z)
phase  = np.unwrap(np.angle(z))
i_freq = (np.diff(phase)/(2*np.pi))*100
band = [[0.4,1.55],[1.55,3.2],[3.2,8.6],[8.6,11.0],[11.0,15.6],[15.6,22.0],[22.0,30]]
power = []
e = 0
print(i_freq.shape)
print(a.shape)
for f1,f2 in band:
	for imf in range(i_freq.shape[0]):
		for t in range(i_freq.shape[1]):
			if i_freq[imf][t] >= f1 and i_freq[imf][t] <= f2:
				e += ((a[imf][t+1]+a[imf][t])/2)
	power.append(e)			
power = np.array(power)
total_power = np.sum(power)
delta = (power[0]+power[1])/total_power
alpha = power[3]/total_power
beta = (power[5]+power[6])/total_power
a_t = power[3]/power[2]
d_t = (power[0]+power[1])/power[2]
k_s = (power[0]+power[4])/total_power

print(delta,alpha,beta)
print(a_t,d_t,k_s)

sig = pywt.wavedec(x1, 'db4', mode='symmetric', level=4, axis=-1)
xold = np.arange(0,len(sig[0]))
yold = sig[0]
f = interp1d(xold,yold,kind='cubic')
xnew = np.arange(0,len(sig[0])-1,(len(sig[0])-1)/3000)
ynew = f(xnew)
plt.subplot(2,1,1)
plt.plot(x1)
plt.subplot(2,1,2)
plt.plot(ynew)
plt.show()
print('***')

x = ynew

f1, power= signal.welch(x, 100, window='hamming',nperseg=500,noverlap=0,nfft=3000)

f2, t, Zxx = signal.stft(x, fs, window='hamming',nperseg=1500,noverlap=0,nfft=1999)#3*1000
#f2, t, Zxx = signal.stft(x, fs, window='hamming',nperseg=600,noverlap=0,nfft=999)#6*500
#f2, t, Zxx = signal.stft(x, fs, window='hamming',nperseg=1001,noverlap=0,nfft=1499) #4*750
print(Zxx.shape)
Zxx = np.reshape(Zxx.T,(-1,1))
Zxx = abs(Zxx)


npoints = len(x)
Nyq = float(fs) / 2
hpoints = npoints // 2
freqs = np.linspace(0, Nyq, hpoints)
p = np.abs(np.fft.fft(x, npoints)) / npoints
p = p[:hpoints]
p[1:] *= 2
p = np.power(p, 2)

#print(f)
print(power.shape, np.sum(power))
print(Zxx.shape, np.sum(Zxx))
print(p.shape, np.sum(p))

plt.subplot(3,1,1)
plt.plot(power)
plt.subplot(3,1,2)
plt.plot(Zxx)
plt.subplot(3,1,3)
plt.plot(p)
plt.show()

print("***")
avg = []
bands = [[4, 8], [8, 10], [10, 13], [13, 25], [25, 40], [0,40]]
for i, b in enumerate(bands):
	avg.append(band_power(freqs=f1, power=power, frequency=b))
	print('band: ',np.mean(avg[i]))
	#print(np.sum(avg[i])/np.sum(power))


x = np.reshape(x,(-1,1))
peeg = np.asarray(eeg.get_power_features(signal=x, sampling_rate=fs, size=5, overlap=0)[1:6])
peeg = np.mean(peeg,axis=1)
print(peeg)



for i in range(len(sig)):
	#print(np.mean(abs(signal[i])))
	print(np.sum(sig[i]**2)/len(sig[i]))
	#print(np.mean(p))
#print(sig[0])

plt.subplot(2,1,1)
plt.semilogy(f,power)
plt.subplot(2,1,2)
plt.semilogy(sig[1])
plt.close()

#plt.show()
plt.semilogy(f, power)
plt.ylim([0.1, 1000])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.savefig('psd.png')
plt.close()


#plt.ylim([0.5e-3, 10])
#plt.xlabel('frequency [Hz]')
#plt.ylabel('PSD [V**2/Hz]')

#plt.close()

print("===")
peak_freq =  f[np.argmax(power)]
q1_freq = sef(f,power, 0.25)
cent_freq = sef(f, power, 0.5)
q3_freq = sef(f,power, 0.75)
high_freq = sef(f, power, 0.95)
sefIR = q3_freq - q1_freq
sefd = high_freq - cent_freq
ssd = np.std(power)
spe_ent = pyeeg.spectral_entropy(X=x, Band=[4,8], Fs=100, Power_Ratio=(avg[5]/np.sum(power)))


fc = np.sum(np.multiply(f, power))/np.sum(power)
fsigma = np.sqrt(np.sum((f-fc)**2*f*power)/np.sum(power))
for i, feq in enumerate(f):
	if feq > fc:
		#print(power[i], power[i-1])
		p = (power[i]*(fc-f[i-1])+power[i-1]*(f[i]-fc))/(f[i]-f[i-1])
		break
print('peak_freq ',peak_freq)
print('q1 ', q1_freq)
print('cent_freq ',cent_freq)
print('q3 ', q3_freq)
print('high_freq ',high_freq)
print('IR ',sefIR)
print('sefd ', sefd)
print('ssd', ssd)
print('skew ', skew(power))
print('kurt ', kurtosis(power))
print('SpEn ', spe_ent)
print('fc ', fc )
print('fsigma', fsigma)
print('fc_power', p)
'''