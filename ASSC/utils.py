import numpy as np
from scipy.signal import butter, lfilter, hilbert
from scipy import signal
from scipy.stats import skew, kurtosis
import pywt
import pyeeg
from biosppy.signals import eeg
from pyentrp import entropy as ent
from pyhht.emd import EMD
import sys
sys.setrecursionlimit(800)


def butter_bandpass_filter(data, fl, fh, fs=100, order=4):
	nyq = 0.5 * fs
	low = fl / nyq
	high = fh / nyq
	b, a = butter(order, [low, high], btype='band', analog=False)
	y = lfilter(b, a, data)
	return y

def standard_statistics(data):
	mean = np.mean(data)
	std = np.std(data)
	Skew = skew(data)
	kurt = kurtosis(data)
	fstdiff = np.sum(order_diff(data,1))/(len(data)-1)
	norfstdiff = fstdiff/std
	secdiff = np.sum(order_diff(data,2))/(len(data)-2)
	norsecdiff = secdiff/std
	name = ['mean','std','skew','kurt','fstdif','norfdf','secdiff','norsdf']
	return name, [mean, std, Skew, kurt, fstdiff, norfstdiff, secdiff, norsecdiff]

def time_analysis(data):
	zc_r = crossing(data,0)
	IEEG = np.sum(abs(data))
	Hj_act = np.var(data)
	Hj_mob = np.std(order_diff(data,1))/np.std(data)
	Hj_cplx = np.std(order_diff(order_diff(data,1),1))/(np.std(order_diff(data,1))*Hj_mob)
	dfa = pyeeg.dfa(data)
	#ts_entpy = tsallis_entropy(data)
	shn_entpy = ent.shannon_entropy(data)
	#HFD = pyeeg.hfd(data, 1500)
	#name = ['zc_r','IEEG','Hj_act','Hj_mob','Hj_cplx','dfa','shn_entpy','HFD']
	#return name, [zc_r, IEEG, Hj_act, Hj_mob, Hj_cplx, dfa, shn_entpy, HFD]
	name = ['zc_r','IEEG','Hj_act','Hj_mob','Hj_cplx','dfa','shn_entpy']
	return name, [zc_r, IEEG, Hj_act, Hj_mob, Hj_cplx, dfa, shn_entpy]

def nonparametric_frequency(data):
	fs = 100
	window_size = 5.0
	bands = [[0.4, 4.5],[4.5, 8.5], [8.5, 10], [10, 13], [13, 25], [25, 40], [8.5,13], [0,40]]
	f, power = signal.welch(data, fs, window='hamming',nperseg=500,noverlap=0,nfft=3000)
	p = []
	for i, b in enumerate(bands):
		p.append(band_power(freqs=f, power=power, frequency=b))

	all_power = np.sum(power)
	delta = np.sum(p[0])/all_power
	theta = np.sum(p[1])/all_power
	alpha_low = np.sum(p[2])/all_power
	alpha_high = np.sum(p[3])/all_power
	beta = np.sum(p[4])/all_power
	gamma = np.sum(p[5])/all_power
	DSI = np.mean(p[0])/(np.mean(p[6])+np.mean(p[1]))
	TSI = np.mean(p[1])/(np.mean(p[6])+np.mean(p[0]))
	ASI = np.mean(p[6])/(np.mean(p[0])+np.mean(p[1]))
	spe_ent = pyeeg.spectral_entropy(X=data, Band=[0,40], Fs=100, Power_Ratio=(p[7])/np.sum(power))
	'''
	all_power = total_power(data, fs)
	x = np.reshape(data,(-1,1))
	power = np.asarray(eeg.get_power_features(signal=x, sampling_rate=fs, size=window_size, overlap=0)[1:6])
	theta = np.mean(power[1])
	alpha_low = np.mean(power[2])
	alpha_high = np.mean(power[3])
	beta = np.mean(power[4])
	gamma = np.mean(power[5])
	'''
	name = ['all_p', 'delta_r','theta_r','alpha_low_r','alpha_high_r','beta_r','gamma_r','DSI','TSI','ASI','spe_ent']
	return name, [all_power,delta,theta,alpha_low,alpha_high,beta,gamma,DSI,TSI,ASI, spe_ent]

def parametric_frequency(data):
	fs = 100
	f, power = signal.welch(data, fs, window='hamming',nperseg=500,noverlap=0,nfft=3000)
	peak_freq =  f[np.argmax(power)]
	q1_freq = sef(f,power, 0.25)
	cent_freq = sef(f, power, 0.5)
	q3_freq = sef(f,power, 0.75)
	high_freq = sef(f, power, 0.95)
	sefIR = q3_freq - q1_freq
	sefd = high_freq - cent_freq
	ssd = np.std(power)
	Skew = skew(power)
	kurt = kurtosis(power)
	name = ['peak_f','q1_f','cent_f','q3_f','95_f','sefIR','sefd','ssd','skew_P','kurt_P']
	return name, [peak_freq, q1_freq, cent_freq, q3_freq, high_freq, sefIR, sefd, ssd, Skew, kurt]


def harmonic_parameter(data):
	fs = 100
	f, power = signal.welch(data, fs, window='hamming',nperseg=500,noverlap=0,nfft=3000)
	fc = np.sum(np.multiply(f, power))/np.sum(power)
	fsigma = np.sqrt(np.sum((f-fc)**2*f*power)/np.sum(power))
	for i, feq in enumerate(f):
		if feq > fc:
			p = (power[i]*(fc-f[i-1])+power[i-1]*(f[i]-fc))/(f[i]-f[i-1])
	name = ['harm_fc', 'harm_fs', 'harm_p']
	return name, [fc, fsigma, p]



def DWT(data):
	sig = pywt.wavedec(data, 'db4', mode='symmetric', level=6, axis=-1)
	mean = []
	ratio_mean = []
	std = []
	avgP = []
	Skew = []
	kurt = []
	name = []
	for i, s in enumerate(sig):
		mean.append(np.mean(abs(s)))
		std.append(np.std(s))
		avgP.append(np.sum(s**2)/len(s))
		Skew.append(skew(s))
		kurt.append(kurtosis(s))
		name.append('DWTmean'+str(i))
		name.append('DWTstd'+str(i))
		name.append('DWTavgP'+str(i))
		name.append('DWTskew'+str(i))
		name.append('DWTkurt'+str(i))
	for i in range(len(mean)-1):
		ratio_mean.append(mean[i+1]/mean[i])
		name.append('DWTrtomean'+str(i))
	DWT_F = mean+std+avgP+Skew+kurt+ratio_mean
	return name,DWT_F

def EMD_analysis(data):
	decomposer = EMD(data)
	imfs = decomposer.decompose() 

	z = hilbert(imfs) + imfs
	a = np.abs(z)
	phase  = np.unwrap(np.angle(z))
	i_freq = (np.diff(phase)/(2*np.pi))*100
	band = [[0.4,1.55],[1.55,3.2],[3.2,8.6],[8.6,11.0],[11.0,15.6],[15.6,22.0],[22.0,30]]
	power = []
	e = 0
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
	name = ['EMD_delta','EMD_alpha','EMD_beta','EMD_at','EMD_dt','EMD_ks']
	return name, [delta, alpha, beta, a_t, d_t, k_s]
	
def complex_measure(data):
	PFD = pyeeg.pfd(data)
	ap_ent = pyeeg.ap_entropy(data,1,0.15*np.std(data))
	hurst_exp = pyeeg.hurst(data)
	#samp_ent = ent.sample_entropy(data,2,0.2*np.std(data)) #list
	#mspe = ent.multiscale_permutation_entropy(data, 5, 1, 20) #list
	#name = ['PFD', 'ap_ent', 'hurst_exp', 'samp_ent1', 'samp_ent2']
	name = ['PFD', 'ap_ent', 'hurst_exp']
	#for i in range(len(mspe)):
	#	name.append('mspe'+str(i))
	#return name, [PFD, ap_ent, hurst_exp]+samp_ent.tolist()+mspe
	return name, [PFD, ap_ent, hurst_exp]



def paper_2017(data):
	delta = butter_bandpass_filter(data, 0.4, 4.5)
	theta =  butter_bandpass_filter(data, 4.5, 8.5)
	low_alpha =  butter_bandpass_filter(data, 8.5, 10)
	high_alpha =  butter_bandpass_filter(data, 10, 13)
	beta =  butter_bandpass_filter(data, 13, 25)
	mdd = [MDD(delta),MDD(theta),MDD(low_alpha),MDD(high_alpha),MDD(beta)]
	esis = [np.sum(delta**2)*100*(0.4+4.5)*0.5,np.sum(theta**2)*100*(4.5+8.5)*0.5,
	np.sum(low_alpha**2)*100*(8.5+10)*0.5,np.sum(high_alpha**2)*100*(10+13)*0.5,np.sum(beta**2)*100*(13+25)*0.5]
	name = ['mdd_delta','mdd_theta','mdd_low_alpha','mdd_high_alpha','mdd_beta', 'esis_delta', 
	'esis_theta','esis_low_alpha','esis_high_alpha','esis_beta']
	return name, mdd+esis


	
def MDD(data):
	d = 0
	for j in range(30):
		maxY = np.max(data[j*100:j*100+100])
		minY = np.min(data[j*100:j*100+100])
		maxX = np.argmax(data[j*100:j*100+100])
		minX = np.argmin(data[j*100:j*100+100])
		d += np.sqrt((maxY-minY)**2 + (maxX-minX)**2)
	return d


def total_power(data, fs):
	signal = x
	npoints = len(signal)
	Nyq = float(fs) / 2
	hpoints = npoints // 2
	freqs = np.linspace(0, Nyq, hpoints)
	power = np.abs(np.fft.fft(signal, npoints)) / npoints
	power = power[:hpoints]
	power[1:] *= 2
	power = np.power(power, 2)
	return np.mean(power)


def crossing(signal, ref):
	count = 0
	test = signal - ref
	for i in range(1, len(signal)):
		if test[i] * test[i-1] < 0:
			count += 1
	return count / (len(signal)-1)

def order_diff(data, order):
	D=[]
	for i in range(0,(len(data)-int(order))):
		D.append(abs(data[i+order]-data[i]))
	return D

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
	p = power[sel]
	return p

def sef(feq, power, ratio):
	for i, f in enumerate(feq):
		if np.sum(power[0:i]) > (np.sum(power)*ratio):
			return (feq[i]*(np.sum(power)*ratio-np.sum(power[0:i-1]))+feq[i-1]*(np.sum(power[0:i])-np.sum(power)*ratio))/power[i]


			
