#Importing modules


import numpy as np
import matplotlib.pyplot as plt
import obspy
import tqdm
import glob
import os


#Defining date arguments
day1 = 1
delta_days = 364 + 1
day2 = day1 + delta_days
year = '2015'
totals = delta_days * 24 *3600

#Defining the windows argument
window_duration_sec =360
average = 20
overlap = 0.5
preproc_spectral_secs = window_duration_sec * average * overlap

#Defining the signal parameters
s_rate = 20 #signal rate
Fn = s_rate/2 #Nyquist
lowfreq = 1.0 #low freq idx
hfreq = 3.0
smoothF = 0.05 #frequency for smoothing
thres = 0.05 #threshold
win_wien = 360

#Defining time and frequency vectors
npts = (window_duration_sec * s_rate *2 ) - 1 #number of points
T =  window_duration_sec *2
DelF = 1/T
DelT = (window_duration_sec/2) * (average/2)
freq_tmp = np.fft.fftfreq(npts,d=1/s_rate)
lf_idx = np.argwhere(freq_tmp[:int(npts/2)]>=lowfreq)[0][0] #lower freq index
hf_idx = np.argwhere(freq_tmp[:int(npts/2)]<=hfreq)[-1][0] #hfreq freq index
fr_vec = freq_tmp[lf_idx:hf_idx]

Winperday = (24 * 60) / (DelT / 60) - (window_duration_sec * average / DelT /2)
Winperyear = delta_days * Winperday


#Cluster window length



