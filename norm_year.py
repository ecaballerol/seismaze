import numpy as np
import matplotlib.pyplot as plt
# import covseisnet as csn
import glob
from scipy import signal
from normalization import *
import matplotlib
from scipy.signal import wiener
from  matplotlib import ticker
#Arguments date
day1 = 1
delta_days = 364 + 1
day2 = day1 + delta_days
year = '2015'
totals = delta_days * 24 *3600


#Arguments windows
window_duration_sec =360
average = 20
overlap = 0.5
preproc_spectral_secs = window_duration_sec * average * overlap
time = np.arange(0,totals,preproc_spectral_secs/2)

#Directory of work
ODIR = './output_wdur' + str(window_duration_sec) + 's' + '_av' + str(average)

# Signal parameter
s_rate = 20 #signal rate
Fn = s_rate/2 #Nyquist
lfreq = 1.0 #low freq idx
hfreq = 10.0
#smoothF = 0.05 #frequency for smoothing
ismooth = [0.05]
thres = 0.05 #threshold

npts = (window_duration_sec * s_rate *2 ) - 1 #number of points
T =  window_duration_sec *2
DelF = 1/T
DelT = (window_duration_sec/2) * (average/2)
freq_tmp = np.fft.fftfreq(npts,d=1/s_rate)
lf_idx = np.argwhere(freq_tmp[:int(npts/2)]>=lfreq)[0][0] #lower freq index
hf_idx = np.argwhere(freq_tmp[:int(npts/2)]<=hfreq)[-1][0] #lower freq index
# fr_vec = freq_tmp[lf_idx:int(np.ceil(npts/2))] # frequency vector
fr_vec = freq_tmp[lf_idx:hf_idx] # frequency vector


#READING THE DATA 
spectral_year = []
for d in range(day1,day2):
    print(d)
    doy = f'{d:03}'
    doy = str(doy)
    SW = ODIR +'/SW_' + year + '_' +doy+ '_*'
    SW_name  = glob.glob(SW)[0]
    SWtmp = np.load(SW_name)
    spectral_year.extend(SWtmp)
    
spectral_year=np.array(spectral_year).T

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Windows to analyze

time_vec = np.linspace(day1-1,delta_days,delta_days*SWtmp.shape[0])

Eru_day =  267 #in days
Eru_win = Eru_day * SWtmp.shape[0]
Noerup = 100
Noeru_win = Noerup * SWtmp.shape[0]
Nosig_day = 311.22
Nosig_win = int(np.round(Nosig_day * SWtmp.shape[0]))



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#SMOOTHING
for smoothF in ismooth:
    smoothN = int(smoothF / DelF) +1
    sw_0 = np.zeros((hf_idx-lf_idx,spectral_year.shape[1]))
    sw_fin = np.zeros((hf_idx-lf_idx,spectral_year.shape[1]))
    win_info={}
    for i in np.arange(spectral_year.shape[1]):
        win_info[i]={}
        spectral_width = smooth1d(spectral_year[:,i],smoothN)
        sw_0[:,i] =  spectral_width[lf_idx:hf_idx]
        
        fav = env_sup(fr_vec, sw_0[:,i], hf_idx, lf_idx)
        win_info[i]['inter_1'] = fav
        
        sw3 = fav(fr_vec)-sw_0[:,i]
        # sw3 = fav(fr_vec)-spectral_year[lf_idx:int(np.ceil(npts/2)),i]
        
        fav2 = env_sup(fr_vec, sw3, hf_idx, lf_idx)
        win_info[i]['inter_2'] = fav2
        
    
        idx = sw3<thres
        sw3[idx] = 0
    
        fav2tmp = fav2(fr_vec)
        fav2tmp [np.argwhere(fav2tmp==0)] = 1e-5
        sw3norm = sw3/fav2tmp
        
        idx = sw3norm>1
        sw3norm[idx]=1   
        sw_fin[:,i] = sw3norm
        
    
    idx_neg =sw_fin<0
    sw_fin[idx_neg]=0
    
    spectral_year_plot = spectral_year[lf_idx:hf_idx,:]
    n_frequencies,n_times = spectral_year_plot.shape
    times = np.linspace(0, 1, n_times) * delta_days
    
    fig, ax = plt.subplots(1, figsize=(11, 6))
    img = ax.pcolorfast(times, fr_vec, spectral_year_plot, rasterized=True, cmap="RdBu")
    ax.set_xlabel('days')
    ax.set_ylabel('Frequency, Hz')
    ax.set_yscale('symlog',linthresh=1e-1,subs=[2,3,4,5,6,7,8,9,10])
    ax.set_yticks(np.arange(11))
    ax.set_ylim(lfreq,hfreq)
    plt.colorbar(img, ax=ax)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.title('spectral width:' + year)
    filename = 'Year/' + year + '_bp'+ str(lfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) + '_erup.png'
    plt.savefig(filename,dpi=200)
    filename = 'Year_Figures/' + year + '_bp'+ str(lfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) + '_erup.pdf'
    plt.savefig(filename)
    
    
    plt.figure(figsize=(15,8))
    plt.imshow(sw_0,origin='lower',extent=[day1-1,day2-1,lfreq,hfreq],aspect='auto',cmap='RdBu')
    plt.colorbar()
    plt.yscale('symlog',linthresh=1e-1,subs=[2,3,4,5,6,7,8,9])
    plt.ylim(lfreq,hfreq)
    plt.xlim(1,345)
    plt.ylabel('Frequency Hz')
    plt.xlabel('days')
    plt.title('smoothed spectral width: ' + str(smoothF) + 'Hz' )
    filename = 'Year_Smooth/' + year + '_bp'+ str(lfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) + '_smooth_' + str(smoothF) + 'Hz_erup.png'
    plt.savefig(filename,dpi=200)
    filename = 'Year_Figures/' + year + '_bp'+ str(lfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) + '_smooth_' + str(smoothF) + 'Hz_erup.pdf'
    plt.savefig(filename)
    
    fig, ax = plt.subplots(1, figsize=(11, 6))
    img = ax.imshow(sw_fin,origin='lower',extent=[day1-1,day2-1,lfreq,hfreq],aspect='auto',cmap='viridis')
    # img = ax.pcolorfast(times, fr_vec, sw_fin, rasterized=True, cmap="viridis")
    ax.set_xlabel('days')
    ax.set_ylabel('Frequency, Hz')
    ax.set_yscale('symlog',linthresh=1e-1,subs=[2,3,4,5,6,7,8,9,10])
    ax.set_yticks(np.arange(11))
    ax.set_ylim(lfreq,hfreq)
    plt.colorbar(img, ax=ax)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.title('Normalized spectral width, smooth:' + str(smoothF) + 'Hz')
    filename = 'Year_SW_spec/' + year + '_bp'+ str(lfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average)  \
        + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' + str(thres) + '.png'
    plt.savefig(filename,dpi=400)
    filename = 'Year_Figures/' + year +'_bp'+ str(lfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) \
        + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' + str(thres) + '.pdf'
    plt.savefig(filename)
    plt.close('all')

    #Median filter
    win_medfilt =  360 #In minutes
    mfiltnum = win_medfilt*60 / DelT +1
    sw_median = medfilt_days(sw_fin,int(mfiltnum))
    
    plt.figure(figsize=(15,8))
    plt.imshow(sw_median,origin='lower',extent=[day1-1,day2-1,lfreq,hfreq],aspect='auto',cmap='viridis',vmin=0,vmax=1)
    plt.colorbar()
    plt.yscale('symlog',linthresh=1e-1,subs=[2,3,4,5,6,7,8,9])
    plt.ylim(lfreq,hfreq)
    plt.xlim(1,345)
    plt.ylabel('Frequency Hz')
    plt.xlabel('days')
    plt.title('Normalized spectral width with median filter: ' + str(win_medfilt) + 'min')
    filename = 'Year_Median_filt/' + year +'_bp'+ str(lfreq) + '_' + str(hfreq) + 'Hz_win' + \
         str(window_duration_sec) + 's_av' + str(average) \
        + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' + str(thres) + 'medfilt_' + str(win_medfilt) + 'm.png'
    plt.savefig(filename,dpi=300)
    filename = 'Year_Figures/' + year + '_bp'+ str(lfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' \
        + str(thres) + 'medfilt_' + str(win_medfilt) + 'm.pdf'
    plt.savefig(filename)
    plt.close('all')

    #Wienner filter
    wien_lens = [360]
    for win_wien in wien_lens:
        #win_wien =  720 #In minutes
        wiennum = win_wien*60 / DelT +1
        # sw_wiener = wiener(sw_fin)
        sw_wiener = wiener(sw_fin,(1,int(wiennum)))
        plt.figure(figsize=(15,8))
        plt.imshow(sw_wiener,origin='lower',extent=[day1-1,day2-1,lfreq,hfreq],aspect='auto',cmap='viridis',vmin=0,vmax=1)
        plt.colorbar()
        plt.yscale('symlog',linthresh=1e-1,subs=[2,3,4,5,6,7,8,9])
        plt.ylim(lfreq,hfreq)
        plt.xlim(1,345)
        plt.ylabel('Frequency Hz')
        plt.xlabel('days')
        plt.title('Normalized spectral width with wiener filter: ' + str(win_wien) + 'min')
        filename = 'Year_Wiener/' + year + '_bp'+ str(lfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) \
            + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' + str(thres) + 'wienfilt_' + str(win_wien) + 'm.png'
        plt.savefig(filename,dpi=300)
        filename = 'Year_Figures/' + year + '_bp'+ str(lfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' \
            + str(thres) + 'wienfilt_' + str(win_wien) + 'm.pdf'
        plt.savefig(filename)
        plt.close('all')

xtime = signal.correlation_lags(sw_wiener.shape[0],sw_wiener.shape[0],mode='same')
sw_corr = corrday(sw_wiener)
sw_corrcoef = np.corrcoef(sw_wiener.T,rowvar='False')

plt.figure(figsize=(15,8))
plt.imshow(sw_corrcoef,origin='lower',extent=[day1-1,day2-1,day1-1,day2-1],\
           aspect='auto',cmap='turbo',vmin=-0.0,vmax=1)
plt.colorbar()
#plt.yscale('symlog',linthresh=1e-1,subs=[2,3,4,5,6,7,8,9])
plt.ylim(day1-1,day2-1)
plt.xlim(day1-1,day2-1)
plt.ylabel('days')
plt.xlabel('days')
plt.title('Normalized spectral width with wiener filter: ' + str(win_wien) + 'min (' + str(lfreq) + '-' + str(hfreq) + ' Hz)')
filename = 'Year_CorCoef/' + year +'_bp'+ str(lfreq) + '_' + str(hfreq) + 'Hz_CorrCoef_win' + str(window_duration_sec) + 's_av' + str(average) \
    + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' + str(thres) + 'wienfilt_' + str(win_wien) + 'min.png'
plt.savefig(filename,dpi=300)
filename = 'Year_Figures/' + year+'_bp'+ str(lfreq) + '_' + str(hfreq) + 'Hz_CorrCoef_win' + str(window_duration_sec) + 's_av' + str(average) + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' \
    + str(thres) + 'wienfilt_' + str(win_wien) + 'min.pdf'
plt.savefig(filename)
plt.close('all')


sw_dict = {}

sw_dict['wiener'] = sw_wiener
sw_dict['median'] = sw_median
sw_dict['norm'] = sw_fin
    
sw_file = year + '_SW_bp' + str(lfreq) + '_' + str(hfreq) + '_win_' + str(window_duration_sec) + '_av_' + str(average) + '_sm_' + str(smoothF) + '_Hz_thres_' + str(thres) + '.npy'
np.save(sw_file,sw_dict)
    
if False:
    
    fig,ax=plt.subplots(1,1,figsize=(15,10))
    ax.plot(fr_vec,spectral_year[:,Eru_win],label='original sw')
    ax.plot(fr_vec,sw_0[:,Eru_win],label='smooth sw')              
    ax.plot(fr_vec,win_info[Eru_win]['inter_1'](fr_vec),label='Upper envelope')
    ax.plot(fr_vec,win_info[Eru_win]['inter_1'](fr_vec)-sw_0[:,Eru_win],label='Upper envelope - smoothSW')
    ax.plot(fr_vec,win_info[Eru_win]['inter_2'](fr_vec),label='New upper envelope')
    ax.plot(fr_vec,sw_fin[:,Eru_win],label='normalized smoothSW')
    ax.legend(loc='lower right')
    plt.ylabel('SW')
    ax.set_title( 'Eruption (day 267)')
    ax.set_xlabel('frequency Hz')
    filename = 'wdur' + str(window_duration_sec) + 's_av' + str(average) + '_sm_' + str(smoothF) +'_thres_' + str(thres) + '_eruption.png'
    fig.savefig(filename,dpi=400)
    filename = 'Figures/wdur' + str(window_duration_sec) + 's_av' + str(average)  + '_sm_' + str(smoothF) + '_thres_' + str(thres) + '_eruption.pdf'
    fig.savefig(filename)
    
    fig,ax=plt.subplots(1,1,figsize=(15,10))
    
    ax.plot(fr_vec,spectral_year[:,Noeru_win],label='original sw')
    ax.plot(fr_vec,sw_0[:,Noeru_win],label='smooth sw')              
    ax.plot(fr_vec,win_info[Noeru_win]['inter_1'](fr_vec),label='Upper envelope')
    ax.plot(fr_vec,win_info[Noeru_win]['inter_1'](fr_vec)-sw_0[:,Noeru_win],label='Upper envelope - smoothSW')
    ax.plot(fr_vec,win_info[Noeru_win]['inter_2'](fr_vec),label='New upper envelope')
    ax.plot(fr_vec,sw_fin[:,Noeru_win],label='normalized smoothSW')
    ax.legend(loc='lower right')
    plt.ylabel('SW')
    ax.set_xlabel('frequency Hz')
    ax.set_title('No eruption but signal (day 100)')
    filename = 'wdur' + str(window_duration_sec) + 's_av' + str(average) + '_sm_' + str(smoothF) + '_thres_' + str(thres) + '_noerup.png'
    fig.savefig(filename,dpi=400)
    filename = 'Figures/wdur' + str(window_duration_sec) + 's_av' + str(average)  + '_sm_' + str(smoothF) +'_thres_' + str(thres) + '_noerup.pdf'
    fig.savefig(filename)
    
    fig,ax=plt.subplots(1,1,figsize=(15,10))
    ax.plot(fr_vec,spectral_year[:,Nosig_win],label='original sw')
    ax.plot(fr_vec,sw_0[:,Nosig_win],label='smooth sw')              
    ax.plot(fr_vec,win_info[Nosig_win]['inter_1'](fr_vec),label='Upper envelope')
    ax.plot(fr_vec,win_info[Nosig_win]['inter_1'](fr_vec)-sw_0[:,Nosig_win],label='Upper envelope - smoothSW')
    ax.plot(fr_vec,win_info[Nosig_win]['inter_2'](fr_vec),label='New upper envelope')
    ax.plot(fr_vec,sw_fin[:,Nosig_win],label='normalized smoothSW')
    ax.legend(loc='lower right')
    plt.ylabel('SW')
    ax.set_title('No emerging signals (day 311.22)')
    ax.set_xlabel('frequency Hz')
    filename = 'wdur' + str(window_duration_sec) + 's_av' + str(average) + '_sm_' + str(smoothF) +'_thres_' + str(thres) + 'nolines.png'
    fig.savefig(filename,dpi=400)
    filename = 'Figures/wdur' + str(window_duration_sec) + 's_av' + str(average)  + '_sm_' + str(smoothF) + '_thres_' + str(thres) + 'nolines.pdf'
    fig.savefig(filename)
    
    
    lines_dict = {}
    peaks, _ = signal.find_peaks(sw_fin[:,Eru_win],height=0.8)
    lines_dict['eruption'] = len(peaks)
    peaks, _ = signal.find_peaks(sw_fin[:,Noeru_win],height=0.8)
    lines_dict['day_100'] = len(peaks)
    peaks, _ = signal.find_peaks(sw_fin[:,Nosig_win],height=0.8)
    lines_dict['day_311'] = len(peaks)
    lines_dict['win_dur'] = window_duration_sec
    lines_dict['average'] = average
    lines_dict['smoothing'] = smoothF
    lines_dict['threshold'] = thres
    
    line_file = 'lines_win_' + str(window_duration_sec) + '_av_' + str(average) + '_sm_' + str(smoothF) + '_Hz_thres_' + str(thres) + '.npy'
    np.save(line_file,lines_dict)
    
    

