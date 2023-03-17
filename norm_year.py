# import covseisnet as csn
import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from scipy import signal
from scipy.signal import wiener

from Arguments import *
import normalization as swm


sw_year = swm.swnorm(ODIR,kw_dict)
#READING THE DATA 
    
spectral_year= sw_year.readSW()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Windows to analyze
Eru_day =  267 #in days
Eru_win = Eru_day * int(Winperday)
Noerup = 100
Noeru_win = Noerup * int(Winperday)
Nosig_day = 311.22
Nosig_win = int(np.round(Nosig_day * Winperday))



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#SMOOTHING
for smoothF in ismooth:
    sw_year.normalization(smoothF)
    
    spectral_year_plot = spectral_year[lf_idx:hf_idx,:]
    n_frequencies,n_times = spectral_year_plot.shape
    times = np.linspace(0, 1, n_times) * delta_days
    
    fig, ax = plt.subplots(1, figsize=(11, 6))
    img = ax.pcolorfast(times, fr_vec, spectral_year_plot, rasterized=True, cmap="RdBu")
    ax.set_xlabel('days')
    ax.set_ylabel('Frequency, Hz')
    ax.set_yscale('symlog',linthresh=1e-1,subs=[2,3,4,5,6,7,8,9,10])
    ax.set_yticks(np.arange(11))
    ax.set_ylim(lowfreq,hfreq)
    plt.colorbar(img, ax=ax)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.title('spectral width:' + year)
    filename = 'Year/' + year + '_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) + '_erup.png'
    plt.savefig(filename,dpi=200)
    filename = 'Year_Figures/' + year + '_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) + '_erup.pdf'
    plt.savefig(filename)
    
    
    fig, ax = plt.subplots(1, figsize=(11, 6))
    ax.imshow(sw_year.sw_smooth,origin='lower',extent=[day1-1,day2-1,lowfreq,hfreq],aspect='auto',cmap='RdBu')
    plt.colorbar(img,ax=ax)
    ax.set_yscale('symlog',linthresh=1e-1,subs=[2,3,4,5,6,7,8,9])
    ax.set_yticks(np.arange(11))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_ylim(lowfreq,hfreq)
    ax.set_xlabel('days')
    ax.set_ylabel('Frequency, Hz')
    plt.title('smoothed spectral width: ' + str(smoothF) + 'Hz' )
    filename = 'Year_Smooth/' + year + '_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) + '_smooth_' + str(smoothF) + 'Hz_erup.png'
    plt.savefig(filename,dpi=200)
    filename = 'Year_Figures/' + year + '_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) + '_smooth_' + str(smoothF) + 'Hz_erup.pdf'
    plt.savefig(filename)
    
    fig, ax = plt.subplots(1, figsize=(11, 6))
    img = ax.imshow(sw_year.sw_norm,origin='lower',extent=[day1-1,day2-1,lowfreq,hfreq],aspect='auto',cmap='viridis')
    # img = ax.pcolorfast(times, fr_vec, sw_fin, rasterized=True, cmap="viridis")
    ax.set_xlabel('days')
    ax.set_ylabel('Frequency, Hz')
    ax.set_yscale('symlog',linthresh=1e-1,subs=[2,3,4,5,6,7,8,9,10])
    ax.set_yticks(np.arange(11))
    ax.set_ylim(lowfreq,hfreq)
    plt.colorbar(img, ax=ax)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.title('Normalized spectral width, smooth:' + str(smoothF) + 'Hz')
    filename = 'Year_SW_spec/' + year + '_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average)  \
        + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' + str(thres) + '.png'
    plt.savefig(filename,dpi=400)
    filename = 'Year_Figures/' + year +'_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) \
        + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' + str(thres) + '.pdf'
    plt.savefig(filename)
    plt.close('all')

    win_medfilt = 360 #in minutes
    sw_median = sw_year.sw_median(win_medfilt)
    
    plt.figure(figsize=(15,8))
    plt.imshow(sw_median,origin='lower',extent=[day1-1,day2-1,lowfreq,hfreq],aspect='auto',cmap='viridis',vmin=0,vmax=1)
    plt.colorbar()
    plt.yscale('symlog',linthresh=1e-1,subs=[2,3,4,5,6,7,8,9])
    plt.ylim(lowfreq,hfreq)
    plt.xlim(1,345)
    plt.ylabel('Frequency Hz')
    plt.xlabel('days')
    plt.title('Normalized spectral width with median filter: ' + str(win_medfilt) + 'min')
    filename = 'Year_Median/' + year +'_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + \
         str(window_duration_sec) + 's_av' + str(average) \
        + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' + str(thres) + 'medfilt_' + str(win_medfilt) + 'm.png'
    plt.savefig(filename,dpi=300)
    filename = 'Year_Figures/' + year + '_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' \
        + str(thres) + 'medfilt_' + str(win_medfilt) + 'm.pdf'
    plt.savefig(filename)
    plt.close('all')

    #Wienner filter
    wien_lens = [360]
    for win_wien in wien_lens:
        #win_wien =  720 #In minutes
        wiennum = win_wien*60 / DelT +1
        # sw_wiener = wiener(sw_fin)
        sw_wiener = wiener(sw_year.sw_norm,(1,int(wiennum)))
        plt.figure(figsize=(15,8))
        plt.imshow(sw_wiener,origin='lower',extent=[day1-1,day2-1,lowfreq,hfreq],aspect='auto',cmap='viridis',vmin=0,vmax=1)
        plt.colorbar()
        plt.yscale('symlog',linthresh=1e-1,subs=[2,3,4,5,6,7,8,9])
        plt.ylim(lowfreq,hfreq)
        plt.xlim(1,345)
        plt.ylabel('Frequency Hz')
        plt.xlabel('days')
        plt.title('Normalized spectral width with wiener filter: ' + str(win_wien) + 'min')
        filename = 'Year_Wiener/' + year + '_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) \
            + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' + str(thres) + 'wienfilt_' + str(win_wien) + 'm.png'
        plt.savefig(filename,dpi=300)
        filename = 'Year_Figures/' + year + '_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' \
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
plt.title('Normalized spectral width with wiener filter: ' + str(win_wien) + 'min (' + str(lowfreq) + '-' + str(hfreq) + ' Hz)')
filename = 'Year_CorCoef/' + year +'_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_CorrCoef_win' + str(window_duration_sec) + 's_av' + str(average) \
    + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' + str(thres) + 'wienfilt_' + str(win_wien) + 'min.png'
plt.savefig(filename,dpi=300)
filename = 'Year_Figures/' + year+'_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_CorrCoef_win' + str(window_duration_sec) + 's_av' + str(average) + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' \
    + str(thres) + 'wienfilt_' + str(win_wien) + 'min.pdf'
plt.savefig(filename)
plt.close('all')


sw_dict = {}

sw_dict['wiener'] = sw_wiener
sw_dict['median'] = sw_median
sw_dict['norm'] = sw_year.sw_norm
    
sw_file = year + '_SW_bp' + str(lowfreq) + '_' + str(hfreq) + '_win_' + str(window_duration_sec) + '_av_' + str(average) + '_sm_' + str(smoothF) + '_Hz_thres_' + str(thres) + '.npy'
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
    
    

