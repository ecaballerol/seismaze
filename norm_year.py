# import covseisnet as csn

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib import ticker
from scipy import signal
from scipy.signal import wiener
import datetime

import normalization as swm
from Arguments import *
plt.rcParams.update({'font.size': 14})

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


spectral_year_plot = spectral_year[lf_idx:hf_idx,:]
n_frequencies,n_times = spectral_year_plot.shape
times = np.linspace(0, 1, n_times) * delta_days + day1

epoch = datetime.datetime(int(year),1,1,0)
months = []
for itimes in times:
    # time_mon = epoch+timedelta(days=itimes)
    months.append(epoch+datetime.timedelta(days=itimes-1))
#time_mon = np.array(time_mon,dtype='datetime64')
months = mdates.date2num(months)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#SMOOTHING
for smoothF in ismooth:
    sw_year.normalization(smoothF)
    
    
    #Spectral year original
    title = 'Spectral width:' + year
    filename = 'Year_Norm/' + year + '_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) + '_erup.png'
    swm.plot_swmat(spectral_year_plot,months,kw_dict,title=title,filename=filename,figsize=(12,5),labels='months')
    filename = 'Year_Figures/' + year + '_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) + '_erup.pdf'
    swm.plot_swmat(spectral_year_plot,months,kw_dict,title=title,filename=filename,figsize=(13,5),labels='months')
    #Spectral year smoothed
    title = 'Smoothed Spectral width:' + year + ' smoothed: ' + str(smoothF) + 'Hz'
    filename = 'Year_Norm/' + year + '_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) + '_smooth_' + str(smoothF) + 'Hz_erup.png'
    swm.plot_swmat(sw_year.sw_smooth,months,kw_dict,title=title,filename=filename,figsize=(12,5),labels='months')
    filename = 'Year_Figures/' + year + '_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) + '_smooth_' + str(smoothF) + 'Hz_erup.pdf'
    swm.plot_swmat(sw_year.sw_smooth,months,kw_dict,title=title,filename=filename,figsize=(12,5),labels='months')
    #Spectral year normalized
    title = 'Normalized spectral width, smooth:' + str(smoothF) + 'Hz'
    filename = 'Year_Norm/' + year + '_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average)  \
        + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' + str(thres) + '.png'
    swm.plot_swmat(sw_year.sw_norm,months,kw_dict,title=title,filename=filename,cmap='viridis',figsize=(12,5),labels='months')
    filename = 'Year_Figures/' + year + '_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average)  \
        + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' + str(thres) + '.pdf'
    swm.plot_swmat(sw_year.sw_norm,months,kw_dict,title=title,filename=filename,cmap='viridis',figsize=(12,5),labels='months')
    
    plt.close('all')

    sw_median = sw_year.sw_median(win_medfilt)

    title='Normalized spectral width with median filter: ' + str(win_medfilt) + 'min'
    filename = 'Year_Norm/' + year +'_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + \
         str(window_duration_sec) + 's_av' + str(average) \
        + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' + str(thres) + 'medfilt_' + str(win_medfilt) + 'm.png'
    swm.plot_swmat(sw_median,months,kw_dict,title=title,filename=filename,cmap='viridis',figsize=(12,5),labels='months')
    filename = 'Year_Figures/' + year + '_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' \
        + str(thres) + 'medfilt_' + str(win_medfilt) + 'm.pdf'
    swm.plot_swmat(sw_median,months,kw_dict,title=title,filename=filename,cmap='viridis',figsize=(12,5),labels='months')

    
    for win_wien in wien_lens:
       
        wiennum = win_wien*60 / DelT +1
        sw_wiener = wiener(sw_year.sw_norm,(1,int(wiennum)))

        title='Normalized spectral width with wiener filter: ' + str(win_wien) + 'min'
        filename = 'Year_Norm/' + year + '_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) \
            + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' + str(thres) + 'wienfilt_' + str(win_wien) + 'm.png'
        swm.plot_swmat(sw_wiener,months,kw_dict,title=title,filename=filename,cmap='viridis',figsize=(12,5),labels='months')
        filename = 'Year_Figures/' + year + '_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_win' + str(window_duration_sec) + 's_av' + str(average) \
            + '_sm_' + str(smoothF) + 'Hz_norm' +'_thres_' + str(thres) + 'wienfilt_' + str(win_wien) + 'm.pdf'
        swm.plot_swmat(sw_wiener,months,kw_dict,title=title,filename=filename,cmap='viridis',figsize=(12,5),labels='months')
        

plt.close('all')

#Correlation coefficient Matrix
sw_corrcoef = np.corrcoef(sw_wiener.T,rowvar='False')

plt.figure(figsize=(9,8))
plt.imshow(sw_corrcoef,origin='lower',extent=[day1-1,day2-1,day1-1,day2-1],\
           aspect='equal',cmap='turbo',vmin=-0.0,vmax=1)
plt.colorbar()
#plt.yscale('symlog',linthresh=1e-1,subs=[2,3,4,5,6,7,8,9])
plt.ylim(day1-1,day2-1)
plt.xlim(day1-1,day2-1)
plt.ylabel('days')
plt.xlabel('days')
plt.title('Correlation Coefficient Matrix: ('  + str(lowfreq) + '-' + str(hfreq) + ' Hz)')
filename = 'Year_Norm/' + year +'_bp'+ str(lowfreq) + '_' + str(hfreq) + 'Hz_CorrCoef_win' + str(window_duration_sec) + 's_av' + str(average) \
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
    
if True:
    
    fig,ax=plt.subplots(1,1,figsize=(10,7))
    ax.plot(fr_vec,sw_year.spectral_year[lf_idx:hf_idx,Eru_win],label='original sw')
    ax.plot(fr_vec,sw_year.sw_smooth[:,Eru_win],label='smooth sw')              
    ax.plot(fr_vec, sw_year.envelope[Eru_win]['inter_1'](fr_vec),label='Upper envelope')
    ax.plot(fr_vec,sw_year.envelope[Eru_win]['inter_1'](fr_vec)-sw_year.sw_smooth[:,Eru_win],label='Upper envelope - smoothSW')
    ax.plot(fr_vec,sw_year.envelope[Eru_win]['inter_2'](fr_vec),label='New upper envelope')
    ax.plot(fr_vec,sw_year.sw_norm[:,Eru_win],label='normalized smoothSW')
    ax.legend(loc='lower right')
    plt.ylabel('SW')
    ax.set_title( 'Eruption (day 267)')
    ax.set_xlabel('frequency Hz')
    filename = 'wdur' + str(window_duration_sec) + 's_av' + str(average) + '_sm_' + str(smoothF) +'_thres_' + str(thres) + '_eruption.png'
    fig.savefig(filename,dpi=400)
    filename = 'Figures/wdur' + str(window_duration_sec) + 's_av' + str(average)  + '_sm_' + str(smoothF) + '_thres_' + str(thres) + '_eruption.pdf'
    fig.savefig(filename)
    
    # fig,ax=plt.subplots(1,1,figsize=(15,10))
    
    # ax.plot(fr_vec,spectral_year[:,Noeru_win],label='original sw')
    # ax.plot(fr_vec,sw_0[:,Noeru_win],label='smooth sw')              
    # ax.plot(fr_vec,win_info[Noeru_win]['inter_1'](fr_vec),label='Upper envelope')
    # ax.plot(fr_vec,win_info[Noeru_win]['inter_1'](fr_vec)-sw_0[:,Noeru_win],label='Upper envelope - smoothSW')
    # ax.plot(fr_vec,win_info[Noeru_win]['inter_2'](fr_vec),label='New upper envelope')
    # ax.plot(fr_vec,sw_fin[:,Noeru_win],label='normalized smoothSW')
    # ax.legend(loc='lower right')
    # plt.ylabel('SW')
    # ax.set_xlabel('frequency Hz')
    # ax.set_title('No eruption but signal (day 100)')
    # filename = 'wdur' + str(window_duration_sec) + 's_av' + str(average) + '_sm_' + str(smoothF) + '_thres_' + str(thres) + '_noerup.png'
    # fig.savefig(filename,dpi=400)
    # filename = 'Figures/wdur' + str(window_duration_sec) + 's_av' + str(average)  + '_sm_' + str(smoothF) +'_thres_' + str(thres) + '_noerup.pdf'
    # fig.savefig(filename)
    
    # fig,ax=plt.subplots(1,1,figsize=(15,10))
    # ax.plot(fr_vec,spectral_year[:,Nosig_win],label='original sw')
    # ax.plot(fr_vec,sw_0[:,Nosig_win],label='smooth sw')              
    # ax.plot(fr_vec,win_info[Nosig_win]['inter_1'](fr_vec),label='Upper envelope')
    # ax.plot(fr_vec,win_info[Nosig_win]['inter_1'](fr_vec)-sw_0[:,Nosig_win],label='Upper envelope - smoothSW')
    # ax.plot(fr_vec,win_info[Nosig_win]['inter_2'](fr_vec),label='New upper envelope')
    # ax.plot(fr_vec,sw_fin[:,Nosig_win],label='normalized smoothSW')
    # ax.legend(loc='lower right')
    # plt.ylabel('SW')
    # ax.set_title('No emerging signals (day 311.22)')
    # ax.set_xlabel('frequency Hz')
    # filename = 'wdur' + str(window_duration_sec) + 's_av' + str(average) + '_sm_' + str(smoothF) +'_thres_' + str(thres) + 'nolines.png'
    # fig.savefig(filename,dpi=400)
    # filename = 'Figures/wdur' + str(window_duration_sec) + 's_av' + str(average)  + '_sm_' + str(smoothF) + '_thres_' + str(thres) + 'nolines.pdf'
    # fig.savefig(filename)
    


