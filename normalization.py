#------------------------ importing basic packages
import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

plt.close("all")


class swnorm(object):
    '''
    Class that deals with the readin of spectral width 
    and with the normalization processing
    
    '''
#--------------------------- A FEW FUNCTIONS -----------
    def __init__(self,SWDIR,year_dict):
        self.SWDIR = SWDIR
        self.year_dict = year_dict
        
    def readSW(self):
        '''
        Function to read the spectral width from directory
        '''

        spectral_year = []
        for d in range(self.year_dict['day1'],self.year_dict['day2']):
            print(d)
            doy = f'{d:03}'
            doy = str(doy)
            SW = self.SWDIR +'/SW_' + self.year_dict['year'] + '_' +doy+ '_*'
            SW_name  = glob.glob(SW)[0]
            SWtmp = np.load(SW_name)
            spectral_year.extend(SWtmp)
        spectral_year = np.array(spectral_year).T

        self.spectral_year = spectral_year
        return spectral_year

    #-----------------------------------------------
    # function to smooth signal
    #-----------------------------------------------
    def smooth1d(x,window_len=11,window='hanning'):
        """smooth the data using a window with requested size.
        
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.
        s
        input:
            x: the input signal 
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        output:
            the smoothed signal
            
        example:

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)
        
        see also: 
        
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
    
        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """

        if x.ndim != 1:
            print("smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            print("Input vector needs to be bigger than window size.")


        if window_len<3:
            return x


        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            print("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),x,mode='same')
        return y


    def env_sup (frec,sign_vec,hf_idx,lf_idx):
        '''This function calculate the upper envelope of a 1d signal
        

        Parameters
        ----------
        frec : 1d array
            Vector of frequencies.
        signal : 1d array
            Vector of the signal for which we calculate the enveloppe.

        Returns
        -------
        Interpolating function.

        '''
        
        idx_max = argrelextrema(sign_vec, np.greater)
        
        xmax = frec[idx_max[0]]
        ymax = sign_vec[idx_max[0]]

        nmax = np.size(xmax)
        xxmax = np.zeros(nmax+2)
        yymax = np.zeros(nmax+2)

        xxmax[1:nmax+1] = xmax
        xxmax[0] = frec[0]
    #    xxmax[nmax+1] = frec[int(np.ceil(npts/2))-1-lf_idx]
        xxmax[nmax+1] = frec[hf_idx-lf_idx-1]

        yymax[1:nmax+1] = ymax
        yymax[0] = ymax[0]
        yymax[nmax+1] = ymax[nmax-1]
        
        
        fav = interp1d(xxmax, yymax,kind='cubic')
        
        return fav

    def medfilt_days(sw, nu_medw=31):
        '''
        Parameters
        ----------
        sw : array 2d
            Covariance matrix with the spectral width.
        nu_medw: int
            Number of samples to make the median filter
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        sw_medfilt = ndimage.median_filter(sw,size=(1,int(nu_medw)))
        return  sw_medfilt


def corrday(matrixcor, mode='same'):
    lenma = matrixcor.shape
    corrmatr = np.zeros((lenma[0],lenma[1]-1))
    for it in np.arange(matrixcor.shape[1]-1):
        tmp1 = matrixcor[:,it] - np.mean(matrixcor[:,it])
        tmp2 = matrixcor[:,it+1] - np.mean(matrixcor[:,it+1])
        cortmp = signal.correlate(tmp1,tmp2,mode=mode)
        #cortmp /= cortmp.max() 
        tmp = signal.detrend(cortmp,bp=int(lenma[0]/2))
        corrmatr[:,it] = cortmp
    return corrmatr


def pearson_2d(x, Y):
    """ Pearson product-moment correlation of vectors `x` and array `Y`
    Parameters
    ----------
    x : array shape (N,)
        One-dimensional array to correlate with every column of `Y`
    Y : array shape (N, P)
        2D array where we correlate each column of `Y` with `x`.
    Returns
    -------
    r_xy : array shape (P,)
        Pearson product-moment correlation of vectors `x` and the columns of
        `Y`, with one correlation value for every column of `Y`.
    """
    # Mean-center x -> mc_x
    # Mean-center every column of Y -> mc_Y
    # a : Get sum of products of mc_x and every column of mc_Y
    # b : Get sum of products of mc_x on mc_x
    # c : Get sum of products of every column of mc_Y[:, i] on itself
    # return a / (sqrt(b) * sqrt(c))
    # +++your code here+++
    # LAB(begin solution)
    mc_x = x - np.mean(x)
    mc_Y = Y - np.mean(Y, axis=0)  # This is numpy broadcasting
    # You could also do the step above with:
    # mc_Y = Y - np.tile(np.mean(Y, axis=0), (len(x), 1))
    a = mc_x.dot(mc_Y)
    b = mc_x.dot(mc_x)
    c = np.sum(mc_Y ** 2, axis=0)
    return a / (np.sqrt(b) * np.sqrt(c))
    # LAB(replace solution)
    # return
    # LAB(end solution)
    
    
# #--------------------------------------------------------------

# #------------------- frequencies
# nfreq = int(1e5)

# f = .01*np.arange(nfreq) + 100


# #------------------------- noise
# nnoise = int(nfreq+2/100)

# fnoise = np.arange(nnoise)

# noise1 = .15*(np.random.rand(nnoise)-.5)

# fn = interp1d(fnoise, noise1)

# noise = fn(f)


# #------------------------------------------------------- synthetic spectral line maxima distribution
# sw0 = np.sin(f**.7 + np.sin(.12*f))**2*np.cos(.075*f)**2


# sw_av = smooth1d(sw0,3000)

# idx_max = argrelextrema(sw_av, np.greater)

# xmax = f[idx_max]
# ymax = sw_av[idx_max]

# nmax = np.size(xmax)
# xxmax = np.zeros(nmax+2)
# yymax = np.zeros(nmax+2)

# xxmax[1:nmax+1] = xmax
# xxmax[0] = f[0]
# xxmax[nmax+1] = f[nfreq-1]

# yymax[1:nmax+1] = ymax
# yymax[0] = ymax[0]
# yymax[nmax+1] = ymax[nmax-1]


# #------------------------ averaging cleaning and normalizing 
# idx = sw_av<.1
# sw1 = sw_av
# sw1[idx] = 0

# fav = interp1d(xxmax, yymax)

# sw0norm = sw1/fav(f)
# idx = sw0norm>1
# sw0norm[idx]=1



# #-------------
# plt.figure()

# plt.title('initial spectral lines')

# plt.plot(f,sw0)

# plt.xlabel('frequency')
# plt.show()


# #-------------
# plt.figure()

# plt.title('smmothed and normalized initial spectral lines')

# plt.plot(f,sw0norm)

# plt.xlabel('frequency')
# plt.show()


# #-------------
# plt.figure()

# plt.title('niose')

# plt.plot(f,noise)

# plt.xlabel('frequency')
# plt.show()



# #--------------------------------------- raw synthetic spectrl width
# sw = 1-sw0 + np.cos(.013*f) + 2 + noise



# #-------------
# plt.figure()

# plt.title('synthetic spectral width')

# plt.plot(f,sw)

# plt.xlabel('frequency')
# plt.show()






# #---------------------------------------- averaging 

# sw_av = smooth1d(sw,3000)

# idx_max = argrelextrema(sw_av, np.greater)

# xmax = f[idx_max]
# ymax = sw_av[idx_max]

# nmax = np.size(xmax)
# xxmax = np.zeros(nmax+2)
# yymax = np.zeros(nmax+2)

# xxmax[1:nmax+1] = xmax
# xxmax[0] = f[0]
# xxmax[nmax+1] = f[nfreq-1]

# yymax[1:nmax+1] = ymax
# yymax[0] = ymax[0]
# yymax[nmax+1] = ymax[nmax-1]




# #-------------
# plt.figure()

# plt.title('smoothed spectral width and upper envelope')

# plt.plot(f,sw_av)
# plt.plot(xxmax,yymax)

# plt.xlabel('frequency')
# plt.show()


# fav = interp1d(xxmax, yymax, kind='cubic')


# #----------------- removing upper envelope
# sw3 = fav(f)-sw_av


# idx_max2 = argrelextrema(sw3, np.greater)
# xmax = f[idx_max2]
# ymax = sw3[idx_max2]

# nmax = np.size(xmax)
# xxmax = np.zeros(nmax+2)
# yymax = np.zeros(nmax+2)

# xxmax[1:nmax+1] = xmax
# xxmax[0] = f[0]
# xxmax[nmax+1] = f[nfreq-1]

# yymax[1:nmax+1] = ymax
# yymax[0] = ymax[0]
# yymax[nmax+1] = ymax[nmax-1]


# #-------------
# plt.figure()

# plt.title('removed upper envelope')

# plt.plot(f,sw3)
# plt.plot(xxmax,yymax)

# plt.xlabel('frequency')
# plt.show()


# idx = sw3<.1
# sw3[idx] = 0

# fav2 = interp1d(xxmax, yymax)

# sw3norm = sw3/fav2(f)
# idx = sw3norm>1
# sw3norm[idx]=1




# #-------------
# plt.figure()

# plt.title('comparizon of input and output')

# plt.plot(f,sw0)
# plt.plot(f,sw3)

# plt.xlabel('frequency')
# plt.show()


# #-------------
# plt.figure()

# plt.title('normalized output')

# plt.plot(f,sw3norm)

# plt.xlabel('frequency')
# plt.show()


# #-------------
# plt.figure()

# plt.title('comparizon of normalized input and output')

# plt.plot(f,sw0norm)
# plt.plot(f,sw3norm)

# plt.xlabel('frequency')
# plt.show()



