#------------------------ importing basic packages
import copy
import glob

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib import ticker
from scipy import ndimage, signal
from scipy.interpolate import interp1d
from scipy.linalg import eigh, eigvals
from scipy.signal import argrelextrema

plt.close("all")


class swcorr(object):
    '''
    Class that deals with the readin of spectral width 
    and with the normalization processing
    
    '''
#--------------------------- A FEW FUNCTIONS -----------
    def __init__(self,SW,sw_dict):
        self.SW = SW
        self.sw_dict = sw_dict
        self.CC_thres = None
        self.stack_len = None

    def correlationSW(self):
        self.CC = np.corrcoef(self.SW.T,rowvar='False')
        return self.CC
    
    def calcEigVal(self,key='Corr'):
        if key == 'Corr':
            self.eigenvalues = np.abs(eigvals(self.CC))
        return self.eigenvalues
    
    def defineClusters(self,numClust=None,thres=0.2):
        if numClust is not None:
            self.NumClusters = numClust
        else:
            self.NumClusters = np.argwhere(self.eigenvalues/self.eigenvalues.max()<thres)[0][0]

    def ClusterCal(self):
        self.nclust = {}
        sw_corrtmp = copy.deepcopy(self.CC)
        central_idx = []
        for iclus in np.arange(self.NumClusters):
            self.nclust[iclus] = {}
            CC_stack = []
            for i in np.arange(self.stack_len,sw_corrtmp.shape[1]-self.stack_len):
                CC_stack.append(np.sum(sw_corrtmp[i,i-self.stack_len:i+self.stack_len]))
            
            CC_stack = np.array(CC_stack)
            max_stack = np.argmax(CC_stack)+ self.stack_len
            idx_clus = np.where(sw_corrtmp[max_stack,:]>self.CC_thres)[0]
            sw_corrtmp[idx_clus,:]=0
            sw_corrtmp[:,idx_clus]=0

            self.nclust[iclus]['CCstack'] = CC_stack
            self.nclust[iclus]['maxstack'] = max_stack
            self.nclust[iclus]['idx_clus'] = idx_clus
            self.nclust[iclus]['sw_tem'] = np.mean(self.SW[:,idx_clus],axis=1)
            # print(max_stack)

        #Iterative step 2 of Soubestre et al. 2018
        cc_whole = copy.deepcopy(self.CC)
        for iclus in np.arange(self.NumClusters):    
            CC_cluster = []
            #Stacking over the seleceted windows
            idx_iter = copy.deepcopy(self.nclust[iclus]['idx_clus'])
            for count, idxCC in enumerate(idx_iter):
                CC_cluster.append(np.sum(cc_whole[idxCC,idx_iter])) 
            CC_cluster = np.array(CC_cluster)
            max_tmp = np.argmax(CC_cluster)
            max_idx = idx_iter[max_tmp]            
            self.nclust[iclus]['maxfinal'] = max_idx
            self.nclust[iclus]['idx_final']=[]
            central_idx.append(max_idx)

        for iday in np.arange(self.CC.shape[1]):
            tmp = cc_whole[iday,central_idx]
            cluster_final = np.argmax(tmp)
            if cc_whole[iday,central_idx[cluster_final]] >self.CC_thres:
                self.nclust[cluster_final]['idx_final'].append(iday)

        #Conver to array, sort the index, and calculate the spectral \
        # width template
        erase_list = []
        for iclus in np.arange(self.NumClusters):
            self.nclust[iclus]['idx_final'] = np.sort(np.array(self.nclust[iclus]['idx_final']))
            if  self.nclust[iclus]['idx_final'].size:
                self.nclust[iclus]['SW_final'] = np.mean(self.SW[:,self.nclust[iclus]['idx_final']],axis=1)
            else:
                print('Not elements in cluster: ' + str(iclus) + ' because similarity' )
                print('Deleting')
                self.nclust[iclus]['SW_final'] = self.nclust[iclus]['sw_tem']
                erase_list.append(iclus)
        if erase_list:
            for ic in erase_list:
                del self.nclust[ic]
                self.NumClusters -=1

        clust_tmp= {}
        date_tmp = []
        for iclus in np.arange(self.NumClusters):
            clust_tmp[iclus] = {}
            clust_tmp[iclus]['idx']= copy.deepcopy(self.nclust[iclus]['idx_final'])
            date_tmp.append([])
        for iter in np.arange(4):
            central_tmp = []
            for iclus in np.arange(self.NumClusters):
                CC_cluster = []
                #Stacking over the seleceted windows
                idx_iter = clust_tmp[iclus]['idx']
                for count, idxCC in enumerate(idx_iter):
                    CC_cluster.append(np.sum(cc_whole[idxCC,idx_iter]))
                CC_cluster = np.array(CC_cluster)
                max_tmp = np.argmax(CC_cluster)
                max_idx = idx_iter[max_tmp]
                central_tmp.append(max_idx)
                #print(max_idx)
            for iday in np.arange(self.CC.shape[1]):
                tmp = cc_whole[iday,central_tmp]
                cluster_final = np.argmax(tmp)
                #if cc_whole[iday,central_tmp[cluster_final]] >self.CC_thres:
                date_tmp[cluster_final].append(iday)            
            for iclus in np.arange(self.NumClusters):                
                self.nclust[iclus]['idx'] = np.array(date_tmp[iclus])
            
        return

    



def plot_CC(CC,times,plot_dict,title,filename,figsize=(9, 8),labels='days',cmap='RdBu',julday=False):
    fig, ax = plt.subplots(1, figsize=figsize)
    img = ax.pcolorfast(times, times, CC, rasterized=False, cmap=cmap,vmin=0)
    ax.set_xlabel(labels)
    ax.set_ylabel(labels)
    if julday == False:
        fmt_month = mdates.MonthLocator()
        fmt_year = mdates.YearLocator()
        ax.xaxis.set_minor_locator(fmt_month)
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(fmt_year)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    # ax.set_yscale('symlog',linthresh=1e-1,subs=[2,3,4,5,6,7,8,9,10])
    # ax.set_yticks(np.arange(11))
    # ax.set_ylim(plot_dict['lowfreq'],plot_dict['hfreq'])
    plt.colorbar(img, ax=ax)
    ax.set_aspect('equal')
    fig.tight_layout()
    #ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_title(title)
    if filename[-3:]=='png':
        plt.savefig(filename,dpi=300,bbox_inches='tight')
    else:
        plt.savefig(filename,bbox_inches='tight')