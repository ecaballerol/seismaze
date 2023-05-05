#------------------------ importing basic packages
import copy
import glob

import matplotlib.pyplot as plt
import numpy as np
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
        excluding = []
        for iclus in np.arange(self.NumClusters):
            self.nclust[iclus] = {}
            CC_stack = []
            for i in np.arange(self.stack_len,sw_corrtmp.shape[1]-self.stack_len):
                CC_stack.append(np.sum(sw_corrtmp[i,i-self.stack_len:i+self.stack_len]))
            
            CC_stack = np.array(CC_stack)
            max_stack = np.argmax(CC_stack)+ self.stack_len
            idx_clus = np.where(sw_corrtmp[max_stack,:]>self.CC_thres)[0]
            excluding.extend(idx_clus)
            sw_corrtmp[idx_clus,:]=0
            sw_corrtmp[:,idx_clus]=0
            #Iterative step 2 of Soubestre et al. 2018
            idx_iter = copy.copy(idx_clus)
            for iter in np.arange(4):
                cc_whole = copy.deepcopy(self.CC)
                CC_cluster = []
                for count, idxCC in enumerate(idx_iter):
                    CC_cluster.append(np.sum(cc_whole[idxCC,idx_iter]))
                CC_cluster = np.array(CC_cluster)
                max_tmp = np.argmax(CC_cluster)
                idx_tmp = np.where(cc_whole[idx_iter[max_tmp],:]>self.CC_thres)[0]
                #print('For cluster: ',iclus, 'index: ',idx_iter[max_tmp])
                idx_iter = idx_tmp
            self.nclust[iclus]['CCstack'] = CC_stack
            self.nclust[iclus]['maxstack'] = max_stack
            self.nclust[iclus]['idx_clus'] = idx_clus
            self.nclust[iclus]['sw_tem'] = np.mean(self.SW[:,idx_clus],axis=1)
            self.nclust[iclus]['CC_cluster'] = CC_cluster
            self.nclust[iclus]['idx_iter'] = idx_iter
            self.nclust[iclus]['CC_tem'] = np.mean(self.SW[:,idx_iter],axis=1)

        return

    



def plot_CC(CC,times,plot_dict,title,filename,labels='days',cmap='RdBu'):
    fig, ax = plt.subplots(1, figsize=(9, 8))
    img = ax.pcolorfast(times, times, CC, rasterized=False, cmap=cmap,vmin=0)
    ax.set_xlabel(labels)
    ax.set_ylabel(labels)
    # ax.set_yscale('symlog',linthresh=1e-1,subs=[2,3,4,5,6,7,8,9,10])
    # ax.set_yticks(np.arange(11))
    # ax.set_ylim(plot_dict['lowfreq'],plot_dict['hfreq'])
    plt.colorbar(img, ax=ax)
    ax.set_aspect('equal')
    fig.tight_layout()
    #ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_title(title)
    if filename[-3:]=='png':
        plt.savefig(filename,dpi=300)
    else:
        plt.savefig(filename)