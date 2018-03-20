import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, kendalltau
import pandas as pd
import os
T = np.transpose
from hashfunctions import *
# os.chdir('/Users/tim/py_code/')
from bloom_filters import * 
import sys
sys.path.append('/Users/tim/py_code/FLY_LSH')
from read_data import *

# import hashlib # might need

max_val_hash = sys.maxsize
import xxhash
import compute_distance_metrics

# get MNIST
# mnist_path = '/Users/tim/py_code/FLY_LSH/mnist10k.txt'
mnist_path = '/home/navlakha/projects/fly_hashing/data/mnist/mnist10k.txt'
mnist = read_generic_data(mnist_path,10000,28*28) # 10 000, 784

def run_distance_analysis(dat_use,n_ex,do_10_fold,sv_nm,dm='euc'):

    ###dat_use = 'MNIST' # Hallem, rand100

    #do_10_fold = True
    # eps = .01 # [.01, 6E-6]

    os.chdir('/home/tsheehan/py_code/fly_dat/') # os.chdir('/Users/tim/py_code/FLY_LSH/fly_dat/')
    dat = pd.read_csv("hallem1.txt",delimiter=' ')
    if dat_use == 'Hallem':
        dat = dat.values
        n_ex = np.shape(dat)[0]
    elif dat_use[:4] == 'rand':
        dIn = int(dat_use[4:])
        dat = np.random.randn(n_ex,dIn) # random data
    elif dat_use == 'MNIST':
        dIn = 28*28
        dat = mnist[:n_ex,:]

    n_odors = np.shape(dat)[0]
    dat_100 = dat/np.mean(dat,axis=0)*100
    dat_100 = np.nan_to_num(dat_100)
    U = dat_100
    #assert(np.shape(U)[0]==l_ex)
    os.chdir('/home/tsheehan/py_code/dist_metrics')
    # Cross-validation
    if do_10_fold:
        n_fold = 10
        f_ind = np.round(np.linspace(0,n_odors,n_fold+1)).astype(int)
        grps = [np.arange(f_ind[i],f_ind[i+1] ) for i in range(n_fold) ] # perfect 

        tag_ind = list()
        query_ind = list()
        for i in range(n_fold):
            this_grp = grps[i]
            a = np.arange(n_odors)
            a = np.delete(a,this_grp)
            if 0:
                ind_probe = [np.random.choice(a,l_ex,replace=False) for i in range(len(this_grp))]
                tag_ind_i = np.concatenate([ind_probe for i in range(reps_per_fold)]) # 1:10, repeated 10 x
                query_ind_i = np.concatenate(T([this_grp for i in range(reps_per_fold)]))

            # we have convinience of all groups being the same size
            ind_probe = a
            #tag_ind_i = np.concatenate([ind_probe for i in range(len(this_grp))])
            tag_ind_i = np.tile(ind_probe,(len(this_grp),1))
            query_ind_i = this_grp

            tag_ind.append(tag_ind_i)
            query_ind.append(query_ind_i)
        else:
            tag_ind = np.array(tag_ind).reshape(n_odors,len(a))
            query_ind = np.array(query_ind).reshape(n_odors)

    else:      # Leave one Odor Out Crossvalidation
        query_ind = list()
        tag_ind = list()
        for i in range(n_ex):
            a = np.arange(n_odors)
            a = np.delete(a,i)
            tag_ind.append(a)
        else:
            query_ind = np.arange(n_ex)
            tag_ind = np.array(tag_ind)

    S = U[tag_ind] # n_ex, l_ex, n_ORN
    q = U[query_ind] # n_ex, n_ORN

    eps_try = np.logspace(np.log10(.5),np.log10(1E-6),15) 
    #dm = 'cos'
    dist_mets = list()
    for ep in eps_try:
        dist_mets.append(compute_distance_metrics.get_distance_metrics(S,q,eps=ep,app_str=dat_use,
                                                                       dist_met=dm))
    mts = {'dist_mets':dist_mets,'eps':eps_try}
    full_save = sv_nm + '_'+ dat_use +'_'+ str(n_ex) +'_'+ dm +'.npy'
    np.save(full_save,mts)