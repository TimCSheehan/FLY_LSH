import numpy as np
import pandas as pd
import os
T = np.transpose
from hashfunctions import *
from bloom_filters import * 
import sys
sys.path.append('/Users/tim/py_code/FLY_LSH')

from multiprocessing import Pool # multithreading baby
import compute_distance_metrics


# # perhaps have m_try depend on n?
n_run = 2
dm = 'both'
def easy_dist(inputs):
    m,k_ratio,S,q,dat_use = inputs
    k = int(m/k_ratio)
    out = list() # just do 1 to test
    for i in range(n_run):
        out.append(compute_distance_metrics.get_distance_metrics(S,q,m=m,k=k,app_str=dat_use,
                                                                   dist_met=dm,text_out=False,ORN_SPECIAL=True,VERBOSE=0))
    out = np.mean(out,axis=0)
    return (out,m,k)


def run_distance_analysis(dat_use,n_ex,nPool=20,do_10_fold=True,dm='both'):

    if dat_use == 'Hallem':
        os.chdir('/home/tsheehan/py_code/fly_dat/') # os.chdir('/Users/tim/py_code/FLY_LSH/fly_dat/')
        dat = pd.read_csv("hallem1.txt",delimiter=' ')
        dat = dat.values
        n_ex = np.shape(dat)[0]
        do_10_fold = False
    elif dat_use[:4] == 'rand':
        dIn = int(dat_use[4:])
        dat = np.random.randn(n_ex,dIn) # random data
    elif dat_use[:5] == 'Erand':
        dIn = int(dat_use[5:])
        scale = 1.0 
        dat = np.random.exponential(scale,(n_ex,dIn)) # random data
    elif dat_use == 'MNIST':
        # get MNIST
        from read_data import read_generic_data
        # mnist_path = '/Users/tim/py_code/FLY_LSH/mnist10k.txt'
        mnist_path = '/home/navlakha/projects/fly_hashing/data/mnist/mnist10k.txt'
        mnist = read_generic_data(mnist_path,10000,28*28) # 10 000, 784
        dIn = 28*28
        dat = mnist[:n_ex,:]

    n_odors = np.shape(dat)[0]
    dat_100 = dat/np.mean(dat,axis=0)*100
    dat_100 = np.nan_to_num(dat_100)
    U = dat_100


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

    # pool = Pool(nPool)
    #n_run = 5
    
    # perhaps have m_try depend on n?
    m_try = (np.logspace(np.log10(200),np.log10(5000),10)).astype(int)
    k_ratio_try = np.logspace(np.log10(5),np.log10(200),10)
    dm = 'both'

    iter_options = ((x,y) for x in m_try for y in k_ratio_try)
    mL = [x for x in m_try for y in k_ratio_try]
    kL = [y for x in m_try for y in k_ratio_try]
    SL = [S for x in m_try for y in k_ratio_try]
    qL = [q for x in m_try for y in k_ratio_try]
    dat_useL = [dat_use for x in m_try for y in k_ratio_try]
    
#     def easy_dist(inputs):
#         m,k_ratio = inputs
#         k = int(m/k_ratio)
#         out = list() # just do 1 to test
#         for i in range(n_run):
#             out.append(compute_distance_metrics.get_distance_metrics(S,q,m=m,k=k,app_str=dat_use,
#                                                                        dist_met=dm,text_out=False,ORN_SPECIAL=True))
#         out = np.mean(out,axis=0)
#         return (out,m,k)
    
    # pool = Pool(nPool)
    info = {'dat_use': dat_use,'S_shape':np.shape(S),'q_shape':np.shape(q),'n_run':n_run}
    save_name = 'm_k_loop_' + dat_use +'_'+ str(n_ex) + '.npy'
    #runOUT = pool.map(easy_dist, zip(mL,kL))
    with Pool(nPool) as pool: # keeps clean
        runOUT = pool.map(easy_dist, zip(mL,kL,SL,qL,dat_useL))
    out = (runOUT,info)
    np.save(save_name,out)