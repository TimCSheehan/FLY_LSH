import numpy as np
from scipy.stats import pearsonr, kendalltau

from hashfunctions import *
import xxhash
mx_hash = 4.5*10**9
mn_hash = 5*10**6

T = np.transpose

def novelty_dist(S,q,dist_met):
    if dist_met == 'cos':
        cs = -(np.max([cos_sim(s,q) for s in S])-1)
    elif dist_met == 'cor':
        cs = -(np.max([cor_sim(s,q) for s in S])-1)
    elif dist_met == 'euc':
        cs = np.min([euc_sim(s,q) for s in S])
    return cs
def cos_sim(a,b):
    return np.dot(a,b)/(np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)))
def euc_sim(a,b):
    return np.sqrt(np.sum((a-b)**2))
def tag_cnt(S,q):
    return np.max([np.sum(s[q==1]) for s in S])

# hash function
def get_hash(v,m,k):
    # using v, get k hashes in range [0,m]
    val = np.sum(v)
    this_hash = (np.array([xxhash.xxh32(str(val+i)).intdigest() for i in range(k)])-mn_hash)/(mx_hash-mn_hash)
    return this_hash*m

def hash_array(A,m,k):
    shp = np.shape(A)
    if len(shp)==2:
        B = list()
        for i in range(shp[0]):
            B.append(get_hash(A[i],m,k))
    elif len(shp)==3: # recursive approach
        B = list()
        for i in range(shp[0]):
            B.append(hash_array(A[i],m,k))
    return  np.array(B).astype(int)

# utility functions 
def min_m(n,eps):
    return int(np.ceil(n*np.log2(np.e)*np.log2(1/eps)))
def opt_k(m,n):
    return int(np.floor(np.log(2)*m/n))
def e_eps(m,k,n):
    # From Broder&Mitenmacher2004
    p = (1.0-1.0/m)**(k*n)
    return (1-p)**k

def cor_fun(x1,x2):
    a1 = pearsonr(x1,x2)[0]
    a2 = kendalltau(x1,x2)[0]
    return np.round((a1,a2),2)

def get_distance_metrics(S,q,m=None,k=None,eps=None,dist_met = 'euc',proj='SB4',app_str=''):
    # must specify either m and k or eps (FP rate)
    n_ex,l_ex,dIn = np.shape(S)
    if not m:
        m = min_m(l_ex,eps)
        k = opt_k(m,l_ex)
    if not eps:
        eps = e_eps(m,l_ex,k)
        
    N = [novelty_dist(S[i],q[i],dist_met) for i in range(n_ex)]
    
    if proj == 'DG':
        M = np.random.randn(dIn,m)
    elif proj[:2]=='SB':
        M = T(create_projection_matrix(dIn,m,proj))
    else:
        assert(0)

        
    Sm = np.matmul(S,M) # n_ex, l_ex, m
    qm = np.matmul(q,M) # n_ex, m
    Sm_LSH = np.argpartition(Sm,-k)[:,:,-k:] # grab the k largest values
    qm_LSH = np.argpartition(qm,-k)[:,-k:]

    Sm_LSH_tag = np.zeros(np.shape(Sm))
    for i,ii in np.ndenumerate(Sm_LSH):      # produce Boolean tags
        Sm_LSH_tag[i[0],i[1],ii] = 1   

    qm_LSH_tag = np.zeros(np.shape(qm))
    for i,ii in enumerate(qm_LSH):
         qm_LSH_tag[i,ii] = 1

    # bloom filter
    Sm_LSH_bloom = np.sum(Sm_LSH_tag,axis=1)>0 
    avg_hash_loc = np.mean(Sm_LSH_bloom,axis=0)
    
    N_LSH_tag = [-(tag_cnt(Sm_LSH_tag[i],qm_LSH_tag[i])/k-1) for i in range(n_ex)]      # Normalized
    N_LSH_bloom = -(np.sum([Sm_LSH_bloom[i,qm_LSH[i]] for i in range(n_ex)],axis=1)/k-1)
    
    Sm_hash = hash_array(S,m,k)
    qm_hash = hash_array(q,m,k)

    Sm_hash_tag = np.zeros(np.shape(Sm))
    for i,ii in np.ndenumerate(Sm_hash):      # produce Boolean tags
        Sm_hash_tag[i[0],i[1],ii] = 1   

    qm_hash_tag = np.zeros(np.shape(qm))
    for i,ii in enumerate(qm_hash):
         qm_hash_tag[i,ii] = 1

    # non-local bloom filter
    Sm_hash_bloom = np.sum(Sm_hash_tag,axis=1)>0 
    avg_hash_loc = np.mean(Sm_hash_bloom,axis=0)

    # closest hash
    N_hash_tag = [-(tag_cnt(Sm_hash_tag[i],qm_hash_tag[i])/k-1) for i in range(n_ex)]     # Normalized
    N_hash_bloom = -(np.sum([Sm_hash_bloom[i,qm_hash[i]] for i in range(n_ex)],axis=1)/k-1)
    
    # get metrics to print
    eps_str = '%.1e' %eps
    nam_str = (app_str + '_d1:' + str(dIn) + '_PROJ:' + proj + '_' + 
        dist_met + '_m:'+ str(m) +'_k:'+str(k)+'_eps:'+eps_str+'  ' )
    
    m_0 = 'OGDIST:%.2f' %np.mean(N) # dist ORN space
    m_1 = cor_fun(N,N_LSH_tag)
    m_2 = cor_fun(N,N_LSH_bloom)
    m_3 = cor_fun(N,N_hash_bloom)
    
    mPRT = str([m_1,m_2,m_3])
    mPRT = ' PROJ: %.2f %.2f, LSHBLOOM: %.2f %.2f, HBLOOM %.2f %.2f' %tuple(np.concatenate([m_1, m_2, m_3]))
    print(nam_str + m_0+ mPRT)
    
    return (nam_str + m_0+ mPRT)
    # return (m_1[0],m_2[0],m_3[0])
    
    