import numpy as np
from hashfunctions import *
import numpy.linalg 


#########################
#### data functions ####
#########################
def generate_hash_space(dIn=500,nIn=5000,m=350,k=17,sparseBinary=False):
    # sparseBinary ex: SB6
    datIn = np.random.uniform(-1,1,(nIn,dIn))
    if sparseBinary:
        #print('Sparse Binary Projection')
        num_sample = sparseBinary[2:]
        MSB6 = np.zeros((dIn,m))
        for row in range(dIn):
            idx = random.sample(range(m), num_sample) 
            M[row,idx] = 1
    else:
        #print('Dense Gaussian Projection')
        M = np.random.randn(dIn,m)
    datHash = np.dot(datIn,M)
    outHash = sparsify_rows(datHash,k)
    return outHash,datHash,datIn

def generate_hash_space_wDat(datIn,m=350,k=17,sparseBinary=False):
    nIn,dIn = np.shape(datIn)
    if sparseBinary:
        M = np.zeros((dIn,m))
        #print('Sparse Binary Projection')
        num_sample = int(sparseBinary[2:])
        MSB6 = np.zeros((dIn,m))
        for row in range(dIn):
            idx = random.sample(range(m), num_sample) 
            M[row,idx] = 1
    else:
        #print('Dense Gaussian Projection')
        M = np.random.randn(dIn,m)
    datHash = np.dot(datIn,M)
    outHash = sparsify_rows(datHash,k)
    return outHash,datHash,datIn

def compute_random_distances(S=None,datHash=None,datIn=None,n_check=1000):
    hasS=False; hasHash=False; hasIn=False; dS = None; ddatHash = None; ddatIn = None
    if np.any(S): dS = np.zeros(n_check); hasS = True; nIn = np.shape(S)[0] 
    if np.any(datHash): ddatHash = np.zeros(n_check); hasHash = True; nIn = np.shape(datHash)[0] 
    if np.any(datIn): ddatIn = np.zeros(n_check); hasIn = True;nIn = np.shape(datIn)[0] 
    assert(hasS or hasHash or hasIn)
    for i in range(n_check):
        ind1,ind2 = np.random.randint(0,nIn,2)
        if hasS:
            dS[i] = hamming_distance(S[ind1],S[ind2])
        if hasHash:
            ddatHash[i] = np.linalg.norm(datHash[ind1]-datHash[ind2])
        if hasIn:
            ddatIn[i] = np.linalg.norm(datIn[ind1]-datIn[ind2])
    return dS,ddatHash, ddatIn



########################
#### BLOOM Filters ####
########################
def standard_bloom(S=None,q=None,bloom=None):
    if np.any(S):
        if np.any(bloom):
            b1 = np.any(S,axis=0)
            bloom = bloom & b1
        else:
            bloom = np.any(S,axis=0)
    if np.any(q):
        this_query = bloom[q==1]
        return this_query
    else: return bloom
    
def timing_bloom(S,O=10,q=None,bloom=None,full_return=False):
    # S, input set
    # O, timing max
    # q, query
    # full_return, get Rx values
    T_ = 0
    if np.any(S):
        n_item,n_dim = np.shape(S)
        bloom = np.ones(n_dim)*np.nan
        for item in range(n_item):
            expire_val = T_+1  # 1,2,...,
            if T_ ==O: expire_val = 0 # when T_ is at max value, wrap time expire to 0.
            bloom[bloom==expire_val] = np.nan
            bloom[S[item]==1] = T_
            T_ +=1
            if T_>O: 
                T_=0
    if np.any(q):
        this_query = bloom[q==1] # might want to loop for multiple queries
        if full_return:
            this_query[this_query>T_] -= O # wrap around values
            return T_ - this_query
        else:
            return ~np.isnan(this_query)
    else:
        return bloom
def stable_bloom(S,O=10,Om = 20,q=None):
    # S, input set
    # O, timing value
    # Om, max value
    # q, query
    if np.any(S):
        n_item,n_dim = np.shape(S)
        bloom = np.ones(n_dim)*0
        for item in range(n_item):
            bloom[bloom>0] -=1 #decay factor
            bloom[S[item]==1] += O # adding factor
            bloom[bloom>Om] = Om # top out
    if np.any(q):
        this_query = bloom[q==1] # might want to loop for multiple queries
        return this_query>0
    else:
        return bloom
def exact_cylinder(S,q,time=14,space=5):
    d = np.array([hamming_distance(s,q) for s in S])
    return np.any(d<=space)

def exact_cone(S,q,space=10):
    n_item,n_dim = np.shape(S)
    d = np.array([hamming_distance(s,q) for s in S])
    d_allow = np.concatenate((np.zeros(n_item-space-1),np.arange(0,space+1)))
    return np.any(d<d_allow)
def objective_fcn(S,q,deg=-1):
    d = np.array([hamming_distance(s,q) for s in S])
    return d_wt(hnorm(d),deg)

def fly_bloom(S=None,q=None,wts=None,rr=0.2):
    # Implement FLY bloom filter
    # rr - recovery rate
    if np.any(S):
        n_row,n_col = np.shape(S)
        if not np.any(wts):
            converge_weight = .887
            wts = np.ones(n_col)*.887
        for i in range(n_row):
            resp,wts = present_odor(S[i,:],wts,rr)
    if np.any(q):
        this_query,_ = present_odor(q,wts)
        return this_query
    else: return resp,wts
def present_odor(odor,wts,rr = .2):
    response = np.dot(wts,odor)/np.sum(odor)
    wt_n = wts
    wt_n[odor==1] = wt_n[odor==1]*.5
    wt_n[odor==0] = wt_n[odor==0] + (1- wt_n[odor==0])*rr
    return response, wt_n
    
    
####################
#### Distances ####
####################
def get_distances(S,q):
    n_item,n_dim = np.shape(S)
    d = np.zeros(n_item)
    for i in range(n_item):
        d[i] = hamming_distance(S[i],q)
    return d
def d_wt(d,degree=-1):
    wts = (np.arange(len(d))+1.0)**degree
    return np.dot(wts,d[-1:None:-1]) # convolution order
hnorm = lambda x: (x-32.32)/32.32



############################
#### Utility Functions ####
############################
def hamming_distance(x1,x2):
    assert(len(x1)==len(x2))
    try:
        return sum(x1^x2) # ^ is xor
    except:
        return sum((x1>0)^(x2>0)) # ^ is xor
def sparsify_rows(space,n_keep):
    nrow,ncol = np.shape(space)
    outHash = np.zeros(np.shape(space))
    inds = np.arange(ncol) # changed to ncol
    for i in range(nrow):
        this_line = space[i]
        val_tuple = list(zip(this_line,inds))
        this_hash = sorted(val_tuple)[-n_keep:]
        this_hash = [ind[1] for ind in this_hash]
        outHash[i,this_hash] = 1
    return outHash
def perterb_hash(brown,n_perterb):
    hashbrown = brown.copy()
    keys = np.where(hashbrown==1)[0]
    assert(len(keys)>=n_perterb)
    key_perterb = np.random.choice(keys,n_perterb,replace=False)
    noMove = keys.copy()
    for i in range(n_perterb):
        randInt = np.random.randint(0,len(hashbrown))
        while any(noMove == randInt):
            randInt = np.random.randint(0,len(hashbrown))
        hashbrown[key_perterb[i]], hashbrown[randInt] = (0,1)
        noMove= np.append(noMove,randInt)
    return hashbrown