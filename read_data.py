import numpy as np
def read_generic_data(filename, x, y):
    D = np.zeros((x,y))
    with open(filename) as f:
        for line_num,line in enumerate(f):
            
            cols = line.strip().split(",")
            # D[line_num,:] = map(float,cols)
            D[line_num,:] = [float(i) for i in cols] # fix for python 3.4
    return standardize_data(D)

def standardize_data(D):
    [x,y] = D.shape
    for col in range(y): # from xrange for python 2
        D[:,col] += abs(min(D[:,col]))

    for row in range(x):
        D[row,:] = D[row,:] * (100/np.mean(D[row,:])) # 100 is the mean, average firing rate per odor
        #D[row,:] = map(int,D[row,:])
        D[row,:] = [int(i) for i in D[row,:]]
    
    return D
