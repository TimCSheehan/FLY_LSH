import numpy as np
from random import randint, shuffle
import scipy


'''
fly_LSH: Uses the hashing algorithm of the fly to return a tag that is used in nearest neighbor search
Inputs:
odor - the input vector that we are trying to create a hash for
hash_length - length of the tag we want to return
probability_factor - number out of 100 for the sparse matrix
feedback - 0 for none, 1 for winner-take-all, 2 for random
expansion - 0 for none or based on feedback, 1 for based on input length
Output: a hash tag
'''
def fly_LSH(odor, hash_length, probability_factor = 10, feedback = 0, expansion = 0):
    i = len(odor)
    j = hash_length

    # Find the expansion factor (2nd exp = 20*hash_length, 3rd exp = 10*input length)
    if feedback > 0 and expansion == 0:
        j = 20 * hash_length
    elif not expansion == 0:
        j = 10 * i

    M = np.zeros((j,i))

	# create matrix M with a 1 in an element based on probability_factor
    for k in range(0,j):
        for l in range(0,i):
            p = randint(1,100)
            if p <= probability_factor:
                M[k][l] = 1

	# compute tag from hash function			
    tag = np.dot(M,odor) 

	# Perform feedback inhibition on tag
    if feedback == 1:
        tag = WTA(tag, hash_length)
    elif feedback == 2:
        tag = random_selection(tag, hash_length)

    return tag
'''
normal_LSH: Like the fly LSH but uses a gaussian to create the matrix M
'''
def normal_LSH(odor, hash_length):
	i = len(odor)
	j = hash_length

	M = np.random.randn(j,i)
	
	# compute tag from hash function			
	tag = np.dot(M,odor) 

	return tag

'''
WTA: takes the top k numbers in the input tag and returns the new tag
'''
def WTA(tag, k):
	new_tag = sort(tag)
	return new_tag[-k:]

'''
random_selection: picks k random values in the input tag and returns them
'''
def random_selection(tag, k):
	new_tag = shuffle(tag)
	return new_tag[-k:]
