from hashfunctions import *
import numpy as np
import random

'''
nearest_neighbor_comparisons: Computes the precision of two nearest neighbor methods
using different hash functions depending on inputcode:
inputcode = 1 for sparse binary,vs. dense gaussian
		  = 2 for WTA vs. random selection using sparse, binary
		  = 3 for WTA expansion vs. random selection expansion
Performs 50 run through picking 1000 inputs and computing their k nearest neighbors from the 
different hashing methods. Returns the mean average precision
'''
def nearest_neighbor_search(input_space, wta, hash_length, projection, num_kenyon, num_repeats, numNNs):
	num_examples 	= input_space.shape[0]  # number of odors
	precision 		= [] # initialize list of precisions
	
	for j in range(0,num_repeats):
		# create random projection matrix
		M = create_projection_matrix(  , num_kenyon, projection)
		
		# Compute Kenyon Cell activity for each example
		tag_space = compute_hash(M, input_space, num_kenyon)

		# Tag compresssion method
		tag_space = compute_wta(tag_space, wta, hash_length)

		# Perform nearest neighbor search on 1000 different odors
		queries = random.sample(range(1,num_examples), 100) 
		for i in queries:

			# Get random odor to use as start node
			input1 = input_space[i]
			input1_tag = tag_space[i]
			
			# compute the distances between current tag and all other tags
			dist_orig,orig_tuple = compute_distances(i, input_space)
			dist_hash,hash_tuple = compute_distances(i, tag_space)
			assert len(dist_orig) == len(dist_hash)
			assert len(hash_tuple) == len(orig_tuple)

			# get true NNs
			true_nns = sorted(orig_tuple)[0:numNNs] # true nearest neighbors
			true_nns = [vals[1] for vals in true_nns] # true nearest neighbor indices

			# get predicted nearest neighbors
			pred_nns = sorted(hash_tuple)[0:numNNs] # predicted nearest neighbors
			pred_nns = [vals[1] for vals in pred_nns] # predicted nearest neighbor indices
			assert len(pred_nns) == len(true_nns)

	        # Compute MAP: https://makarandtapaswi.wordpress.com/2012/07/02/intuition-behind-average-precision-and-map/
	        # E.g.  if the top NUM_NNS results are:   1, 0, 0,   1,   1,   1
	        #       then the MAP is:            avg(1/1, 0, 0, 2/4, 3/5, 4/6)			
			precision.append(compute_AP(pred_nns, true_nns))
		# mean precision

	MAP = np.mean(precision)
	std_dev = np.std(precision)

	return MAP, std_dev # return mean average precision and standard deviation

'''
compute_AP: computes the average precision of the predicted and actual
vectors at position k.
'''
def compute_AP(predicted, actual):
	score = []
	num_hits = 0.0

	for i,p in enumerate(predicted):
		if p in actual:
			num_hits += 1.0
			score.append(num_hits / (i + 1.0))
	score = np.mean(score) if len(score) > 0 else 0
	assert 0.0 <= score <= 1.0
	return score

'''
compute_distances: computes the k nearest neighbors of input1 in input_list
based on their euclidean distance to input1
'''
def compute_distances(index_i, input_list):
	num_examples 	= input_list.shape[0]
	input_i 		= input_list[index_i,:]
	distances 		= [] # list of pairwise distances
	temp_tuples		= [] # list of (dist, odor#)

	# loop over input_list to compute and update distances vector
	for j in range(0, num_examples):
		if index_i == j: continue

		# distance between i and j
		input_j = input_list[j,:]
		dist = np.linalg.norm((input_i - input_j), ord=1)
		temp_tuples.append((dist, j))
		distances.append(dist)

	return distances, temp_tuples



