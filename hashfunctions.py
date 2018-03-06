import numpy as np
import random
import scipy


'''
create_projection_matrix: Computes the projection matrix to go from input space to hash space
Inputs:
odor_space - the input matrix
num_kenyon - number of rows in the matrix
Output: Projection matrix
'''
def create_projection_matrix(odor_space, num_kenyon, projection):
	i = odor_space.shape[1]
	j = num_kenyon

	M = np.zeros((j,i))

	# create projection matrix M based on type of projection (dense random, or sparse binary)
	if projection == "DG":
		M = np.random.randn(j,i)
	elif projection.startswith("SB"):
		num_sample = int(projection[2:])  # SB6 -> 6
		for row in range(0,j):
			idx = random.sample(range(1,i), num_sample) 
			M[row,idx] = 1

			assert sum(M[row,:]) == num_sample
	
	else: assert False

	assert M.shape[0] == num_kenyon
	assert M.shape[1] == odor_space.shape[1]

	return M

'''
Computes the tag space of the given input space with the given projection matrix
'''
def compute_hash(projection_matrix, input_space, num_kenyon):
	num_examples = input_space.shape[0]
	tag_space 	 = np.dot(input_space, np.transpose(projection_matrix))
	assert tag_space.shape[0] == num_examples
	assert tag_space.shape[1] == num_kenyon

	return tag_space


def compute_wta(tags, wta, hash_length):
	num_examples = tags.shape[0]
	num_kenyon   = tags.shape[1]

	# new tag space - same shape as old but with 0's in more places
	new_tags = np.zeros((num_examples, num_kenyon))

	if wta == "random":
		rand_indices = random.sample(range(num_kenyon), hash_length)

	for i in range(num_examples):

		# Take all neurons
		if wta == "all":
			assert num_kenyon == hash_length
			indices = range(0,num_kenyon)

		# Highest firing neurons
		elif wta == "top":
			indices = np.argsort(tags[i,:])[-hash_length:]

		# Lowest firing neurons
		elif wta == "bottom":
			indices = np.argsort(tags[i,:])[:hash_length]

	 	# Random neurons
		elif wta == "random":
			indices = rand_indices

		else: assert False

		new_tags[i,:][indices] = tags[i,:][indices]

	return new_tags 




