from nearest_neighbor_search import *
from read_data import *
# Constants
hash_lengths = [2, 4, 8, 12, 16, 20, 24, 28, 32]

# Data paths
sift_small_path = '/home/navlakha/projects/fly_hashing/data/sift_small/sift10k.txt'
gist_small_path = '/home/navlakha/projects/fly_hashing/data/gist_small/gist10k.txt'
glove_path = '/home/navlakha/projects/fly_hashing/data/glove/glove10k.txt'
mnist_path = '/home/navlakha/projects/fly_hashing/data/mnist/mnist10k.txt'

# Load in data
# first number is number of examples, 2nd number is number of features
sift = read_generic_data(sift_small_path, 10000, 128)
gist = read_generic_data(gist_small_path, 10000, 960)
mnist = read_generic_data(mnist_path, 10000, 784)
glove = read_generic_data(glove_path, 10000, 300)

# experiments
sift_LSH_accuracy = np.zeros(len(hash_lengths)) # dense, gaussian normal LSH
sift_fly1_accuracy = np.zeros(len(hash_lengths)) # sparse, binary
sift_fly2_accuracy = np.zeros(len(hash_lengths)) # sparse, binary with expansion and winner-take-all selection
sift_fly3_accuracy = np.zeros(len(hash_lengths)) # sparse, binary with expansion and random selection
sift_fly4_accuracy = np.zeros(len(hash_lengths)) # Expansion based on input WTA
sift_fly5_accuracy = np.zeros(len(hash_lengths)) # Expansion based on input random selection


gist_LSH_accuracy = np.zeros(len(hash_lengths)) # dense, gaussian normal LSH
gist_fly1_accuracy = np.zeros(len(hash_lengths)) # sparse, binary
gist_fly2_accuracy = np.zeros(len(hash_lengths)) # sparse, binary with expansion and winner-take-all selection
gist_fly3_accuracy = np.zeros(len(hash_lengths)) # sparse, binary with expansion and random selection
gist_fly4_accuracy = np.zeros(len(hash_lengths)) # Expansion based on input WTA
gist_fly5_accuracy = np.zeros(len(hash_lengths)) # Expansion based on input random selection


mnist_LSH_accuracy = np.zeros(len(hash_lengths)) # dense, gaussian normal LSH
mnist_fly1_accuracy = np.zeros(len(hash_lengths)) # sparse, binary
mnist_fly2_accuracy = np.zeros(len(hash_lengths)) # sparse, binary with expansion and winner-take-all selection
mnist_fly3_accuracy = np.zeros(len(hash_lengths)) # sparse, binary with expansion and random selection
mnist_fly4_accuracy = np.zeros(len(hash_lengths)) # Expansion based on input WTA
mnist_fly5_accuracy = np.zeros(len(hash_lengths)) # Expansion based on input random selection


glove_LSH_accuracy = np.zeros(len(hash_lengths)) # dense, gaussian normal LSH
glove_fly1_accuracy = np.zeros(len(hash_lengths)) # sparse, binary
glove_fly2_accuracy = np.zeros(len(hash_lengths)) # sparse, binary with expansion and winner-take-all selection
glove_fly3_accuracy = np.zeros(len(hash_lengths)) # sparse, binary with expansion and random selection
glove_fly4_accuracy = np.zeros(len(hash_lengths)) # Expansion based on input WTA
glove_fly5_accuracy = np.zeros(len(hash_lengths)) # Expansion based on input random selection


# find accuracies for the above models at varying hash lengths, using 1000 nearest neighborts
for i in range(0,len(hash_lengths)):
	[sift_fly1_accuracy[i], sift_LSH_accuracy[i]] = nearest_neighbor_comparisons(sift, 1000, hash_lengths[i], 1)
	[sift_fly2_accuracy[i], sift_fly3_accuracy[i]] = nearest_neighbor_comparisons(sift, 1000, hash_lengths[i], 2)
	[sift_fly4_accuracy[i], sift_fly5_accuracy[i]] = nearest_neighbor_comparisons(sift, 1000, hash_lengths[i], 3)

	[gist_fly1_accuracy[i], gist_LSH_accuracy[i]] = nearest_neighbor_comparisons(gist, 1000, hash_lengths[i], 1)
	[gist_fly2_accuracy[i], gist_fly3_accuracy[i]] = nearest_neighbor_comparisons(gist, 1000, hash_lengths[i], 2)
	[gist_fly4_accuracy[i], gist_fly5_accuracy[i]] = nearest_neighbor_comparisons(gist, 1000, hash_lengths[i], 3)

	[mnist_fly1_accuracy[i], sift_LSH_accuracy[i]] = nearest_neighbor_comparisons(mnist, 1000, hash_lengths[i], 1)
	[mnist_fly2_accuracy[i], sift_fly3_accuracy[i]] = nearest_neighbor_comparisons(mnist, 1000, hash_lengths[i], 2)
	[mnist_fly4_accuracy[i], sift_fly5_accuracy[i]] = nearest_neighbor_comparisons(mnist, 1000, hash_lengths[i], 3)

	[glove_fly1_accuracy[i], glove_LSH_accuracy[i]] = nearest_neighbor_comparisons(glove, 1000, hash_lengths[i], 1)
	[glove_fly2_accuracy[i], glove_fly3_accuracy[i]] = nearest_neighbor_comparisons(glove, 1000, hash_lengths[i], 2)
	[glove_fly4_accuracy[i], glove_fly5_accuracy[i]] = nearest_neighbor_comparisons(glove, 1000, hash_lengths[i], 3)


f = open("text_results.txt", "w")
f.write("---------------------\n")
f.write("Sift Results\n")
f.write("Dense, Gaussian\n")
f.write(sift_LSH_accuracy)
f.write("Sparse, binary\n")
f.write(sift_fly1_accuracy)
f.write("WTA with mini expansion\n")
f.write(sift_fly2_accuracy)
f.write("Random selection with mini expansion\n")
f.write(sift_fly3_accuracy)
f.write("Super expansion\n")
f.write(sift_fly4_accuracy)

f.write("------------------\n")
f.write("\n")

f.write("Gist Results\n")
f.write("Dense, Gaussian\n")
f.write(gist_LSH_accuracy)
f.write("Sparse, binary\n")
f.write(gist_fly1_accuracy)
f.write("WTA with mini expansion\n")
f.write(gist_fly2_accuracy)
f.write("Random selection with mini expansion\n")
f.write(gist_fly3_accuracy)
f.write("Super expansion\n")
f.write(gist_fly4_accuracy)

f.write("------------------\n")
f.write("\n")

f.write("MNIST Results\n")
f.write("Dense, Gaussian\n")
f.write(mnist_LSH_accuracy)
f.write("Sparse, binary\n")
f.write(mnist_fly1_accuracy)
f.write("WTA with mini expansion\n")
f.write(mnist_fly2_accuracy)
f.write("Random selection with mini expansion\n")
f.write(mnist_fly3_accuracy)
f.write("Super expansion\n")
f.write(mnist_fly4_accuracy)

f.write("------------------\n")
f.write("\n")

f.write("Glove Results\n")
f.write("Dense, Gaussian\n")
f.write(glove_LSH_accuracy)
f.write("Sparse, binary\n")
f.write(glove_fly1_accuracy)
f.write("WTA with mini expansion\n")
f.write(glove_fly2_accuracy)
f.write("Random selection with mini expansion\n")
f.write(glove_fly3_accuracy)
f.write("Super expansion\n")
f.write(glove_fly4_accuracy)

f.write("------------------\n")
f.write("\n")

f.close()

