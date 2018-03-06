from nearest_neighbor_search import *
from read_data import *
import time
from optparse import OptionParser

# Command line parameters
NUM_KENYON 	= -1		# Number of Kenyon cells
PROJECTION 	= -1		# type of projection from glomeruli onto kenyon cells
HASH_LENGTH = -1 	# hash length

# Dataset parameters
NUM_NNS		= -1		# number of nearest neighbors to validate over
FEATURES 	= -1		# number of features in the dataset
NUM_ODORS 	= -1		# number of odors in the dataset

start = time.time()

usage="usage: %prog [options]"
parser = OptionParser(usage=usage)
parser.add_option("-p", "--projection", action="store", type="string", dest="projection", default="DG",help="type of random projection: DG (dense Gaussian), SB6 (sparse, binary with sampling=6)")
parser.add_option("-y", "--kenyon", action="store", type="int", dest="num_kenyon", default=1000,help="number of kenyon cells (i.e. expansion size)")    
parser.add_option("-w", "--wta", action="store", type="string", dest="wta", default=None,help="type of WTA to perform (top, bottom, rand)")
parser.add_option("-l", "--hash", action="store", type="int", dest="hash_length", default=8,help="length of the hash")
parser.add_option("-d", "--dataset", action="store", type="string", dest="dataset", default="halem",help="name of the dataset")

# read in command line inputs
(options, args) = parser.parse_args()
NUM_REPEATS 	= 50
NUM_KENYON 		= options.num_kenyon
PROJECTION 		= options.projection
HASH_LENGTH 	= options.hash_length
DATASET 		= options.dataset
WTA 			= options.wta   

# check dataset
# GIST data: 10000 images x 960 gist desciptor
if DATASET == "gist10k":
	NUM_ODORS 	= 10000
	FEATURES 	= 960
	D 			= read_generic_data('/home/navlakha/projects/fly_hashing/data/gist_small/gist10k.txt', NUM_ODORS, FEATURES)

# SIFT data: 10000 images x 128 sift descriptor
elif DATASET == "sift10k":
	NUM_ODORS 	= 10000
	FEATURES 	= 128
	D 			= read_generic_data('/home/navlakha/projects/fly_hashing/data/sift_small/sift10k.txt', NUM_ODORS, FEATURES) 

# MNIST data: 10000 images x 784 pixels
elif DATASET == "mnist10k":
	NUM_ODORS 	= 10000
	FEATURES 	= 784
	D 			= read_generic_data('/home/navlakha/projects/fly_hashing/data/mnist/mnist10k.txt', NUM_ODORS, FEATURES) 

# GLOVE data: 10000 words x 300 features
elif DATASET == "glove10k":
	NUM_ODORS 	= 10000
	FEATURES 	= 300
	D 			= read_generic_data('/home/navlakha/projects/fly_hashing/data/glove/glove10k.txt', NUM_ODORS, FEATURES) 

else: assert False

assert D.shape[0] == NUM_ODORS
assert D.shape[1] == FEATURES

numNNs 				= max(10, int(0.02*NUM_ODORS))
accuracy, std_dev 	= nearest_neighbor_search(D, WTA, HASH_LENGTH, PROJECTION, NUM_KENYON, NUM_REPEATS, numNNs)

print "%i\t%s\t%i\t%s\t%i\t%.3f\t%.3f\t%s\t%.2f (mins)" %(FEATURES,PROJECTION,NUM_KENYON,WTA,HASH_LENGTH,accuracy, std_dev,DATASET,(time.time()-start) / 60)



