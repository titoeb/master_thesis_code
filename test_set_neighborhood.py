import RecModel
import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import time

# Optimal Hyper paramters
distance_function = 'jaccard'
neighborhood_size = 2

# Load data
train_mat = scipy.sparse.load_npz("data/mat_bin_train.npz")
test_mat = scipy.sparse.load_npz("data/mat_bin_test.npz")
n_users, n_items = train_mat.shape

# Train the model
test_neighbor = RecModel.Neighborhood(num_items=n_items, num_users=n_users, nb_size=neighborhood_size)
start = time.time()
test_neighbor.train(train_mat.copy(), distance_function, cores=8)
start = time.time()

# Compute the performance and print it
perf=test_neighbor.eval_topn(test_mat=test_mat.copy(), rand_sampled=1000, topn=np.array([4, 10, 20, 50], dtype=np.int32), random_state=1993, cores=7)
print(f"Perf was: {perf}")

# Compute the coverage and print it!
count_vec = RecModel.test_coverage(test_neighbor, test_mat, 4)
np.save('data/count_vec_neighbor.npy', count_vec)
