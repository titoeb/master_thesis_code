import scipy.sparse
import numpy as np
import time
import RecModel
import os

# Params
alpha=4.427181
l1_ratio=0.318495
max_iter=27
tol=0.006841
cores=8

test_utility_mat = scipy.sparse.load_npz("data/mat_bin_train.npz")
test_eval_utility_mat = scipy.sparse.load_npz("data/mat_bin_test.npz")

test_utility_mat.sort_indices()
test_utility_mat = test_utility_mat.astype(np.float64)

test_eval_utility_mat.sort_indices()
test_eval_utility_mat = test_eval_utility_mat.astype(np.float64)

n_users, n_items = test_utility_mat.shape

# Create the two class objects
slim = RecModel.Slim(num_items=n_items, num_users=n_users)

# Train the model
start  = time.time()
slim.train(X=test_utility_mat, alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tolerance=tol, cores=4, verbose=1)
print(f"Execution took {(time.time() - start) / 60} minutes")

# Evaluate the model
start = time.time()
recall = slim.eval_topn(test_eval_utility_mat, rand_sampled=1000, topn=np.array([4, 10, 20, 50], dtype=np.int32), random_state=1993, cores=cores)
print(f"Recall was {recall} and execution took {time.time() - start} seconds")


count_vec = RecModel.test_coverage(slim, test_eval_utility_mat, 4)
np.save('data/count_vec_slim.npy', count_vec)