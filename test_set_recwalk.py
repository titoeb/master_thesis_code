import RecModel
import numpy as np
import scipy.sparse
import multiprocessing
import time

# Load data
train_mat = scipy.sparse.load_npz("data/mat_count_train.npz")
eval_mat = scipy.sparse.load_npz("data/mat_bin_test.npz")

# Optimal hyper parameters
alpha = 1.191168
damping	= np.NaN
eval_method	= 'k_step'
l1_ratio = 4.423678
max_iter = 11
phi	= 0.997003
steps = 3
tol = 0.05244

# Define and train the model
rec = RecModel.Recwalk(num_items=train_mat.shape[1], num_users=train_mat.shape[0], eval_method=eval_method, k_steps=steps, damping=damping)
rec.train(train_mat=train_mat.copy(), phi=phi, alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tolerance=tol, cores=8, verbose=1)

# Evaluate the model
start = time.time()
recall = rec.eval_topn(eval_mat.copy(), rand_sampled=1000, topn=np.array([4, 10, 20, 50], dtype=np.int32), random_state=1993, cores=8)

# Print out evaluation scores
print(f"Recall was {recall}.")

# Compute the coverage of the model.
count_vec = RecModel.test_coverage(rec, eval_mat, 4)
np.save('data/count_vec_recwalk.np', count_vec)
    
