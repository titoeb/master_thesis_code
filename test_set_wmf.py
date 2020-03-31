import RecModel
import numpy as np
import scipy.sparse
import multiprocessing

# Load the data
train_mat = scipy.sparse.load_npz("data/mat_bin_train.npz")
eval_mat = scipy.sparse.load_npz("data/mat_bin_validate.npz")
count_mat = scipy.sparse.load_npz("data/mat_count_train.npz")
test_mat = scipy.sparse.load_npz("data/mat_bin_test.npz")

# Set the optimal hyperparameters
iterations = 30
verbose = 1
cores = multiprocessing.cpu_count()
dim = 193
gamma = 809.923459		
stopping_rounds = 2
stopping_percentage = 0.01
seed = 1993
bias=False
weighted=True
min_improvement = 0.01

alpha=21.951598	
beta=4.014181
preprocess = 'log'
rand_sampled = 1000

# Copy the matrices.
train_mat_save = train_mat.copy()
eval_mat_save = eval_mat.copy()
count_mat_save = count_mat.copy()

# Create the class object
test_MF = RecModel.WMF(num_items=train_mat.shape[1], num_users=train_mat.shape[0], dim=dim, gamma=gamma, weighted=weighted, bias=bias)

# Train the model
iter_run = test_MF.train(utility_mat=train_mat.copy(), count_mat=count_mat.copy(), iterations=iterations, verbose=verbose,
 eval_mat=eval_mat.copy(), cores=cores, alpha=alpha, stopping_rounds=stopping_rounds, dtype='float32', min_improvement=min_improvement,
                        pre_process_count=preprocess, beta=beta, preprocess_mat = (preprocess != "None"))

print(f"Training run for {iter_run} rounds.")

# Compute the performance
perf_all = test_MF.eval_topn(test_mat, train_mat, topn=np.array([4, 10, 20, 50]), rand_sampled=rand_sampled, cores=cores)
print(f"The recalls are {perf_all}")

# Compute the coverage
count_vec = RecModel.test_coverage(test_MF, test_mat, 4)
np.save('data/count_vec_wmf.npy', count_vec)
