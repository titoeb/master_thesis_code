import RecModel
import numpy as np
import scipy.sparse
import multiprocessing

eval_mat = scipy.sparse.load_npz("data/mat_bin_test.npz")

# Fill in from the mlflow run

# Optimal hyperparameters
rand_sampled = 1000
cores = 1

# Define the variable
test_naive_baseline = RecModel.NaiveBaseline(eval_mat.shape[0])

# Compute the performance and write it out
perf_all = test_naive_baseline.eval_topn(test_mat=eval_mat, rand_sampled=1000, topn=np.array([4, 10, 20, 50], dtype=np.int32), random_state=1993)
print(f"The recalls are {perf_all}")

# Compute the coverage
count_vec = RecModel.test_coverage(test_naive_baseline, eval_mat, 4)

# Compute and print the catalog coverage
print((count_vec > 0.0).sum() / len(count_vec))
