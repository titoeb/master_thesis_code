import RecModel
import numpy as np
import scipy.sparse
import multiprocessing

eval_mat = scipy.sparse.load_npz("data/mat_bin_test.npz")

# Fill in from the mlflow run

# Static variables
rand_sampled = 1000
cores = 1

test_naive_baseline = RecModel.NaiveBaseline(eval_mat.shape[0])

perf_all = test_naive_baseline.eval_topn(test_mat=eval_mat, rand_sampled=1000, topn=np.array([4, 10, 20, 50], dtype=np.int32), random_state=1993)
print(f"The recalls are {perf_all}")

count_vec = RecModel.test_coverage(test_naive_baseline, eval_mat, 4)

print((count_vec > 0.0).sum() / len(count_vec))
