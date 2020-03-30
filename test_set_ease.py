import RecModel
import numpy as np
import scipy.sparse
import multiprocessing
import time

train_mat = scipy.sparse.load_npz("data/mat_bin_train.npz")
eval_mat = scipy.sparse.load_npz("data/mat_bin_test.npz")

# Fill in from the mlflow-run
cores= 8
alpha = 1338.409547
verbose=1

ease = RecModel.Ease(num_items=train_mat.shape[1], num_users=train_mat.shape[0])


start = time.time()
ease.train(train_mat.copy(), alpha=alpha, verbose=verbose, cores=cores)
print(f"fitted ease in  {time.time() - start} seconds")

print(ease.eval_topn(test_mat=eval_mat.copy(), topn=np.array([4, 10, 20, 50], dtype=np.int32), rand_sampled =1000, cores=cores))

count_vec = RecModel.test_coverage(ease, eval_mat, 4)
np.save('data/count_vec_ease.npy', count_vec)