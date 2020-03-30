import RecModel
import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import time

if __name__ == "__main__":
    train_mat = scipy.sparse.load_npz("data/mat_bin_train.npz")
    test_mat = scipy.sparse.load_npz("data/mat_bin_test.npz")
    n_users, n_items = train_mat.shape

    test_neighbor = RecModel.Neighborhood(num_items=n_items, num_users=n_users, nb_size=2)
    start = time.time()
    test_neighbor.train(train_mat.copy(), 'jaccard', cores=8)
    
    start = time.time()
    perf=test_neighbor.eval_topn(test_mat=test_mat.copy(), rand_sampled=1000, topn=np.array([4, 10, 20, 50], dtype=np.int32), random_state=1993, cores=7)
    print(f"nb_size was {2} and perf is {perf}")

    count_vec = RecModel.test_coverage(test_neighbor, test_mat, 4)
    np.save('data/count_vec_neighbor.npy', count_vec)