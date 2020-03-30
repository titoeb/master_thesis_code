import RecModel
import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import time
import sharedmem

def test_coverage(cls, Train, topN):
    """Testing the coverage of the algorithm:
        It is assumed cls is a object of classes derived from RecModel and is able to rank items with a rank function.
    """
    item_counts = np.zeros(Train.shape[0], dtype=np.int32)

    for user in range(Train.shape[0]):
        start_usr = Train.indptr[user]
        end_usr = Train.indptr[user+1]

        items_to_rank = np.delete(np.arange(Train.shape[1], dtype=np.int32), Train.indices[start_usr:end_usr])
        ranked_items = cls.rank(users=user, items=items_to_rank, topn=topN)

        item_counts[item_counts[ranked_items[:topN]]] += 1
    
    return item_counts

def visualize_coverage(item_counts, Train, ax, topN):
    # Row sums
    item_frequencys = Train.sum(axis=0)

    # Sort the row sums
    sorted_indices = np.argsort(item_frequencys)[::-1]

    # Plot the results
    ax.scatter(np.arange(topN), item_counts[sorted_indices[:topN]])

def nb_test_coverage():
    train_mat = scipy.sparse.load_npz("data/mat_bin_train_test.npz")
    test_mat = scipy.sparse.load_npz("data/mat_bin_validate_test.npz")
    n_users, n_items = train_mat.shape

    test_neighbor = RecModel.Neighborhood(num_items=n_items, num_users=n_users, nb_size=50)
    test_neighbor.train(train_mat.copy(), 'cosine', cores=8)
    perf=test_neighbor.eval_topn(test_mat=test_mat.copy(), rand_sampled=1000, topn=np.array([4, 10, 20, 50], dtype=np.int32), random_state=1993, cores=7)
    print(f"The performance is {perf}")
    
    # Do the coverage evaluation.
    start = time.time()
    coverage = test_coverage(test_neighbor,train_mat, 10)
    print(coverage[:10])

if __name__ == "__main__":
    nb_test_coverage()