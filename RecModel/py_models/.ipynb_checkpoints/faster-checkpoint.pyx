#distutils: extra_compile_args = -fopenmp
#distutils: extra_link_args = -fopenmp
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False


import numpy as np
cimport numpy as np

cpdef np.ndarray[np.double_t, ndim=1] _predict_ease(int[:] X_indptr, int[:] X_idx, float[:] X_data, float[:, :] W, int[:] users, int[:] items):
    cdef:
        int i, j, length, user, item
    if len(users) == 0 or len(items) == 0:
        return np.full(1, 0.0)
    else:
        length = max(len(items), len(users))
        #Pre-allocate output array.
        output = np.empty(length)
        for i in range(length):
            user = users[i % len(users)]
            item = items[i % len(items)]
            for j in range(X_indptr[user], X_indptr[user + 1]):
                output[i] += X_data[j] *  W[X_idx[j], item]
        return output

