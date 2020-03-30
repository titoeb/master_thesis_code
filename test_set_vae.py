import scipy.sparse
import torch.optim
from RecModel import Mult_VAE
import RecModel
from sklearn.preprocessing import normalize
import numpy as np

train_mat = scipy.sparse.load_npz('data/mat_count_train.npz')
eval_mat = scipy.sparse.load_npz('data/mat_bin_validate.npz')
test_mat = scipy.sparse.load_npz('data/mat_bin_test.npz')

"""
train_mat = scipy.sparse.load_npz('data/mat_count_train_test.npz')
eval_mat = scipy.sparse.load_npz('data/mat_bin_validate_test.npz')
"""

# Create a train test VAE.
# Hyper paramter that are static
batch_size = 1000
epochs = 1912
verbose = 1
rand_sampled = 1000

k = 195
beta = 0.131660
learning_rate = 0.000307
weight_decay_rate = np.NaN
weight_decay = 'no_decay'

dense_layers_encoder_sigma = [379]
dropout_rate_encoder_sigma = 0.506927
dropout_rate_sparse_encoder_sigma = 0.456308
batch_norm_encoder_sigma = True

dense_layers_encoder_mu = [925]
dropout_rate_encoder_mu = 0.552048
dropout_rate_sparse_encoder_mu = 0.245139
batch_norm_encoder_mu = True

dense_layers_decoder = [81]
dropout_rate_decoder = 0.483643
batch_norm_decoder	= False
												
# Create Mult_VAE
test = Mult_VAE(k = k, num_items = train_mat.shape[1], dense_layers_encoder_mu=dense_layers_encoder_mu, dense_layers_encoder_sigma=dense_layers_encoder_sigma, dense_layers_decoder=dense_layers_decoder,
    batch_norm_encoder_mu=batch_norm_encoder_mu, batch_norm_encoder_sigma=batch_norm_encoder_sigma,  batch_norm_decoder=batch_norm_decoder, dropout_rate_decoder=dropout_rate_decoder,
    dropout_rate_encoder_mu=dropout_rate_encoder_mu, dropout_rate_encoder_sigma=dropout_rate_encoder_sigma,  dropout_rate_sparse_encoder_mu=dropout_rate_sparse_encoder_mu,
    dropout_rate_sparse_encoder_sigma=dropout_rate_sparse_encoder_sigma, beta=beta)

if weight_decay == 'decay':
    test.set_optimizer(torch.optim.AdamW(test.parameters(), lr=learning_rate, weight_decay=weight_decay_rate))
elif weight_decay == 'no_decay':
    test.set_optimizer(torch.optim.Adam(test.parameters(), lr=learning_rate))
else:
    raise ValueError(f"'{weight_decay}' is not a valid value for the weight decay")
#test.set_writer('logs/')

test.train(X_train=train_mat.copy(), X_validate=eval_mat.copy(), batch_size=batch_size, epochs=epochs, verbose=verbose)

top_n_on_train = test.eval_topn(test_mat=train_mat.copy(), batch_size=batch_size, topn=np.array([4, 10, 20, 50]), rand_sampled =1000, random_state=None)
print(f"topn on train: {top_n_on_train}")

top_n_on_test = test.eval_topn(test_mat=test_mat.copy(), batch_size=batch_size, topn=np.array([4, 10, 20, 50]), rand_sampled =1000, random_state=None)
print(f"topn on test: {top_n_on_test}")

count_vec = RecModel.test_coverage(test, test_mat, 4)
np.save('data/count_vec_vae.npy', count_vec)