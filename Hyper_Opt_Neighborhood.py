import os
import numpy as np
import scipy.sparse
import yaml
import mlflow
import time
import utils.config_helpers
import os
import RecModel
import datetime
import hydra
import logging
import hyperopt as hp
from functools import partial
import pickle
from RecModel import unfold_config

log = logging.getLogger(__name__)

# Helper functions
def eval_neighborhood(params, cfg, train_mat_bin, train_mat_count, eval_mat, experiment):
    # This function is what Hyperopt is going to optimize (minimize 'loss' value)
    with mlflow.start_run(experiment_id=experiment):

        # flatten the config.
        params = unfold_config(params)

        # Make the neighborhood size to integer.
        params['neighborhood_size'] = max(int(params['neighborhood_size']), 1)
        try:
            params['p'] = int(params['p'])
        except KeyError:
            # give it some dummy value, will be provided if needed.
            params['p'] = 1

        # Log the config
        utils.config_helpers.log_config(dict(cfg.model))        

        n_users, n_items = train_mat_bin.shape

        # Log relevant parameters for this run.
        print("Testing the following hyper parmaters!")
        for key, val in dict(params).items():
            mlflow.log_param(key, val)
            if int(cfg.model.verbose) > 0:
                print(f"{key}: {val}")
        
        # Select the correct matrix to train.
        if params['matrix'] == 'count':
            train_mat = train_mat_count.copy()

        elif params['matrix'] == 'binary':
            train_mat = train_mat_bin.copy()

        else:
            raise ValueError(f"mat can only take values 'count' or 'bin' and not {params['matrix']}")
        
        # Create model
        # Create Mult_VAE
        neighborhood_model = RecModel.Neighborhood(num_items=n_items, num_users=n_users, nb_size=params['neighborhood_size'])

        print(f"start training!")
        start = time.time()
        neighborhood_model.train(X=train_mat.copy(), similarity_function=params['similarity_function'], cores=int(cfg.model.cores), p=params['p'])

        if params['matrix'] == 'count':
            neighborhood_model.weights_only=False
        elif params['matrix'] == 'binary':
            neighborhood_model.weights_only=True
        else:
            raise ValueError(f"mat can only take values 'count' or 'bin' and not {params['matrix']}")

        # Log run-time
        mlflow.log_metric("Runtime", int(round(time.time() - start, 0)))

        # Evaluate model
        perf_all = neighborhood_model.eval_topn(test_mat=eval_mat.copy(),  topn=np.array(cfg.model.top_n_performances, dtype=np.int32), random_state=int(cfg.model.random_state), cores=int(cfg.model.cores))
        
        # Log the performance of the model
        for pos in range(len(cfg.model.top_n_performances)):
            mlflow.log_metric(f"recallAT{cfg.model.top_n_performances[pos]}_of_{cfg.model.rand_sampled}", perf_all[f"Recall@{cfg.model.top_n_performances[pos]}"])
        
        #We will always choose the first topn performance. Hopefully, that is also the smallest is most relevant for us.
        rel_topn_perf = perf_all[f"Recall@{cfg.model.top_n_performances[0]}"]

        log.info(f"Current recallAT{cfg.model.top_n_performances[0]}_of_{cfg.model.rand_sampled} performance was {rel_topn_perf}.")
        loss = -rel_topn_perf
        return {'loss': loss, 'status': hp.STATUS_OK, 'eval_time': time.time()}

def hyper_opt_fmin(space, fun, additional_evals, verbose = 0, trials_path='../trials.p', **kwargs):

    # This is a wrapper around the training process that enables warm starts from file.
    objective = partial(fun, **kwargs)
    
    # Try to recover trials object, else create new one!
    try:
        trials = pickle.load(open(trials_path, "rb"))
        if verbose > 0:
            print(f"Loaded trails from {trials_path}")
    except FileNotFoundError:
        trials = hp.Trials()
        
    # Compute the effect number of new trials that have to be run.
    past_evals = len(trials.losses())
    new_evals = past_evals + additional_evals

    best = hp.fmin(fn = objective, space=space, algo=hp.tpe.suggest,  max_evals = new_evals, trials=trials)
    if verbose > 0:
        print(f"HyperOpt got best loss {trials.best_trial['result']['loss']} with the following hyper paramters: \n{trials.best_trial['misc']['vals']}")
        
    # Store the trials object
    pickle.dump(trials, open(trials_path, "wb"))
    
    return best, trials

# Work around to get the working directory (after release use hydra.utils.get_original_cwd())
from hydra.plugins.common.utils import HydraConfig
def get_original_cwd():
    return HydraConfig().hydra.runtime.cwd

@hydra.main(config_path='configs/config.yaml')
def my_app(cfg):
    # Main 

    # Load mat.
    # Be aware that hydra changes the working directory
    train_mat_bin = scipy.sparse.load_npz(os.path.join(get_original_cwd(), cfg.model.train_mat_bin_path))
    train_mat_count = scipy.sparse.load_npz(os.path.join(get_original_cwd(), cfg.model.train_mat_count_path))
    n_users, n_items = train_mat_bin.shape
    eval_mat = scipy.sparse.load_npz(os.path.join(get_original_cwd(), cfg.model.eval_mat_path))

    train_mat_bin = train_mat_bin.astype(np.float32)
    train_mat_count = train_mat_count.astype(np.float32)
    eval_mat = eval_mat.astype(np.float32)
    eval_mat.sort_indices()
    train_mat_bin.sort_indices()
    train_mat_count.sort_indices()

    # Setup HyperOpt
    space = {
        'similarity_function': hp.hp.choice('similarity_function', [
            {'type': 'jaccard', 'neighborhood_size': hp.hp.lognormal('nb_size_jaccard', 0, 1), 'matrix': hp.hp.choice('matrix_jaccard', ['count', 'binary'])},
            {'type': 'cosine', 'neighborhood_size': hp.hp.lognormal('nb_size_cosine', 0, 1), 'matrix': hp.hp.choice('matrix_cosine', ['count', 'binary'])},
            {'type': 'euclidean', 'neighborhood_size': hp.hp.lognormal('nb_size_euclidean', 0, 1), 'matrix': hp.hp.choice('matrix_euclidian', ['count', 'binary'])},
            {'type': 'correlation', 'neighborhood_size': hp.hp.lognormal('nb_size_correlation', 0, 1), 'matrix': hp.hp.choice('matrix_correlation', ['count', 'binary'])},
            {'type':  'adjusted_correlation', 'neighborhood_size': hp.hp.lognormal('nb_size_adjusted_correlation', 0, 1), 'matrix': hp.hp.choice('matrix_adjusted_correlation', ['count', 'binary'])},
            {'type': 'adjusted_cosine', 'neighborhood_size': hp.hp.lognormal('nb_size_adjusted_cosine', 0, 1), 'matrix': hp.hp.choice('matrix_adjusted_cosine', ['count', 'binary'])},
            {'type': 'cityblock', 'neighborhood_size': hp.hp.lognormal('nb_size_cityblock', 0, 1), 'matrix': hp.hp.choice('matrix_cityblock', ['count', 'binary'])},
            {'type': 'minowski', 'neighborhood_size': hp.hp.lognormal('nb_size_minowski', 0, 1), 'p': hp.hp.lognormal('p', 0, 1), 'matrix': hp.hp.choice('matrix_minowski', ['count', 'binary'])}
        ])}

    # Set up MLFlow experiment
    experiment_name = f"HyperOpt_neighborhood_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M').replace(' ', '_')}"
    experiment = mlflow.create_experiment(experiment_name)

    # Log the config
    log.info("Starting Optimization")
    hyper_opt_fmin(space, eval_neighborhood, cfg.gridsearch.num_evals, verbose = 0, cfg=cfg, train_mat_count=train_mat_count, train_mat_bin=train_mat_bin, eval_mat=eval_mat, experiment=experiment)
    
    log.info("Optimization finished\n")
    # Shutdown VM when grid-search is finished
    if cfg.model.shutdown == 1:
        os.system("shutdown now -h")
    
if __name__ == "__main__":
    my_app()
