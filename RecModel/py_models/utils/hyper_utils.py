import numpy as np
def unfold_config(cfg):
    new = {}
    for key, val in cfg.items():
        if isinstance(val, dict):
            tmp = unfold_config(val)
            res = tmp.pop('type')
            tmp[key] = res
            new.update(tmp)
        else:
            new.update({key: val})
    return new

def test_coverage(cls, Train, topN):
    """Testing the coverage of the algorithm:
        It is assumed cls is a object of classes derived from RecModel and is able to rank items with a rank function.
    """
    item_counts = np.zeros(Train.shape[0], dtype=np.int32)

    for user in range(Train.shape[0]):
        start_usr = Train.indptr[user]
        end_usr = Train.indptr[user+1]

        items_to_rank = np.delete(np.arange(Train.shape[1], dtype=np.int32), Train.indices[start_usr:end_usr])
        ranked_items = cls.rank(users=user, items=items_to_rank, topn=topN).reshape(-1)
        item_counts[ranked_items[:topN]] += 1
    
    return item_counts