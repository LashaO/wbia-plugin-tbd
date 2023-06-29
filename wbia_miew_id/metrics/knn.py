# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neighbors import NearestNeighbors


def predict_k_neigh(db_emb, db_lbls, test_emb, k=5, remove_duplicates=False):
    """Get k nearest solutions from the database for test embeddings (query)
    using k-NearestNeighbors algorithm.
    Input:
        db_emb (float array): database embeddings of size (num_emb, emb_size)
        db_lbls (str or int array): database labels of size (num_emb,)
        test_emb (float array): test embeddings of size (num_emb_t, emb_size)
        k (int): number of predictions to return
    Returns:
        neigh_lbl_un (str or int array): labels of predictions of shape (num_emb_t, k)
        neigh_ind_un (int array): labels of indices of nearest points of shape (num_emb_t, k)
        neigh_dist_un (float array): distances of predictions of shape (num_emb_t, k)
    """
    # Set number of nearest points (with duplicated labels)
    k_w_dupl = min(50, len(db_emb))
    nn_classifier = NearestNeighbors(n_neighbors=k_w_dupl, metric='cosine')
    nn_classifier.fit(db_emb, db_lbls)

    # Predict nearest neighbors and distances for test embeddings
    neigh_dist, neigh_ind = nn_classifier.kneighbors(test_emb)

    # Get labels of nearest neighbors
    neigh_lbl = np.zeros(shape=neigh_ind.shape, dtype=db_lbls.dtype)
    for i, preds in enumerate(neigh_ind):
        for j, pred in enumerate(preds):
            neigh_lbl[i, j] = db_lbls[pred]

    # Remove duplicates
    neigh_lbl_un = []
    neigh_ind_un = []
    neigh_dist_un = []

    for j in range(neigh_lbl.shape[0]):
        indices = np.arange(0, len(neigh_lbl[j]))
        a, b = neigh_lbl[j], indices
        if remove_duplicates:
            a, b = rem_dupl(a, b)
        neigh_lbl_un.append(a[:k])
        neigh_ind_un.append(neigh_ind[j][b][:k].tolist())
        neigh_dist_un.append(neigh_dist[j][b][:k].tolist())

    return neigh_lbl_un, neigh_ind_un, neigh_dist_un


def pred_light(query_embedding, db_embeddings, db_labels, n_results=50):
    """Get k nearest solutions from the database for one query embedding
    using k-NearestNeighbors algorithm.
    """
    neigh_lbl_un, neigh_ind_un, neigh_dist_un = predict_k_neigh(
        db_embeddings, db_labels, query_embedding, k=n_results, remove_duplicates=False
    )

    neigh_lbl_un = neigh_lbl_un[0]
    neigh_dist_un = neigh_dist_un[0]

    ans_dict = [
        {'label': lbl, 'distance': dist} for lbl, dist in zip(neigh_lbl_un, neigh_dist_un)
    ]
    return ans_dict


def rem_dupl(seq, seq2=None):
    """Remove duplicates from a sequence and keep the order of elements.
    Do it in unison with a sequence 2."""
    seen = set()
    seen_add = seen.add
    if seq2 is None:
        return [x for x in seq if not (x in seen or seen_add(x))]
    else:
        a = [x for x in seq if not (x in seen or seen_add(x))]
        seen = set()
        seen_add = seen.add
        b = [seq2[i] for i, x in enumerate(seq) if not (x in seen or seen_add(x))]
        return a, b
