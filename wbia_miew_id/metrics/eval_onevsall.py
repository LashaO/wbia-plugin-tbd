import numpy as np

def eval_onevsall(distmat, q_pids, max_rank=50):
    """Evaluation with one vs all on query set."""
    num_q = distmat.shape[0]

    if num_q < max_rank:
        max_rank = num_q
        print('Note: number of gallery samples is quite small, got {}'.format(num_q))

    indices = np.argsort(distmat, axis=1)
    #    print('indices\n', indices)

    matches = (q_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    #    print('matches\n', matches)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.0  # number of valid query

    for q_idx in range(num_q):
        # remove the query itself
        order = indices[q_idx]
        keep = order != q_idx

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep
        ]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity has only one example
            # => cannot evaluate retrieval
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()

        # P(k)
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        # P(K) * rel(k)
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        # Sigma(P(k) * rel(k))) / number of relevant samples
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    print('Computed metrics on {} examples'.format(len(all_cmc)))

    assert num_valid_q > 0, 'Error: all query identities have one example'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP